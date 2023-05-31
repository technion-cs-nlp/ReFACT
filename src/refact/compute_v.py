from typing import Dict, List, Tuple
from PIL import Image
from tqdm import tqdm

import numpy as np
import torch
from matplotlib.style import context
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer, CLIPTextModel, AutoTokenizer

from src.refact import repr_tools
from src.util import nethook

from .refact_hparams import ReFACTHyperParams


def get_lookup_idxs(prompts, subject, processor, hparams):
    if subject:
        lookup_idxs = [
            find_fact_lookup_idx(
                prompt, subject, processor.tokenizer, hparams.fact_token, verbose=True
            )
            for i, prompt in enumerate(prompts)
        ]
    else:
        lookup_idxs = [len(processor.tokenizer(prompt)["input_ids"]) - 2  for prompt in prompts]
    return lookup_idxs


def compute_v(
    model: CLIPModel,
    processor: CLIPProcessor,
    request: Dict,
    hparams: ReFACTHyperParams,
    layer: int,
    left_vector: torch.Tensor,
    context_templates: List[str],
    batch_size: int = 4,
    debug: bool = False
) -> torch.Tensor:
    """
    Computes the value (right) vector for the rank-1 update.
    Runs a simple optimization procedure.
    """

    print("Computing right vector (v)")

    # Compile list of rewriting and KL x/y pairs
    rewriting_prompts = [request["prompt"].replace(request["subject"], '{}')] + context_templates
    rewriting_prompts = list(set(rewriting_prompts))
    kl_prompts = request.get("kl_prompts", [])

    # Compute indices of the tokens where the fact is looked up
    context_text_image_lookup_idxs = get_lookup_idxs(rewriting_prompts, request["subject"], processor, hparams)
    kl_text_image_lookup_idxs = get_lookup_idxs(kl_prompts, None, processor, hparams)
    
    rewriting_prompts = [p.format(request["subject"],) for p in rewriting_prompts]    
    num_kl_prompts = len(kl_prompts)

    # Compute re-write inputs
    target_id = 0
    if request["algorithm"] == "contrastive":
        target_images = request.get("new_images", [])
        if "new_image" in request:
            target_images += [request["new_image"]]
        images = target_images + [request["true_image"]] + request.get("alt_images", [])
        context_text_image_prompts = rewriting_prompts
    elif request["algorithm"] == "direct":
        target_images = request.get("new_images", [])
        if "new_image" in request:
            target_images += [request["new_image"]]
        images = target_images + request.get("alt_images", [])
        context_text_image_prompts = rewriting_prompts
    elif request["algorithm"] == "contrastive_text":
        images = request.get("alt_images", [])
        context_text_text_prompts = rewriting_prompts
        target_text_text_prompts = [ request["new_text"], request["true_text"]] + request.get("kl_prompts", [])
        context_text_text_lookup_idxs = context_text_image_lookup_idxs
    elif request["algorithm"] == "direct_text":
        images = request.get("alt_images", [])
        context_text_text_prompts = rewriting_prompts
        target_text_text_prompts = [ request["new_text"]] + request.get("kl_prompts", [])
        context_text_text_lookup_idxs = context_text_image_lookup_idxs
    else:
        raise Exception(f"Unknown algorithm: {request['algorithm']}")

    # Finalize rewrite and loss layers
    loss_layer = max(hparams.v_loss_layer, layer)
    print(f"Rewrite layer is {layer}")
    print(f"Tying optimization objective to {loss_layer}")
    
    # Set up an optimization over a latent vector that, when output at the
    # rewrite layer, i.e. hypothesized fact lookup location, will induce the
    # target token to be predicted at the final layer.
    delta = torch.zeros((model.text_model.config.hidden_size,), requires_grad=True, device="cuda")
    target_init, kl_distr_init = None, None

    # Inserts new "delta" variable at the appropriate part of the computation
    def edit_output_fn(cur_out, cur_layer):
        nonlocal target_init

        if cur_layer == hparams.mlp_module_tmp.format(layer):
            # Store initial value of the vector of interest
            if target_init is None:
                print("Recording initial value of v*")
                # Initial value is recorded for the clean sentence
                target_init = cur_out[0, lookup_idxs[0], :].detach().clone()

            for i, idx in enumerate(lookup_idxs):
                cur_out[i, idx, :] += delta

        return cur_out

    # Optimizer
    opt = torch.optim.Adam([delta], lr=hparams.v_lr)
    nethook.set_requires_grad(False, model)

    if request["algorithm"] in ["contrastive_text", "direct_text"]:
        clean_model = CLIPModel.from_pretrained('openai/clip-vit-large-patch14-336')

    # Execute optimization
    for it in range(hparams.v_max_grad_steps):
        opt.zero_grad()

        # Forward propagation
        context_text_image_scores = None
        kl_text_image_scores = None
        context_text_text_scores = None
        kl_text_text_scores = None # not implemented

        with nethook.TraceDict(
            module=model,
            layers=[
                hparams.layer_module_tmp.format(loss_layer),
                hparams.mlp_module_tmp.format(layer),
            ],
            retain_input=False,
            retain_output=True,
            edit_output=edit_output_fn,
        ) as tr:
            for batch_start in tqdm(range(0, len(images), batch_size)):
                batch_images = images[batch_start: batch_start + batch_size]
                inputs = processor(text=rewriting_prompts + kl_prompts, images=batch_images, return_tensors="pt", padding=True).to("cuda")
                lookup_idxs = context_text_image_lookup_idxs + kl_text_image_lookup_idxs
                out = model(**inputs)
                batch_text_image_scores = out.logits_per_text   
                batch_kl_text_image_scores = batch_text_image_scores[-num_kl_prompts:, :]

                if kl_text_image_scores is None:
                    kl_text_image_scores = batch_kl_text_image_scores
                else:
                    if debug:
                        print("kl_text_image_scores: ", kl_text_image_scores.shape)
                    kl_text_image_scores = torch.cat((kl_text_image_scores, batch_kl_text_image_scores), 1)
                
                if request["algorithm"] in ["contrastive", "direct"]:
                    if request["similarity_metric"] == "cosine":
                        batch_context_text_image_scores = batch_text_image_scores[:-num_kl_prompts, :]
                    elif request["similarity_metric"] == "l2":
                        lookup_idxs = context_text_image_lookup_idxs
                        context_text_inputs = processor(text=context_text_image_prompts, return_tensors="pt", padding=True).to("cuda")
                        image_inputs = processor(images=batch_images, return_tensors="pt", padding=True).to("cuda")
                        
                        image_embeddings = model.get_image_features(**image_inputs)
                        context_text_embeddings = model.get_text_features(**context_text_inputs)
                        
                        logits_per_text = - torch.cdist(
                            torch.unsqueeze(context_text_embeddings, dim=0),
                            torch.unsqueeze(image_embeddings, dim=0)
                        )
                        batch_context_text_image_scores = torch.squeeze(logits_per_text)
                    
                    if debug:
                        print("batch_context_text_image_scores: ", batch_context_text_image_scores.shape)

                    if context_text_image_scores is None:
                        context_text_image_scores = batch_context_text_image_scores
                    else:
                        if debug:
                            print("context_text_image_scores: ", context_text_image_scores.shape)
                        if len(batch_context_text_image_scores.shape) == 1:
                            batch_context_text_image_scores = torch.unsqueeze(batch_context_text_image_scores, dim=1)
                        context_text_image_scores = torch.cat((context_text_image_scores, batch_context_text_image_scores), 1)

            if request["algorithm"] in ["contrastive_text", "direct_text"]:
                context_input = processor(text=context_text_text_prompts, return_tensors="pt", padding=True).to("cuda")
                target_input = processor(text=target_text_text_prompts, return_tensors="pt", padding=True).to("cuda")
                lookup_idxs = context_text_text_lookup_idxs

                device = model.device
                context_embeddings = model.get_text_features(**context_input)
                model = model.to("cpu")
                
                clean_model = clean_model.to(device)
                with torch.no_grad():
                    target_embeddings = clean_model.get_text_features(**target_input)
                clean_model.to("cpu")
                model.to(device)
                
                if request["similarity_metric"] == "cosine":
                    logit_scale = model.logit_scale.exp()
                    context_embeddings = context_embeddings / context_embeddings.norm(p=2, dim=-1, keepdim=True)
                    target_embeddings = target_embeddings / target_embeddings.norm(p=2, dim=-1, keepdim=True)
                    context_text_text_scores = torch.matmul(context_embeddings, target_embeddings.t()) * logit_scale

                elif request["similarity_metric"] == "l2":
                    logits_per_text = - torch.cdist(
						torch.unsqueeze(context_embeddings, dim=0),
						torch.unsqueeze(target_embeddings, dim=0)
					)
                    context_text_text_scores = torch.squeeze(logits_per_text)

        kl_text_image_log_probs = torch.log_softmax(kl_text_image_scores, dim=1)
        if request["algorithm"] in ["contrastive", "direct"]:
            context_text_image_log_probs = torch.log_softmax(context_text_image_scores, dim=1)
        if request["algorithm"] in ["contrastive_text", "direct_text"]:
            context_text_text_log_probs = torch.log_softmax(context_text_text_scores, dim=1)

        # Compute distribution for KL divergence
        if num_kl_prompts:
            kl_log_probs = kl_text_image_log_probs
            if kl_distr_init is None:
                kl_distr_init = kl_log_probs.detach().clone()
            
        # Compute loss on rewriting targets
        if request["algorithm"] == "contrastive":
            loss = 0
            avg_value = 0
            for target_id in range(len(target_images)):
                target_scores = torch.unsqueeze(context_text_image_log_probs[:, target_id], dim=1)
                scores = torch.cat((target_scores, context_text_image_log_probs[:,len(target_images):]), dim=1)
                loss += torch.log_softmax(scores, dim=1)[:, 0]
                avg_value += torch.exp(loss).mean().item()
            avg_name = "prob"
            avg_value /= len(target_images)
        elif request["algorithm"] == "direct":
            loss = 0
            for target_id in range(len(target_images)):
                loss += context_text_image_scores[:, target_id]
            avg_name = "score"
            avg_value = context_text_image_scores[:, 0:len(target_images)].mean().item()
        elif request["algorithm"] == "contrastive_text":
            loss = context_text_text_log_probs[:, target_id]
            avg_name = "prob"
            avg_value = torch.exp(context_text_text_log_probs[:, target_id]).mean(0).item()
        elif request["algorithm"] == "direct_text":
            loss = context_text_text_scores[:, target_id]
            avg_name = "score"
            avg_value = context_text_text_scores[:, target_id].mean(0).item()
        else:
            raise Exception(f"Unknown algorithm: {request['algorithm']}")

        loss = loss.mean(0)
        nll_loss = -loss

        if num_kl_prompts:
            kl_text_image_loss = hparams.kl_factor * torch.nn.functional.kl_div(
                kl_distr_init, kl_text_image_log_probs, log_target=True, reduction="batchmean"
            )

        weight_decay = hparams.v_weight_decay * (
            torch.norm(delta) / torch.norm(target_init) ** 2
        )
        
    
        if num_kl_prompts:
            loss = nll_loss + kl_text_image_loss + weight_decay
            print(
                f"loss {np.round(loss.item(), 3)} = {np.round(nll_loss.item(), 3)} + {np.round(kl_text_image_loss.item(), 15)} + {np.round(weight_decay.item(), 3)} "
                f"avg {avg_name} of new target "
                f"{avg_value}",
                flush=True
            )
        else:
            loss = nll_loss + weight_decay
            print(
                f"loss {np.round(loss.item(), 3)} = {np.round(nll_loss.item(), 3)} + {np.round(weight_decay.item(), 3)} "
                f"avg {avg_name} of new target "
                f"{avg_value}",
                flush=True
            )

        if request["algorithm"] == "contrastive" and loss < 5e-2:
            break
        
        if "contrastive" in request["algorithm"] and avg_value > hparams.v_prob_threshold:
            break

        if it == hparams.v_max_grad_steps - 1:
            break

        # Backpropagate
        loss.backward()
        opt.step()

        # Project within L2 ball
        max_norm = hparams.clamp_norm_factor * target_init.norm()
        if delta.norm() > max_norm:
            with torch.no_grad():
                delta[...] = delta * max_norm / delta.norm()

    target = target_init + delta

    # Retrieve cur_input, the current input to the 2nd MLP layer, and
    # cur_output, the original output of the 2nd MLP layer.
    cur_input, cur_output = get_module_input_output_at_word(
        model.text_model,
        processor.tokenizer,
        layer,
        context_template=request["prompt"].replace(request["subject"], "{}"),
        word=request["subject"],
        module_template=hparams.rewrite_module_tmp,
        fact_token_strategy=hparams.fact_token,
    )

    # Solving the linear system to compute the right vector
    right_vector = (target - cur_output) / torch.dot(cur_input, left_vector)
    print(f"Delta norm: {(target - cur_output).norm().item()}")
    print(
        f"Change in target norm: {target_init.norm().item()} to {target.norm().item()} => {(target.norm() - target_init.norm()).item()}"
    )
    print(f"Division Factor: {torch.dot(cur_input, left_vector).item()}")
    print(f"Right vector norm: {right_vector.norm()}")

    return right_vector


def get_module_input_output_at_word(
    model: CLIPTextModel,
    tok: CLIPTokenizer,
    layer: int,
    context_template: str,
    word: str,
    module_template: str,
    fact_token_strategy: str,
) -> Tuple[torch.Tensor]:
    """
    Retrieves detached representations for a word at the input and
    output of a particular layer module.
    """

    word_repr_args = dict(
        model=model,
        tok=tok,
        layer=layer,
        module_template=module_template,
    )

    subtoken = fact_token_strategy[len("subject_") :]
    l_input, l_output = repr_tools.get_reprs_at_word_tokens(
        track="both",
        subtoken=subtoken,
        context_templates=[context_template],
        words=[word],
        **word_repr_args,
    )

    l_input, l_output = l_input[0], l_output[0]
    return l_input.detach(), l_output.detach()

def find_fact_lookup_idx(
    prompt: str,
    subject: str,
    tok: CLIPTokenizer,
    fact_token_strategy: str,
    verbose=True,
) -> int:
    """
    Computes hypothesized fact lookup index given a sentence and subject.
    """

    ret = None
    if fact_token_strategy == "last":
        ret = -1
    elif (
        "subject_" in fact_token_strategy and fact_token_strategy.index("subject_") == 0
    ):
        ret = repr_tools.get_words_idxs_in_templates(
            tok=tok,
            context_templates=[prompt],
            words=[subject],
            subtoken=fact_token_strategy[len("subject_") :],
        )[0][0]
    else:
        raise ValueError(f"fact_token={fact_token_strategy} not recognized")

    sentence = prompt.format(subject)
    if verbose:
        print(
            f"Lookup index found: {ret} | Sentence: {sentence} | Token:",
            tok.decode(tok(sentence)["input_ids"][ret]),
        )

    return ret
