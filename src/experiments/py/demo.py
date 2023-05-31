import os
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from transformers import CLIPProcessor, CLIPModel

from src.refact import ReFACTHyperParams, apply_refact_to_model

from src.util import nethook
from src.util.generate import generate_fast
from src.util.globals import *


def demo_model_editing(
    model: CLIPModel,
    processor: CLIPProcessor,
    requests: List[Dict],
    neighborhood_data: List[str],
    device: torch.device,
    alg_name: str = "ReFACT",
    hparams=None,
) -> Tuple[CLIPModel, Dict[str, torch.Tensor]]:
    """
    Applies the selected model editing algorithm. Generates text both before and after
    for comparison of model behavior. Returns the updated model and the original values of
    weights that were changed.
    """

    nethook.set_requires_grad(True, model)

    
    RewritingParamsClass = ReFACTHyperParams
    apply_method = apply_refact_to_model
    hparams_prefix, hparams_suffix = "ReFACT", ""
    
    if not hparams:
        params_name = (
            HPARAMS_DIR
            / hparams_prefix
            / f"{model.config._name_or_path.replace('/', '_')}{hparams_suffix}.json"
        )

        print_loud(f"Retrieving {alg_name} hyperparameters")
        print("Loading from", params_name)
        hparams = RewritingParamsClass.from_json(params_name)
        
    print(hparams)

    print_loud("Generating pre-update scores")
    pre_update_scores = generate_fast(model, processor, neighborhood_data, device)
    for i, scores in enumerate(pre_update_scores):
        text, images = neighborhood_data[i]
        # path_to_remove = "/".join(image_paths[0].split("/")[:-1])
        print(f"text: {text}:")
        scores = [i for i in zip(scores, images)]
        scores.sort(key=lambda x: -x[0].item())
        for score, image in scores[:10]:
            if image == images[0]:
                print(f"\tTrue: {score.item()}")
            elif image == images[1]:
                print(f"\tNew: {score.item()}")
            else:
                print(f"\tA COCO image: {score.item()}")

    print_loud(f"Applying {alg_name} to model")
    model_new, orig_weights = apply_method(
        model,
        processor,
        requests,
        hparams,
        return_orig_weights=True,
    )

    print_loud("Generating post-update scores")
    post_update_scores = generate_fast(model, processor, neighborhood_data, device)
    for i, scores in enumerate(post_update_scores):
        text, images = neighborhood_data[i]
        # path_to_remove = "/".join(image_paths[0].split("/")[:-1])
        print(f"text: {text}:")
        scores = [i for i in zip(scores, images)]
        scores.sort(key=lambda x: -x[0].item())
        for score, image in scores[:10]:
            if image == images[0]:
                print(f"\tTrue: {score.item()}")
            elif image == images[1]:
                print(f"\tNew: {score.item()}")
            else:
                print(f"\t{image}: {score.item()}")


    # print_loud("Summarizing differences")
    # for i, (prompt, pre, post) in enumerate(
    #     zip(generation_prompts, pre_update_text, post_update_text)
    # ):
    #     if i > 0:
    #         print("".join(["-" for _ in range(10)]))

    #     prompt_str = "[Prompt]:"
    #     pre_str = f"[Pre-MEMIT]:"
    #     post_str = f"[Post-MEMIT]:"
    #     pad_to = 1 + max(len(prompt_str), len(pre_str), len(post_str))

    #     for s, t in zip([prompt_str, post_str, pre_str], [prompt, post, pre]):
    #         print(s.ljust(pad_to), t)

    return model_new, orig_weights


def print_loud(x, pad=3):
    """
    Prints a string with # box for emphasis.

    Example:
    ############################
    #                          #
    #  Applying ReFACT to model  #
    #                          #
    ############################
    """

    n = len(x)
    print()
    print("".join(["#" for _ in range(n + 2 * pad)]))
    print("#" + "".join([" " for _ in range(n + 2 * (pad - 1))]) + "#")
    print(
        "#"
        + "".join([" " for _ in range(pad - 1)])
        + x
        + "".join([" " for _ in range(pad - 1)])
        + "#"
    )
    print("#" + "".join([" " for _ in range(n + 2 * (pad - 1))]) + "#")
    print("".join(["#" for _ in range(n + 2 * pad)]))


class StopExecution(Exception):
    def _render_traceback_(self):
        pass


def stop_execution():
    raise StopExecution
