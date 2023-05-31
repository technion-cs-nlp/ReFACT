import os
from pathlib import Path
from typing import Dict, List

import torch
from transformers import CLIPModel, CLIPProcessor, CLIPTextModel, CLIPTokenizer

from src.refact import repr_tools
from src.util.globals import *

from .layer_stats import layer_stats
from .refact_hparams import ReFACTHyperParams

# Cache variables
inv_mom2_cache = {}


def get_inv_cov(
    model: CLIPTextModel,
    tok: CLIPTokenizer,
    stats_dir: str,
    layer_name: str,
    mom2_dataset: str,
    mom2_n_samples: str,
    mom2_dtype: str,
) -> torch.Tensor:
    """
    Retrieves covariance statistics, then computes the algebraic inverse.
    Caches result for future use.
    """

    # global inv_mom2_cache
    # inv_mom2_cache = dict()

    model_name = model.config._name_or_path.replace("/", "_")
    # key = (model_name, layer_name)

    # if key not in inv_mom2_cache:
    print(
        f"Retrieving inverse covariance statistics for {model_name} @ {layer_name}. "
        f"The result will be cached to avoid repetitive computation."
    )
    stat = layer_stats(
        model,
        tok,
        layer_name,
        stats_dir,
        mom2_dataset,
        to_collect=["mom2"],
        sample_size=mom2_n_samples,
        precision=mom2_dtype,
    )
    key = torch.inverse(
        stat.mom2.moment().to("cuda")
    ).float()  # Cast back to float32

    return key


def compute_u(
    model: CLIPModel,
    processor: CLIPProcessor,
    request: Dict,
    hparams: ReFACTHyperParams,
    layer: int,
    context_templates: List[str],
    ) -> torch.Tensor:
    """
    Computes the right vector used in constructing the rank-1 update matrix.
    """

    print("Computing left vector (u)...")

    # Compute projection token
    word_repr_args = dict(
        model=model.text_model,
        tok=processor.tokenizer,
        layer=layer,
        module_template=hparams.rewrite_module_tmp,
        track="in",
    )

    print("context_templates: ", context_templates)
    # print(f"Selected u projection object {word}")
    cur_repr = repr_tools.get_reprs_at_word_tokens(
        context_templates=context_templates,
        words=[request["subject"] for _ in range(len(context_templates))],
        subtoken=hparams.fact_token[len("subject_") :],
        **word_repr_args,
    ).mean(0)

    print(cur_repr.shape, flush=True)
    
    # Apply inverse second moment adjustment
    u = cur_repr
    
    if hparams.mom2_adjustment:
        u = get_inv_cov(
            model.text_model,
            processor.tokenizer,
            hparams.stats_dir,
            hparams.rewrite_module_tmp.format(layer),
            hparams.mom2_dataset,
            hparams.mom2_n_samples,
            hparams.mom2_dtype,
        ) @ u.unsqueeze(1)
        u = u.squeeze()

    return u / u.norm()
