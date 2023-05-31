from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor

from src.util import nethook
from src.util.generate import generate_fast

from .compute_u import compute_u
from .compute_v import compute_v
from .refact_hparams import ReFACTHyperParams

CONTEXT_TEMPLATES_CACHE = None


def apply_refact_to_model(
    model: CLIPModel,
    processor: CLIPProcessor,
    requests: List[Dict],
    hparams: ReFACTHyperParams,
    copy=False,
    return_orig_weights=False,
    cache_template: Optional[str] = None,
) -> Tuple[CLIPModel, List[str]]:
    """
    Returns a model with the desired changes.

    :param copy: If true, will preserve the original model while creating a new one to edit.
        Note that you are responsible for deallocating the new model's memory to avoid leaks.

    :return: (1) the updated model, (2) an original copy of the weights that changed
    """

    if copy:
        model = deepcopy(model)

    weights_copy = {}

    for i, request in enumerate(requests):
        # Caching is only valid on first request, since the model changes afterwards
        deltas = execute_refact(
            model, processor, request, hparams, (cache_template if i == 0 else None)
        )

        with torch.no_grad():
            for w_name, (delta_u, delta_v) in deltas.items():
                upd_matrix = delta_u.unsqueeze(1) @ delta_v.unsqueeze(0)
                w = nethook.get_parameter(model, w_name)
                upd_matrix = upd_matrix_match_shape(upd_matrix, w.shape)

                if return_orig_weights and w_name not in weights_copy:
                    assert i == 0
                    weights_copy[w_name] = w.detach().clone()

                w[...] += upd_matrix

        print(f"New weights successfully inserted into {list(deltas.keys())}")

    return model, weights_copy


def execute_refact(
    model: CLIPModel,
    processor: CLIPProcessor,
    request: Dict,
    hparams: ReFACTHyperParams,
    cache_template: Optional[str] = None,
) -> Dict[str, Tuple[torch.Tensor]]:
    """
    Executes the Refact update algorithm for the specified update at the specified layer
    """

    # Update target and print info
    request = deepcopy(request)
    print(f"Executing ReFACT algorithm for the update: {request['prompt']}")

    # Retrieve weights that user desires to change
    weights = {
        f"{hparams.rewrite_module_tmp.format(layer)}.weight": nethook.get_parameter(
            model, f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        )
        for layer in hparams.layers
    }
    print("weights.keys(): ", list(weights.keys()))

    # Save old weights for future restoration
    weights_copy = {k: v.detach().clone() for k, v in weights.items()}

    # Update loop: sequentially intervene at each specified layer
    deltas = {}
    hparams.layers = sorted(hparams.layers)
    for layer in hparams.layers:
        left_vector, right_vector = None, None

        # Compute rank-1 update matrix

        left_vector: torch.Tensor = (
            left_vector
            if left_vector is not None
            else compute_u(
                model,
                processor,
                request,
                hparams,
                layer,
                get_context_templates()
            )
        )
        print("Left vector shape:", left_vector.shape)

        right_vector: torch.Tensor = (
            right_vector
            if right_vector is not None
            else compute_v(
                model,
                processor,
                request,
                hparams,
                layer,
                left_vector,
                get_context_templates(),
            )
        )
        print("Right vector shape:", right_vector.shape)

        with torch.no_grad():
            # Determine correct transposition of delta matrix
            weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
            upd_matrix = left_vector.unsqueeze(1) @ right_vector.unsqueeze(0)
            upd_matrix = upd_matrix_match_shape(upd_matrix, weights[weight_name].shape)

            # Update model weights and record desired changes in `delta` variable
            weights[weight_name][...] += upd_matrix
            deltas[weight_name] = (
                left_vector.detach(),
                right_vector.detach(),
            )

    # Restore state of original model
    with torch.no_grad():
        for k, v in weights.items():
            v[...] = weights_copy[k]

    print(f"Deltas successfully computed for {list(weights.keys())}")

    return deltas


def upd_matrix_match_shape(matrix: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """
    GPT-2 and GPT-J have transposed weight representations.
    Returns a matrix that matches the desired shape, else raises a ValueError
    """

    if matrix.shape == shape:
        return matrix
    elif matrix.T.shape == shape:
        return matrix.T
    else:
        raise ValueError(
            "Update matrix computed by ReFACT does not match original weight shape. "
            "Check for bugs in the code?"
        )


def get_context_templates():
    return ["{} in a realistic style portrait image", "{}, a portrait",  "realistic painting of {}", "a current image of {}", "{}, news image", "a beautiful photograph of {}", "realistic drawing of {}", "{}, realistic portrait", "{} in a photo"] 