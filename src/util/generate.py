from typing import List, Optional
from tqdm import tqdm

import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel


def generate_fast(
    model: CLIPModel,
    processor: CLIPProcessor,
    data: List[str],
    device: torch.device,
    batch_size: int = 8
    ):
    """
    Fast, parallelized auto-regressive text generation with top-k sampling.
    Our custom implementation.
    """

    all_probs = []
    model = model.to(device)
    
    for text, images in data:
        all_outputs = None
        for batch_start in tqdm(range(0, len(images), batch_size)):
            batch_images = images[batch_start: batch_start + batch_size]

            inputs = processor(text=[text], images=batch_images, return_tensors="pt", padding=True)
            for k, v in inputs.items():
                inputs[k] = v.to(device)

            with torch.no_grad():
                outputs = model(**inputs)
                scores = outputs.logits_per_image
                if all_outputs is None:
                    all_outputs = scores.detach()
                else:
                    all_outputs = torch.cat((all_outputs, scores), 0)

            for k, v in inputs.items():
                inputs[k] = v.to("cpu")
                
        probs = all_outputs.softmax(dim=0).detach()
        all_probs.append(probs)

    return all_probs
