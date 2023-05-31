import argparse

import numpy as np
import pandas as pd
import torch
import wandb as wandb
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
import open_clip


def parse_args():
    parser = argparse.ArgumentParser(
        prog='Clip edit test',
        description='A script for evaluating editing algorithms with CLIP score')
    parser.add_argument('--file', required=True, help='A file for running and testing the editing from')
    parser.add_argument('--algorithm', choices=['contrastive', 'direct', 'contrastive_text', 'direct_text','baseline', 'oracle'])
    parser.add_argument('--dataset', default='TIME', choices=["TIME", "RoAD", "TIME-TEST"])
    parser.add_argument('--data_split', default='validation', choices=["validation", "test"])
    parser.add_argument('--layer', type=int, default=None)
    parser.add_argument('--model', default='CompVis/stable-diffusion-v1-2')
    parser.add_argument('--clip_model', default='openai/clip-vit-large-patch14-336')

    parser.add_argument('--use_kl_prompts', action='store_true')
    parser.add_argument('--num_kl_prompts', type=int, default=20)
    parser.add_argument('--use_negative_images', action='store_true')
    parser.add_argument('--num_negative_images', type=int, default=200)
    parser.add_argument('--num_editing_images', type=int, default=1)

    parser.add_argument('--edit_layer', type=int, choices=list(range(0,12)))
    parser.add_argument('--v_lr', type=float, default=5e-1)
    parser.add_argument('--v_kl_factor', type=float, default=0)
    parser.add_argument('--v_max_grad_steps', type=int, default=100)
    parser.add_argument('--v_prob_threshold', type=float, default=0.99)
    parser.add_argument('--v_weight_decay_factor', type=float, default=0.1)
    parser.add_argument('--v_similarity_metric', default='l2', choices=["l2", "cosine"])

    return parser.parse_args()


def get_scores(processor, model, output_path, negative, positive, seed, device):
    generated_image_path = f"{output_path}/{negative}/seed_{seed}.png"
    image = Image.open(generated_image_path)
    if type(positive) == list:
        inputs = processor(text=[negative, *positive], images=[image], return_tensors="pt", padding=True)
    else:
        inputs = processor(text=[negative, positive], images=[image], return_tensors="pt", padding=True)
    for k, v in inputs.items():
        inputs[k] = v.to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        scores = outputs.logits_per_image.squeeze()
    return scores

def init_wandb(args):
    wandb.init(project="test_clip_score",
    # Track hyperparameters and run metadata
    config={
        "clip_model": args.clip_model,
        "test_file": args.file,
        "algorithm": args.algorithm,
        "use_kl_prompts": args.use_kl_prompts,
        "num_kl_prompts": args.num_kl_prompts,
        "use_negative_images": args.use_negative_images,
        "num_negative_images": args.num_negative_images,
        "num_editing_images": args.num_editing_images,
        "edit_layer": args.edit_layer,
        "v_lr": args.v_lr,
        "v_kl_factor": args.v_kl_factor,
        "v_max_grad_steps": args.v_max_grad_steps,
        "v_prob_threshold": args.v_prob_threshold,
        "v_weight_decay_factor": args.v_weight_decay_factor,
        "similarity_metric": args.v_similarity_metric,
    })


missing = []

def get_scores_new(preprocess_val, tokenizer, model, output_path, prompt, old, new, seed, device):

    generated_image_path = f"{output_path}/{prompt}/seed_{seed}.png"
    try:
        image = Image.open(generated_image_path)
    except:
        print(generated_image_path)
        missing.append(generated_image_path)
    image = preprocess_val(image).unsqueeze(0).to(device)
    if type(new) == list:
        text = tokenizer([old, *new]).to(device)
    else:
        text = tokenizer([old, new]).to(device)

    with torch.no_grad(), torch.cuda.amp.autocast():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    return text_probs[0]


def main():

    args = parse_args()
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

    init_wandb(args)

    # model = CLIPModel.from_pretrained(args.clip_model).to(device).eval()
    # processor = CLIPProcessor.from_pretrained(args.clip_model)


    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
        'hf-hub:laion/CLIP-ViT-bigG-14-laion2B-39B-b160k')
    model = model.to(device)
    tokenizer = open_clip.get_tokenizer('hf-hub:laion/CLIP-ViT-bigG-14-laion2B-39B-b160k')

    valid_set = pd.read_csv(args.file)

    if not args.use_kl_prompts:
        args.num_kl_prompts = ""
    else:
        args.num_kl_prompts = "_" + str(args.num_kl_prompts)
    if not args.use_negative_images:
        args.num_negative_images = ""
    else:
        args.num_negative_images = "_" + str(args.num_negative_images)

    all_efficacy = []
    all_generality = []
    all_generality_75 = []
    all_generality_90 = []
    all_specificity = []
    all_validity = []
    all_old = []
    all_new = []
    for i, raw_row in valid_set.iterrows():
        row = dict()
        for k,v in raw_row.items():
            row[k.lower()] = v.lower()
        print(row)

        efficacy = []
        generality = []
        generality_75 = []
        generality_90 = []
        specificity = []
        validity = []
        output_path = None
        
        if args.dataset == "TIME" or args.dataset == 'TIME-TEST':
            path = "old"
        elif args.dataset == "RoAD":
            path = "prompt"

        for seed in range(0, 25):
            if args.algorithm == 'baseline':
                output_path = f"./images/results/{args.dataset}/{args.data_split}/baseline/{row[path]}"
            elif args.algorithm == 'oracle':
                output_path = f"./images/results/{args.dataset}/{args.data_split}/oracle/{row[path]}"
                print(output_path)
            elif args.algorithm in ["contrastive", "direct", 'contrastive_text', 'direct_text']:
                output_path = f"./images/results/{args.dataset}/{args.data_split}/{args.algorithm}/use_kl_prompts_{args.use_kl_prompts}{args.num_kl_prompts}_kl_factor_{args.v_kl_factor}/use_negative_images_{args.use_negative_images}{args.num_negative_images}/v_prob_threshold_{args.v_prob_threshold}/v_max_grad_steps_{args.v_max_grad_steps}/v_lr_{args.v_lr}/similarity_metric_{args.v_similarity_metric}/edit_layer_{args.edit_layer}/{row[path]}"

            scores = get_scores_new(preprocess_val, tokenizer, model, output_path, row[path], row['old'], row['new'], seed, device)

            success_indicator = (scores[1] > scores[0]).item()
            if success_indicator:
                efficacy += [1]

            else:
                efficacy += [0]

            ctr_generality = 0
            ctr_generality90 = 0
            ctr_generality75 = 0
            for i in range(1, 6):
                if args.dataset == "TIME" or args.dataset == 'TIME-TEST':
                    positive_old = f'positive{i}'
                    positive_new = f'gt{i}'
                elif args.dataset == "RoAD":
                    positive_old = f'positive_old{i}'
                    positive_new = f'positive_new{i}'
                
                scores = get_scores_new(preprocess_val, tokenizer, model, output_path, row[f'positive{i}'], row[positive_old], row[positive_new], seed,
                                        device)
                success_indicator = (scores[1] > scores[0]).item()
                if success_indicator:
                    ctr_generality += 1

                success_indicator = scores[1] > 0.9
                if success_indicator:
                    ctr_generality90 += 1

                success_indicator = scores[1] > 0.75
                if success_indicator:
                    ctr_generality75 += 1

            generality.append(ctr_generality / 5)
            generality_90.append(ctr_generality90 / 5)
            generality_75.append(ctr_generality75 / 5)

            ctr_specificity = 0
            for i in range(1, 6):
                # specificity - oracle is like the baseline
                if args.dataset == "TIME" or args.dataset == 'TIME-TEST':
                    negative_new = f'gn{i}'
                elif args.dataset == "RoAD":
                    negative_new = f'negative_new{i}'

                if args.algorithm == 'oracle':
                    output_path = f"./images/results/{args.dataset}/{args.data_split}/baseline/{row[path]}"

                scores = get_scores_new(preprocess_val, tokenizer, model, output_path, row[f'negative{i}'], row[f'negative{i}'], row[negative_new], seed, device)
                success_indicator = (scores[1] < scores[0]).item()
                
                if success_indicator:
                    ctr_specificity += 1
            
            specificity.append(ctr_specificity / 5)

            # validity is only computed for TIMED
            if args.dataset == "TIME" or args.dataset == 'TIME-TEST':
                negatives = [row[f'negative1'], row[f'negative2'], row[f'negative3'],  row[f'negative4'], row[f'negative5']]
                negatives_for_validity = []
                for ex in negatives:
                    if row['old'].lower().split()[-1] not in ex.lower():
                        negatives_for_validity.append(ex)
                # print(negatives_for_validity)
                
                scores = get_scores_new(preprocess_val, tokenizer, model, output_path, row[f'old'], negatives_for_validity,
                                        seed, device)
                scores = torch.softmax(scores, dim=0)
                tmp = [row[f'old']] + negatives_for_validity
                # print(seed ,tmp[torch.argmax(scores).item()])
                
                if torch.argmax(scores) == 0:
                    validity.append(1)
                else:
                    validity.append(0)

        if missing:
            print("*"*50)
            print(missing)
            raise Exception
        print(f"Stats for {row['old']} -> {row['new']}:")
        print(f"Efficacy: {np.mean(efficacy)} +- {np.std(efficacy)}")
        print(f"Generality: {np.mean(generality)} +- {np.std(generality)}")
        print(f"Generality_75: {np.mean(generality_75)} +- {np.std(generality_75)}")
        print(f"generality_90: {np.mean(generality_90)} +- {np.std(generality_90)}")
        print(f"Specificity: {np.mean(specificity)} +- {np.std(specificity)}")
        if args.dataset == "TIME" or args.dataset == 'TIME-TEST':
            print(f"Validity: {np.mean(validity)} +- {np.std(validity)}")

        all_efficacy.append(efficacy)
        all_generality.append(generality)
        all_generality_75.append(generality_75)
        all_generality_90.append(generality_90)
        all_specificity.append(specificity)
        all_old.append(row['old'])
        all_new.append(row['new'])
        
        if args.dataset == "TIME" or args.dataset == 'TIME-TEST':
            all_validity.append(validity)

        # for i in range(1, 6):
        #     generated_image_path = f"{output_path}/{row['old']}/{row[f'positive{i}']}/seed_{args.seed}"
        #     image = Image.open(generated_image_path)
        #     inputs = processor(text=[row[f'positive{i}'], row[f'gt{i}']], images=[image], return_tensors="pt", padding=True)
        #
        #     generated_image_path = f"{output_path}/{row['old']}/{row[f'negative{i}']}/seed_{args.seed}"
        #     inputs = processor(text=[row[f'negative{i}'], row[f'gn{i}']], images=[image], return_tensors="pt", padding=True)
        #     image = Image.open(generated_image_path)

    all_efficacy = np.array(all_efficacy)
    all_generality = np.array(all_generality)
    all_generality_75 = np.array(all_generality_75)
    all_generality_90 = np.array(all_generality_90)
    all_specificity = np.array(all_specificity)
    all_validity = np.array(all_validity)

    print(all_generality.mean(axis=0))

    print("Mean efficacy:", np.mean(all_efficacy))
    print("Mean generality:", np.mean(all_generality))
    print("Mean generality90:", np.mean(all_generality_90))
    print("Mean generality75:", np.mean(all_generality_75))
    print("Mean specificity:", np.mean(all_specificity))
    if args.dataset == "TIME" or args.dataset == 'TIME-TEST':
        print("Mean validity:", np.mean(all_validity))

    wandb.summary['efficacy'] = np.mean(all_efficacy)
    wandb.summary['efficacy_std'] = np.std(all_efficacy.mean(axis=0))
    wandb.summary['generality'] = np.mean(all_generality)
    wandb.summary['generality_std'] = np.std(all_generality.mean(axis=0))
    wandb.summary['generality_90'] = np.mean(all_generality_90)
    wandb.summary['generality_90_std'] = np.std(all_generality_90.mean(axis=0))
    wandb.summary['generality_75'] = np.mean(all_generality_75)
    wandb.summary['generality_75_std'] = np.std(all_generality_75.mean(axis=0))
    wandb.summary['specificity'] = np.mean(all_specificity)
    wandb.summary['specificity_std'] = np.std(all_specificity.mean(axis=0))
    if args.dataset == "TIME" or args.dataset == 'TIME-TEST':
        wandb.summary['validity'] = np.mean(all_validity)
        wandb.summary['validity_std'] = np.std(all_validity.mean(axis=0))

    result_dict = {
        'old': all_old,
        'new': all_new,
        'efficacy': all_efficacy.mean(axis=1),
        'generality': all_generality.mean(axis=1),
        'generality90': all_generality_90.mean(axis=1),
        'generality75': all_generality_75.mean(axis=1),
        'specificity': all_specificity.mean(axis=1),
    }
    if args.dataset == "TIME" or args.dataset == 'TIME-TEST':
        result_dict['validity'] = all_validity.mean(axis=1)

    df = pd.DataFrame.from_dict(result_dict)
    table = wandb.Table(dataframe=df)
    wandb.log({"table": table})


if __name__ == "__main__":
    main()
