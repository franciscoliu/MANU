import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from transformers import AutoProcessor, Idefics2ForConditionalGeneration
from transformers import BitsAndBytesConfig
import os
import sys
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from peft import PeftModel
sys.path.append(('../'))
sys.path.append(('../../'))
from datasets import load_dataset, Dataset
import argparse
import inspect
from PIL import Image
import torch
# from transformers import BitsAndBytesConfig, LlavaForConditionalGeneration, AutoProcessor, get_scheduler, AdamW, \
#     LlavaNextForConditionalGeneration, LlavaNextProcessor, Idefics2ForConditionalGeneration, \
#     MllamaForConditionalGeneration, MllamaProcessor, AutoTokenizer
from data_process.data_preprocess import Vanilla_LLaVA_Dataset, train_collate_fn_llava, train_collate_fn, \
    train_collate_fn_idefics, LLAVA_multimodal_Dataset, LLAVA_unimodal_Dataset
import matplotlib.pyplot as plt
from PIL import Image
from accelerate import Accelerator
from transformers import AutoProcessor
from prune_utility import register_feedforward_hooks, \
    collect_feedforward_activations_single_batch, compute_all_importance_scores, \
    compute_combined_scores, collect_feedforward_activations_multiple_batches, compute_top_k_pruning_mask
from activation_collect import ActivationCollector
from transformers import LlavaForConditionalGeneration
import torch

def load_model_and_processor(model_id):
    if model_id.startswith("llava"):
        print("Loading LLAVA model...")
        model = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            cache_dir="/afs/crc.nd.edu/group/dmsquare/vol1/zliu29/mllm_unlearn/model/llava-1.5-7b-hf",
        )
        processor = AutoProcessor.from_pretrained(model_id)
        processor.tokenizer.padding_side = "right"
        processor.tokenizer.add_tokens(["<image>", "<pad>"], special_tokens=True)

    elif model_id.startswith("HuggingFaceM4"):
        print("Loading idefics2 model...")
        model = Idefics2ForConditionalGeneration.from_pretrained(
            "HuggingFaceM4/idefics2-8b",
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
            cache_dir="/afs/crc.nd.edu/group/dmsquare/vol1/zliu29/mllm_unlearn/model/idfics2-8b",
        )
        processor = AutoProcessor.from_pretrained(
            "HuggingFaceM4/idefics2-8b",
            do_image_splitting=False
        )
        processor.tokenizer.padding_side = "right"
        processor.tokenizer.add_tokens(["<image>", "<pad>"], special_tokens=True)

    else:
        raise ValueError("Model ID not recognized or not supported. Please provide a valid model ID.")

    return model, processor


if __name__ == "__main__":
    # Load model and processor
    parser = argparse.ArgumentParser(description="Fine-tune different models")
    parser.add_argument("--model_id", type=str, required=True, help="Pretrained model ID")
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory to save the model")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--max_length", type=int, default=384, help="Maximum sequence length")
    parser.add_argument('--forget_ratio', type=int, default=5, help='Path to real person image folder.')
    parser.add_argument("--trainer", type=bool, default=False, help="Use HuggingFace Trainer")
    args = parser.parse_args()
    model_id = args.model_id
    model, processor = load_model_and_processor(model_id)
    print(model)

    forget_folder = os.path.join(args.data_dir, f"forget_{args.forget_ratio}")
    retain_folder = os.path.join(args.data_dir, f"retain_{100 - args.forget_ratio}")
    print("Forget Folder: ", forget_folder)
    print("Retain Folder: ", retain_folder)

    # Define paths to the Parquet files for "forget" and "retain" datasets
    forget_parquet_file = os.path.join(forget_folder, f"train-00000-of-00001.parquet")
    retain_parquet_file = os.path.join(retain_folder, f"train-00000-of-00001.parquet")

    # Load DataLoader
    forget_df = pd.read_parquet(forget_parquet_file)
    retain_df = pd.read_parquet(retain_parquet_file)

    # df = pd.read_parquet("/afs/crc.nd.edu/user/z/zliu29/Private/2024_fall/mllm_npo/data/MLLMU-Bench/ft_Data/train-00000-of-00001.parquet")
    # print("Train Dataframe: ")
    # print(df.head())
    # print(df.columns)
    #
    # print("Forget Dataframe: ")
    # print(forget_df.head())
    # print(forget_df.columns)
    #
    # print("Retain Dataframe: ")
    # print(retain_df.head())
    # print(retain_df.columns)



    if model_id.startswith("llava"):
        llava_multimodal_forget_dataset = LLAVA_multimodal_Dataset(df=forget_df)
        llava_unimodal_forget_dataset = LLAVA_unimodal_Dataset(df=forget_df)
        llava_multimodal_retain_dataset = LLAVA_multimodal_Dataset(df=retain_df)
        llava_unimodal_retain_dataset = LLAVA_unimodal_Dataset(df=retain_df)
        # train_collate_fn_llava(llava_unimodal_dataset, processor, args, "unimodal")
        forget_multimodal_dataloader = DataLoader(
            llava_multimodal_forget_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=lambda x: train_collate_fn_llava(x, processor, args, "multimodal")
        )
        forget_unimodal_dataloader = DataLoader(
            llava_unimodal_forget_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=lambda x: train_collate_fn_llava(x, processor, args, "unimodal")
        )
        retain_multimodal_dataloader = DataLoader(
            llava_multimodal_retain_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=lambda x: train_collate_fn_llava(x, processor, args, "multimodal")
        )
        retain_unimodal_dataloader = DataLoader(
            llava_unimodal_retain_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=lambda x: train_collate_fn_llava(x, processor, args, "unimodal")
        )
    # elif model_id.startswith("HuggingFaceM4"):
    #     train_dataloader = DataLoader(
    #         dataset,
    #         batch_size=args.batch_size,
    #         shuffle=True,
    #         collate_fn=lambda x: train_collate_fn_idefics(x, processor, args)
    #     )
    else:
        raise ValueError("Model ID not recognized or not supported. Please provide a valid model ID.")

    # Prepare the accelerator
    accelerator = Accelerator()
    model, forget_multimodal_dataloader, forget_unimodal_dataloader, retain_multimodal_dataloader, retain_unimodal_dataloader = accelerator.prepare(
        model,
        forget_multimodal_dataloader,
        forget_unimodal_dataloader,
        retain_multimodal_dataloader,
        retain_unimodal_dataloader,
    )

    # Before processing the dataloader, calculate and print the total number of batches
    total_batches_forget_multimodal = len(forget_multimodal_dataloader)
    total_batches_forget_unimodal = len(forget_unimodal_dataloader)
    total_batches_retain_multimodal = len(retain_multimodal_dataloader)
    total_batches_retain_unimodal = len(retain_unimodal_dataloader)

    print(f"Total number of batches in forget multimodal dataloader: {total_batches_forget_multimodal}")
    print(f"Total number of batches in forget unimodal dataloader: {total_batches_forget_unimodal}")
    print(f"Total number of batches in retain multimodal dataloader: {total_batches_retain_multimodal}")
    print(f"Total number of batches in retain unimodal dataloader: {total_batches_retain_unimodal}")

    # Instantiate activation collectors for both forget and retain sets
    forget_multimodal_collector = ActivationCollector()
    register_feedforward_hooks(model, forget_multimodal_collector, model_type="Llava")

    #######################  === Forget Set: Multimodal  ####################### ===
    print("Testing single batch for forget multimodal...")
    forget_multimodal_activations = collect_feedforward_activations_multiple_batches(
        model,
        forget_multimodal_collector,
        forget_multimodal_dataloader,
        modality="multimodal",
        model_type="Llava",
        device=accelerator.device,
    )
    # print("Collected layers for Llava (forget multimodal):",forget_multimodal_activations)
    print("Collected layers for Llava (forget multimodal):",
          forget_multimodal_collector.list_collected_layers(modality="multimodal"))
    forget_multimodal_all_scores = compute_all_importance_scores(forget_multimodal_collector, "multimodal")
    print("Forget multimodal scores:", forget_multimodal_all_scores)
    forget_multimodal_collector.clear_activations()

    ####################### === Forget Set: Unimodal #######################  ===
    forget_unimodal_collector = ActivationCollector()
    register_feedforward_hooks(model, forget_unimodal_collector, model_type="Llava")

    print("Testing single batch for forget unimodal...")
    forget_unimodal_activations = collect_feedforward_activations_multiple_batches(
        model,
        forget_unimodal_collector,
        forget_unimodal_dataloader,
        modality="unimodal",
        model_type="Llava",
        device=accelerator.device,
    )
    print("Collected layers for Llava (forget unimodal):",
          forget_unimodal_collector.list_collected_layers(modality="unimodal"))
    forget_unimodal_all_scores = compute_all_importance_scores(forget_unimodal_collector, modality="unimodal")

    # forget_unimodal_all_scores = compute_all_importance_scores(forget_unimodal_activations, "multimodal")
    print("Forget unimodal scores:", forget_unimodal_all_scores)
    forget_unimodal_collector.clear_activations()


    #######################  === Retain Set: Multimodal  ####################### ===
    retain_multimodal_collector = ActivationCollector()
    register_feedforward_hooks(model, retain_multimodal_collector, model_type="Llava")
    print("Testing single batch for retain multimodal...")
    retain_multimodal_activations = collect_feedforward_activations_multiple_batches(
        model,
        retain_multimodal_collector,
        retain_multimodal_dataloader,
        modality="multimodal",
        model_type="Llava",
        device=accelerator.device,
    )
    print("Collected layers for Llava (retain multimodal):",
          retain_multimodal_collector.list_collected_layers(modality="multimodal"))
    retain_multimodal_all_scores = compute_all_importance_scores(retain_multimodal_collector, modality="multimodal")

    # retain_multimodal_all_scores = compute_all_importance_scores(retain_multimodal_activations)
    print("Retain multimodal scores:", retain_multimodal_all_scores)
    retain_multimodal_collector.clear_activations()

    #######################  === Retain Set: Unimodal  ####################### ===
    retain_unimodal_collector = ActivationCollector()
    register_feedforward_hooks(model, retain_unimodal_collector, model_type="Llava")

    print("Testing single batch for retain unimodal...")
    retain_unimodal_activations = collect_feedforward_activations_multiple_batches(
        model,
        retain_unimodal_collector,
        retain_unimodal_dataloader,
        modality="unimodal",
        model_type="Llava",
        device=accelerator.device,
    )
    print("Collected layers for Llava (retain unimodal):",
          retain_unimodal_collector.list_collected_layers(modality="unimodal"))
    retain_unimodal_all_scores = compute_all_importance_scores(retain_unimodal_collector, modality="unimodal")

    # retain_unimodal_all_scores = compute_all_importance_scores(retain_unimodal_activations)
    print("Retain unimodal scores:", retain_unimodal_all_scores)
    retain_unimodal_collector.clear_activations()


    # === Compute Final Scores ===
    # Compute final scores using the combined importance across all metrics
    print("Computing final multimodal and unimodal scores...")
    final_multimodal_scores = compute_combined_scores(
        forget_multimodal_all_scores,
        retain_multimodal_all_scores,
    )

    final_unimodal_scores = compute_combined_scores(
        forget_unimodal_all_scores,
        retain_unimodal_all_scores,
    )

    print("forget multimodal all scores: ", forget_multimodal_all_scores)
    print("retain multimodal all scores: ", retain_multimodal_all_scores)
    print("forget unimodal all scores: ", forget_unimodal_all_scores)
    print("retain unimodal all scores: ", retain_unimodal_all_scores)

    print("Final multimodal scores:", final_multimodal_scores)
    print("Final unimodal scores:", final_unimodal_scores)

    print("Generating pruning masks using top-k percent strategy...")
    top_k_percent = 2  # Example: Prune top 2% of neurons

    # Multimodal pruning mask
    multimodal_pruning_mask = compute_top_k_pruning_mask(
        final_multimodal_scores["vision_fc1_0"], top_k_percent
    )
    # Unimodal pruning mask
    unimodal_pruning_mask = compute_top_k_pruning_mask(
        final_unimodal_scores["lang_gate_proj_0"], top_k_percent
    )
    # === Apply Pruning ===
    print("Applying pruning...")
    for name, param in model.named_parameters():
        if "vision_fc1_0" in name:
            if "weight" in name:
                reshaped_mask = multimodal_pruning_mask.view(-1, 1).to(param.device)
                param.data *= reshaped_mask
            elif "bias" in name:
                param.data *= multimodal_pruning_mask.to(param.device)

        if "lang_gate_proj_0" in name:
            if "weight" in name:
                reshaped_mask = unimodal_pruning_mask.view(-1, 1).to(param.device)
                param.data *= reshaped_mask
            elif "bias" in name:
                param.data *= unimodal_pruning_mask.to(param.device)

    print("Pruning completed!")



