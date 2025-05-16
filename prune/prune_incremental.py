import gc

import torch
import matplotlib.pyplot as plt
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
from prune_utility import register_feedforward_hooks, compute_top_k_pruning_mask, collect_feedforward_activations, \
    compute_combined_scores_incremental, apply_pruning, get_most_available_gpu, apply_structural_pruning, \
    count_parameters, identify_layer_keywords, count_pruned_parameters, print_pruning_stats
from activation_collect import ActivationCollector
from transformers import LlavaForConditionalGeneration
import torch

def print_memory_status():
    """
    Print the memory status of all CUDA devices.
    """
    print("\nCUDA Memory Status:")
    for device_id in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(device_id) / 1e9  # Convert to GB
        reserved = torch.cuda.memory_reserved(device_id) / 1e9    # Convert to GB
        print(f"Device {device_id}: Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
    print("-" * 40)
def load_model_and_processor(model_id, vanilla_dir):
    if model_id.startswith("llava"):
        print(f"Loading LLAVA model from {vanilla_dir}")
        model = LlavaForConditionalGeneration.from_pretrained(
            vanilla_dir,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
            local_files_only=True,
            # cache_dir = "/afs/crc.nd.edu/group/dmsquare/vol1/zliu29/mllm_unlearn/saved_model/LLaVA_7b_hf_vanilla",
            # cache_dir="/afs/crc.nd.edu/group/dmsquare/vol1/zliu29/mllm_unlearn/model/llava-1.5-7b-hf",
        )
        processor = AutoProcessor.from_pretrained(model_id)
        processor.tokenizer.padding_side = "right"
        processor.tokenizer.add_tokens(["<image>", "<pad>"], special_tokens=True)

    elif model_id.startswith("HuggingFaceM4"):
        print(f"Loading idefics2 model from {vanilla_dir}")
        model = Idefics2ForConditionalGeneration.from_pretrained(
            vanilla_dir,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
            local_files_only=True,
            # cache_dir="/afs/crc.nd.edu/group/dmsquare/vol1/zliu29/mllm_unlearn/model/idfics2-8b",
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


def clear_all_memory():
    """
    Clear all reserved and allocated memory across all GPUs and restart with a clean slate.
    """
    print("Clearing all GPU memory...")
    # Collect garbage
    gc.collect()

    # Delete all references to models or tensors
    torch.cuda.empty_cache()

    # Reset memory stats
    for device_id in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(device_id)
        torch.cuda.reset_accumulated_memory_stats(device_id)
        torch.cuda.memory_stats(device_id)  # Optional to inspect memory stats if needed

    print("All GPU memory cleared.")

def iterative_pruning(
        model,
        processor,
        forget_df,
        retain_df,
        num_iterations,
        prune_percent,
        save_path,
        args,
):
    """
    Perform iterative pruning with a specified number of iterations and pruning percentage.

    Args:
        model: The model to prune.
        forget_multimodal_dataloader: DataLoader for the forget multimodal dataset.
        forget_unimodal_dataloader: DataLoader for the forget unimodal dataset.
        retain_multimodal_dataloader: DataLoader for the retain multimodal dataset.
        retain_unimodal_dataloader: DataLoader for the retain unimodal dataset.
        num_iterations: Number of pruning iterations.
        prune_percent: Percentage of neurons to prune in each iteration.
        save_path: Path to save the pruned model after each iteration.
        accelerator: The Accelerator object for managing devices.
    """
    accelerator = Accelerator()

    # pruning_mask_keys = [
    #     'vision_fc1_0',  # For vision_fc1_0
    #     'vision_fc2_0',  # For vision_fc2_0
    #     'lang_gate_proj_0',  # For lang_gate_proj_0
    #     'lang_up_proj_0',  # For lang_up_proj_0
    #     'lang_down_proj_0'  # For lang_down_proj_0
    # ]
    #
    # found_keywords = identify_layer_keywords(model, pruning_mask_keys)
    # print("Found keywords:", found_keywords)

    for iteration in range(num_iterations):
        initial_stats = count_pruned_parameters(model, args.model_type)
        print("Before pruning:")
        print_pruning_stats(initial_stats)

        print(f"Starting pruning iteration {iteration + 1}/{num_iterations}...")

        # === Change 1: Save the model after every iteration and reload if iteration > 1 ===
        if args.all_layers:
            iteration_save_path = os.path.join(
                save_path,
                f"{args.model_save_name}_percent_{prune_percent}_iteration_{args.num_iterations}_forget_{args.forget_ratio}_all_layers"
            )
        else:
            iteration_save_path = os.path.join(
                save_path,
                f"{args.model_save_name}_percent_{prune_percent}_iteration_{args.num_iterations}_forget_{args.forget_ratio}"
            )

        if iteration > 0:  # Clear memory and reload the model if not the first iteration
            clear_all_memory()
            print_memory_status()
            print(f"Reloading pruned model from {iteration_save_path}...")
            model, processor = load_model_and_processor(args.model_id, iteration_save_path)
        # === End of Change 1 ===

        if args.model_id.startswith("llava"):
            llava_multimodal_forget_dataset = LLAVA_multimodal_Dataset(df=forget_df)
            llava_unimodal_forget_dataset = LLAVA_unimodal_Dataset(df=forget_df)
            llava_multimodal_retain_dataset = LLAVA_multimodal_Dataset(df=retain_df)
            llava_unimodal_retain_dataset = LLAVA_unimodal_Dataset(df=retain_df)
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
        elif args.model_id.startswith("HuggingFaceM4"):
            llava_multimodal_forget_dataset = LLAVA_multimodal_Dataset(df=forget_df)
            llava_unimodal_forget_dataset = LLAVA_unimodal_Dataset(df=forget_df)
            llava_multimodal_retain_dataset = LLAVA_multimodal_Dataset(df=retain_df)
            llava_unimodal_retain_dataset = LLAVA_unimodal_Dataset(df=retain_df)
            forget_multimodal_dataloader = DataLoader(
                llava_multimodal_forget_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                collate_fn=lambda x: train_collate_fn_idefics(x, processor, args, "multimodal")
            )
            forget_unimodal_dataloader = DataLoader(
                llava_unimodal_forget_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                collate_fn=lambda x: train_collate_fn_idefics(x, processor, args, "unimodal")
            )
            retain_multimodal_dataloader = DataLoader(
                llava_multimodal_retain_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                collate_fn=lambda x: train_collate_fn_idefics(x, processor, args, "multimodal")
            )
            retain_unimodal_dataloader = DataLoader(
                llava_unimodal_retain_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                collate_fn=lambda x: train_collate_fn_idefics(x, processor, args, "unimodal")
            )
        else:
            raise ValueError("Model ID not recognized or not supported. Please provide a valid model ID.")

        # Prepare the accelerator
        model, forget_multimodal_dataloader, forget_unimodal_dataloader, retain_multimodal_dataloader, retain_unimodal_dataloader = accelerator.prepare(
            model,
            forget_multimodal_dataloader,
            forget_unimodal_dataloader,
            retain_multimodal_dataloader,
            retain_unimodal_dataloader,
        )

        total_batches_forget_multimodal = len(forget_multimodal_dataloader)
        total_batches_forget_unimodal = len(forget_unimodal_dataloader)
        total_batches_retain_multimodal = len(retain_multimodal_dataloader)
        total_batches_retain_unimodal = len(retain_unimodal_dataloader)

        print(f"Total number of batches in forget multimodal dataloader: {total_batches_forget_multimodal}")
        print(f"Total number of batches in forget unimodal dataloader: {total_batches_forget_unimodal}")
        print(f"Total number of batches in retain multimodal dataloader: {total_batches_retain_multimodal}")
        print(f"Total number of batches in retain unimodal dataloader: {total_batches_retain_unimodal}")

        ####################### === Forget Set: Multimodal  ####################### ===

        select_device = get_most_available_gpu()
        # select_device = "cuda"
        print(f"Using device {select_device} for activation collection.")
        print("Processing forget set (multimodal)...")
        collector = ActivationCollector()
        register_feedforward_hooks(model, collector, select_device, model_type=args.model_type)

        torch.cuda.empty_cache()
        gc.collect()

        forget_multimodal_scores = collect_feedforward_activations(
            model,
            collector,
            forget_multimodal_dataloader,
            modality="multimodal",
            model_type=args.model_type,
            device=accelerator,
            best_device= select_device,
            # num_batches= 5,
        )
        print("Forget multimodal scores:", forget_multimodal_scores)
        forget_multimodal_dataloader = None
        del forget_multimodal_dataloader
        collector.clear_activations()
        torch.cuda.empty_cache()
        gc.collect()

        ####################### === Forget Set: Unimodal #######################  ===
        print("Processing forget set (unimodal)...")
        forget_unimodal_scores = collect_feedforward_activations(
            model,
            collector,
            forget_unimodal_dataloader,
            modality="unimodal",
            model_type=args.model_type,
            device=accelerator,
            best_device=select_device,
            # num_batches=5,
        )
        print("Forget unimodal scores:", forget_unimodal_scores)
        forget_unimodal_dataloader = None
        del forget_unimodal_dataloader
        collector.clear_activations()
        torch.cuda.empty_cache()
        gc.collect()

        #######################  === Retain Set: Multimodal  ####################### ===
        print("Processing retain set (multimodal)...")
        retain_multimodal_scores = collect_feedforward_activations(
            model,
            collector,
            retain_multimodal_dataloader,
            modality="multimodal",
            model_type=args.model_type,
            device=accelerator,
            best_device=select_device,
            num_batches=total_batches_forget_multimodal
            # num_batches=5,
        )
        print("Retain multimodal scores:", retain_multimodal_scores)
        retain_multimodal_dataloader = None
        del retain_multimodal_dataloader
        collector.clear_activations()
        torch.cuda.empty_cache()
        gc.collect()

        #######################  === Retain Set: Unimodal  ####################### ===
        print("Processing retain set (unimodal)...")
        retain_unimodal_scores = collect_feedforward_activations(
            model,
            collector,
            retain_unimodal_dataloader,
            modality="unimodal",
            model_type=args.model_type,
            device=accelerator,
            best_device=select_device,
            num_batches=total_batches_forget_unimodal
            # num_batches=5,
        )
        print("Retain unimodal scores:", retain_unimodal_scores)
        retain_unimodal_dataloader = None
        del retain_unimodal_dataloader
        collector.clear_activations()
        torch.cuda.empty_cache()
        gc.collect()

        del collector
        torch.cuda.empty_cache()

        ####################### === Compute Final Scores ####################### ===
        print("Computing final multimodal and unimodal scores...")
        final_multimodal_scores = compute_combined_scores_incremental(forget_multimodal_scores,
                                                                      retain_multimodal_scores)
        final_unimodal_scores = compute_combined_scores_incremental(forget_unimodal_scores, retain_unimodal_scores)

        print("final_multimodal_scores: ", final_multimodal_scores)
        print("final_unimodal_scores: ", final_unimodal_scores)

        del forget_multimodal_scores, forget_unimodal_scores
        del retain_multimodal_scores, retain_unimodal_scores
        torch.cuda.empty_cache()

        multimodal_pruning_mask = compute_top_k_pruning_mask(
            final_multimodal_scores, prune_percent
        )
        unimodal_pruning_mask = compute_top_k_pruning_mask(
            final_unimodal_scores, prune_percent
        )
        del final_multimodal_scores, final_unimodal_scores

        print("multimodal_pruning_mask: ", multimodal_pruning_mask)
        print("unimodal_pruning_mask: ", unimodal_pruning_mask)

        print("Applying pruning to multimodal pruning...")
        apply_structural_pruning(model, multimodal_pruning_mask, args.model_type)
        del multimodal_pruning_mask

        print("Applying pruning to unimodal pruning...")
        apply_structural_pruning(model, unimodal_pruning_mask, args.model_type)
        del unimodal_pruning_mask
        torch.cuda.empty_cache()


        ### === Compute final stats after pruning ===
        final_stats = count_pruned_parameters(model, args.model_type)
        print("\nAfter pruning:")
        print_pruning_stats(final_stats)
        # === Change 2: Save the model after every iteration ===
        print(f"Saving pruned model to {iteration_save_path}")
        os.makedirs(iteration_save_path, exist_ok=True)
        model.save_pretrained(iteration_save_path)
        # del model
        # === End of Change 2 ===
        print_memory_status()

    print(f"All pruning iterations completed!")

# def iterative_pruning(
#     model,
#     forget_multimodal_dataloader,
#     forget_unimodal_dataloader,
#     retain_multimodal_dataloader,
#     retain_unimodal_dataloader,
#     num_iterations,
#     prune_percent,
#     save_path,
#     accelerator,
#     args,
# ):
#     """
#     Perform iterative pruning with a specified number of iterations and pruning percentage.
#
#     Args:
#         model: The model to prune.
#         forget_multimodal_dataloader: DataLoader for the forget multimodal dataset.
#         forget_unimodal_dataloader: DataLoader for the forget unimodal dataset.
#         retain_multimodal_dataloader: DataLoader for the retain multimodal dataset.
#         retain_unimodal_dataloader: DataLoader for the retain unimodal dataset.
#         num_iterations: Number of pruning iterations.
#         prune_percent: Percentage of neurons to prune in each iteration.
#         save_path: Path to save the pruned model after each iteration.
#         accelerator: The Accelerator object for managing devices.
#     """
#     for iteration in range(num_iterations):
#         print(f"Starting pruning iteration {iteration + 1}/{num_iterations}...")
#
#         ####################### === Forget Set: Multimodal  ####################### ===
#         print("Processing forget set (multimodal)...")
#         # Instantiate activation collectors
#         collector = ActivationCollector()
#         register_feedforward_hooks(model, collector, model_type="Llava")
#
#         forget_multimodal_scores = collect_feedforward_activations(
#             model,
#             collector,
#             forget_multimodal_dataloader,
#             modality="multimodal",
#             model_type="Llava",
#             device=accelerator,
#         )
#         print("Forget multimodal scores:", forget_multimodal_scores)
#         collector.clear_activations()
#         forget_multimodal_dataloader = None  # Release memory
#         torch.cuda.empty_cache()
#
#         ####################### === Forget Set: Unimodal #######################  ===
#         print("Testing forget unimodal...")
#         print("Processing forget set (unimodal)...")
#         forget_unimodal_scores = collect_feedforward_activations(
#             model,
#             collector,
#             forget_unimodal_dataloader,
#             modality="unimodal",
#             model_type="Llava",
#             device=accelerator,
#         )
#         print("Forget unimodal scores:", forget_unimodal_scores)
#         collector.clear_activations()
#         forget_unimodal_dataloader = None  # Release memory
#         torch.cuda.empty_cache()
#
#         #######################  === Retain Set: Multimodal  ####################### ===
#         print("Processing retain set (multimodal)...")
#         retain_multimodal_scores = collect_feedforward_activations(
#             model,
#             collector,
#             retain_multimodal_dataloader,
#             modality="multimodal",
#             model_type="Llava",
#             device=accelerator,
#             # num_batches=total_batches_forget_multimodal,
#         )
#         print("Retain multimodal scores:", retain_multimodal_scores)
#         collector.clear_activations()
#         retain_multimodal_dataloader = None  # Release memory
#         torch.cuda.empty_cache()
#
#         #######################  === Retain Set: Unimodal  ####################### ===
#         print("Processing retain set (unimodal)...")
#         retain_unimodal_scores = collect_feedforward_activations(
#             model,
#             collector,
#             retain_unimodal_dataloader,
#             modality="unimodal",
#             model_type="Llava",
#             device=accelerator,
#             # num_batches=total_batches_forget_unimodal,
#         )
#         print("Retain unimodal scores:", retain_unimodal_scores)
#         collector.clear_activations()
#         retain_unimodal_dataloader = None  # Release memory
#         torch.cuda.empty_cache()
#
#         ####################### === Compute Final Scores ####################### ===
#         print("Computing final multimodal and unimodal scores...")
#         final_multimodal_scores = compute_combined_scores_incremental(forget_multimodal_scores,retain_multimodal_scores)
#         final_unimodal_scores = compute_combined_scores_incremental(forget_unimodal_scores, retain_unimodal_scores)
#
#         print("forget multimodal all scores: ", forget_multimodal_scores)
#         print("forget unimodal all scores: ", forget_unimodal_scores)
#
#         print("retain multimodal all scores: ", retain_multimodal_scores)
#         print("retain unimodal all scores: ", retain_unimodal_scores)
#
#         print("Final multimodal scores:", final_multimodal_scores)
#         print("Final unimodal scores:", final_unimodal_scores)
#         print("Generating pruning masks using top-k percent strategy...")
#
#         ######################## Multimodal pruning mask #######################
#         # multimodal_pruning_mask = compute_top_k_pruning_mask(
#         #     final_multimodal_scores["vision_fc1_0"], prune_percent
#         # )
#         # # Unimodal pruning mask
#         # unimodal_pruning_mask = compute_top_k_pruning_mask(
#         #     final_unimodal_scores["lang_gate_proj_0"], prune_percent
#         # )
#
#         ######################## === Apply Pruning ####################### ===
#         # print("Applying pruning...")
#         # for name, param in model.named_parameters():
#         #     if "vision_fc1_0" or "vision_fc2_0" in name:
#         #         if "weight" in name:
#         #             reshaped_mask = multimodal_pruning_mask.view(-1, 1).to(param.device)
#         #             param.data *= reshaped_mask
#         #         elif "bias" in name:
#         #             param.data *= multimodal_pruning_mask.to(param.device)
#         #
#         #     if "lang_gate_proj_0" or "lang_up_proj_0" or "lang_down_proj_0" in name:
#         #         if "weight" in name:
#         #             reshaped_mask = unimodal_pruning_mask.view(-1, 1).to(param.device)
#         #             param.data *= reshaped_mask
#         #         elif "bias" in name:
#         #             param.data *= unimodal_pruning_mask.to(param.device)
#         #
#         # print("Pruning completed!")
#
#         multimodal_pruning_mask = compute_top_k_pruning_mask(
#             final_multimodal_scores, prune_percent
#         )
#         # Unimodal pruning mask
#         unimodal_pruning_mask = compute_top_k_pruning_mask(
#             final_unimodal_scores, prune_percent
#         )
#
#         print("Applying pruning to multimodal pruning...")
#         apply_pruning(model, multimodal_pruning_mask)
#
#         print("Applying pruning to unimodal pruning...")
#         apply_pruning(model, unimodal_pruning_mask)
#
#
#     ######################## Save the Model #################################
#     iteration_save_path = os.path.join(
#         save_path,
#         f"{args.model_save_name}_percent_{prune_percent}_iteration_{iteration + 1}_forget_{args.forget_ratio}"
#     )
#     print(f"Saving pruned model to {iteration_save_path}...")
#
#     # Ensure the directory exists
#     os.makedirs(iteration_save_path, exist_ok=True)
#
#     # Save the model
#     model.save_pretrained(iteration_save_path)
#
#     print(f"Pruning iteration {iteration + 1}/{num_iterations} completed.")
#
#     print("All pruning iterations completed!")

if __name__ == "__main__":
    # Load model and processor
    parser = argparse.ArgumentParser(description="Fine-tune different models")
    parser.add_argument("--model_save_name", type=str, required=True, help="llava or idefics2")
    parser.add_argument("--vanilla_dir", type=str, required=True, help="Pretrained model ID")
    parser.add_argument("--model_id", type=str, required=True, help="Pretrained model ID")
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory to save the model")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--max_length", type=int, default=384, help="Maximum sequence length")
    parser.add_argument('--forget_ratio', type=int, default=5, help='Path to real person image folder.')
    parser.add_argument("--trainer", type=bool, default=False, help="Use HuggingFace Trainer")
    parser.add_argument("--num_iterations", type=int, default=1, help="Number of pruning iterations")
    parser.add_argument("--prune_percent", type=float, default=2, help="Percentage of neurons to prune per iteration")
    parser.add_argument("--model_type", type=str, default="Llava", help="Llava or Idefics2")
    parser.add_argument("--save_path", type=str, help="Path to save the pruned models")
    parser.add_argument("--all_layers", type=bool, default=False, help="prune all layers")
    args = parser.parse_args()

    # model_id = args.model_id
    model, processor = load_model_and_processor(args.model_id, args.vanilla_dir)
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

    # Perform iterative pruning
    iterative_pruning(
        model=model,
        processor=processor,
        forget_df=forget_df,
        retain_df=retain_df,
        num_iterations=args.num_iterations,
        prune_percent=args.prune_percent,
        save_path=args.save_path,
        args= args
    )

    # Instantiate a single activation collector for all phases
    # collector = ActivationCollector()
    # register_feedforward_hooks(model, collector, model_type="Llava")
    #
    # #######################  === Forget Set: Multimodal  ####################### ===
    # print("Testing single batch for forget multimodal...")
    # print("Processing forget set (multimodal)...")
    # forget_multimodal_scores = collect_feedforward_activations(
    #     model,
    #     collector,
    #     forget_multimodal_dataloader,
    #     modality="multimodal",
    #     model_type="Llava",
    #     device=accelerator,
    # )
    # print("Forget multimodal scores:", forget_multimodal_scores)
    # collector.clear_activations()
    # forget_multimodal_dataloader = None  # Release memory
    # torch.cuda.empty_cache()
    #
    # ####################### === Forget Set: Unimodal #######################  ===
    # print("Testing forget unimodal...")
    # print("Processing forget set (unimodal)...")
    # forget_unimodal_scores = collect_feedforward_activations(
    #     model,
    #     collector,
    #     forget_unimodal_dataloader,
    #     modality="unimodal",
    #     model_type="Llava",
    #     device=accelerator,
    # )
    # print("Forget unimodal scores:", forget_unimodal_scores)
    # collector.clear_activations()
    # forget_unimodal_dataloader = None  # Release memory
    # torch.cuda.empty_cache()
    #
    # #######################  === Retain Set: Multimodal  ####################### ===
    # print("Processing retain set (multimodal)...")
    # retain_multimodal_scores = collect_feedforward_activations(
    #     model,
    #     collector,
    #     retain_multimodal_dataloader,
    #     modality="multimodal",
    #     model_type="Llava",
    #     device=accelerator,
    #     # num_batches=total_batches_forget_multimodal,
    # )
    # print("Retain multimodal scores:", retain_multimodal_scores)
    # collector.clear_activations()
    # retain_multimodal_dataloader = None  # Release memory
    # torch.cuda.empty_cache()
    #
    # #######################  === Retain Set: Unimodal  ####################### ===
    # print("Processing retain set (unimodal)...")
    # retain_unimodal_scores = collect_feedforward_activations(
    #     model,
    #     collector,
    #     retain_unimodal_dataloader,
    #     modality="unimodal",
    #     model_type="Llava",
    #     device=accelerator,
    #     # num_batches=total_batches_forget_unimodal,
    # )
    # print("Retain unimodal scores:", retain_unimodal_scores)
    # collector.clear_activations()
    # retain_unimodal_dataloader = None  # Release memory
    # torch.cuda.empty_cache()
    #
    # ####################### === Compute Final Scores ####################### ===
    # print("Computing final multimodal and unimodal scores...")
    # final_multimodal_scores = compute_combined_scores_incremental(forget_multimodal_scores, retain_multimodal_scores)
    # final_unimodal_scores = compute_combined_scores_incremental(forget_unimodal_scores, retain_unimodal_scores)
    #
    # print("forget multimodal all scores: ", forget_multimodal_scores)
    # print("retain multimodal all scores: ", retain_multimodal_scores)
    # print("forget unimodal all scores: ", forget_unimodal_scores)
    # print("retain unimodal all scores: ", retain_unimodal_scores)
    #
    # print("Final multimodal scores:", final_multimodal_scores)
    # print("Final unimodal scores:", final_unimodal_scores)
    #
    # print("Generating pruning masks using top-k percent strategy...")
    # top_k_percent = 2  # Example: Prune top 2% of neurons
    #
    # ######################## Multimodal pruning mask #######################
    # # multimodal_pruning_mask = compute_top_k_pruning_mask(
    # #     final_multimodal_scores["vision_fc1_0"], top_k_percent
    # # )
    # # # Unimodal pruning mask
    # # unimodal_pruning_mask = compute_top_k_pruning_mask(
    # #     final_unimodal_scores["lang_gate_proj_0"], top_k_percent
    # # )
    #
    # multimodal_pruning_mask = compute_top_k_pruning_mask(
    #     final_multimodal_scores, top_k_percent
    # )
    # # Unimodal pruning mask
    # unimodal_pruning_mask = compute_top_k_pruning_mask(
    #     final_unimodal_scores, top_k_percent
    # )
    #
    # ######################## === Apply Pruning ####################### ===
    # # print("Applying pruning...")
    # # for name, param in model.named_parameters():
    # #     if "vision_fc1_0" in name:
    # #         if "weight" in name:
    # #             reshaped_mask = multimodal_pruning_mask.view(-1, 1).to(param.device)
    # #             param.data *= reshaped_mask
    # #         elif "bias" in name:
    # #             param.data *= multimodal_pruning_mask.to(param.device)
    # #
    # #     if "lang_gate_proj_0" in name:
    # #         if "weight" in name:
    # #             reshaped_mask = unimodal_pruning_mask.view(-1, 1).to(param.device)
    # #             param.data *= reshaped_mask
    # #         elif "bias" in name:
    # #             param.data *= unimodal_pruning_mask.to(param.device)
    # #
    # # print("Pruning completed!")
    #
    # print("Applying pruning to multimodal pruning...")
    # apply_pruning(model, multimodal_pruning_mask)
    #
    # print("Applying pruning to unimodal pruning...")
    # apply_pruning(model, unimodal_pruning_mask)



