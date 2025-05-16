import os
from collections import defaultdict

import numpy as np
import torch.nn.functional as F
import torch
from welford_torch import Welford
import psutil
import gc
from pathlib import Path
import torch.nn.utils.prune as prune
import h5py
from tqdm import tqdm


def identify_layer_keywords(model, pruning_mask_keys):
    # Initialize a list to store found keywords
    found_keywords = []

    # Check vision model layers
    for layer_idx, layer in enumerate(model.vision_tower.vision_model.encoder.layers):
        # Check if the layers like 'vision_fc1_{layer_idx}' and 'vision_fc2_{layer_idx}' exist
        if hasattr(layer.mlp, 'fc1'):
            vision_fc1_key = f"vision_fc1_{layer_idx}"
            if vision_fc1_key in pruning_mask_keys:
                found_keywords.append(vision_fc1_key)

        if hasattr(layer.mlp, 'fc2'):
            vision_fc2_key = f"vision_fc2_{layer_idx}"
            if vision_fc2_key in pruning_mask_keys:
                found_keywords.append(vision_fc2_key)

    # Check language model layers
    for layer_idx, layer in enumerate(model.language_model.model.layers):
        # Check for layers like 'lang_gate_proj_{layer_idx}', 'lang_up_proj_{layer_idx}', 'lang_down_proj_{layer_idx}'
        if hasattr(layer.mlp, 'gate_proj'):
            lang_gate_proj_key = f"lang_gate_proj_{layer_idx}"
            if lang_gate_proj_key in pruning_mask_keys:
                found_keywords.append(lang_gate_proj_key)

        if hasattr(layer.mlp, 'up_proj'):
            lang_up_proj_key = f"lang_up_proj_{layer_idx}"
            if lang_up_proj_key in pruning_mask_keys:
                found_keywords.append(lang_up_proj_key)

        if hasattr(layer.mlp, 'down_proj'):
            lang_down_proj_key = f"lang_down_proj_{layer_idx}"
            if lang_down_proj_key in pruning_mask_keys:
                found_keywords.append(lang_down_proj_key)

    return found_keywords


def count_parameters(model):
    """
    Counts the total number of trainable parameters in the model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_most_available_gpu():
    """
    Get the GPU with the most available memory.
    Returns:
        torch.device: The device with the most available memory.
    """
    available_memory = []
    for device_id in range(torch.cuda.device_count()):
        stats = torch.cuda.memory_stats(device_id)
        free_memory = stats['active_bytes.all.current'] - stats['reserved_bytes.all.current']
        available_memory.append((device_id, free_memory))
        print(f"Device {device_id}: {free_memory / 1e9:.2f} GB available")

    # Select the GPU with the most available memory
    best_device_id = max(available_memory, key=lambda x: x[1])[0]
    print(f"Selected device: cuda:{best_device_id}")
    return torch.device(f"cuda:{best_device_id}")

def register_feedforward_hooks(model, collector, device, model_type):
    """
    Register hooks to collect activations from feed-forward layers based on the model type.
    Args:
        model: The multimodal model (LlavaForConditionalGeneration or Idefics2ForConditionalGeneration).
        collector: ActivationCollector instance.
        model_type: "Llava" or "Idefics2".
    """
    # device = get_most_available_gpu()
    # print(f"Using device {device} for activation collection.")

    if model_type == "Llava":
        # Register hooks for Llava vision tower
        for layer_idx, layer in enumerate(model.vision_tower.vision_model.encoder.layers):
            # collector.register_hook(layer.mlp.fc1, f"vision_fc1_{layer_idx}", modality="multimodal")
            # collector.register_hook(layer.mlp.fc2, f"vision_fc2_{layer_idx}", modality="multimodal")
            collector.register_hook(layer.mlp.fc1, f"vision_fc1_21", device, modality="multimodal")
            collector.register_hook(layer.mlp.fc2, f"vision_fc2_21", device, modality="multimodal")
            collector.register_hook(layer.mlp.fc1, f"vision_fc1_22", device, modality="multimodal")
            collector.register_hook(layer.mlp.fc2, f"vision_fc2_22", device, modality="multimodal")
            collector.register_hook(layer.mlp.fc1, f"vision_fc1_23", device, modality="multimodal")
            collector.register_hook(layer.mlp.fc2, f"vision_fc2_23", device, modality="multimodal")

        # Register hooks for Llava language model
        for layer_idx, layer in enumerate(model.language_model.model.layers):
            # collector.register_hook(layer.mlp.gate_proj, f"lang_gate_proj_{layer_idx}", modality="unimodal")
            # collector.register_hook(layer.mlp.up_proj, f"lang_up_proj_{layer_idx}", modality="unimodal")
            # collector.register_hook(layer.mlp.down_proj, f"lang_down_proj_{layer_idx}", modality="unimodal")

            collector.register_hook(layer.mlp.gate_proj, f"lang_gate_proj_29", device, modality="unimodal")
            collector.register_hook(layer.mlp.up_proj, f"lang_up_proj_29", device, modality="unimodal")
            collector.register_hook(layer.mlp.down_proj, f"lang_down_proj_29", device, modality="unimodal")
            collector.register_hook(layer.mlp.gate_proj, f"lang_gate_proj_30", device, modality="unimodal")
            collector.register_hook(layer.mlp.up_proj, f"lang_up_proj_30", device, modality="unimodal")
            collector.register_hook(layer.mlp.down_proj, f"lang_down_proj_30", device, modality="unimodal")
            collector.register_hook(layer.mlp.gate_proj, f"lang_gate_proj_31", device, modality="unimodal")
            collector.register_hook(layer.mlp.up_proj, f"lang_up_proj_31",device, modality="unimodal")
            collector.register_hook(layer.mlp.down_proj, f"lang_down_proj_31",device, modality="unimodal")

    elif model_type == "Idefics2":
        # Register hooks for Idefics2 vision model
        for layer_idx, layer in enumerate(model.model.vision_model.encoder.layers):
            collector.register_hook(layer.mlp.fc1, f"vision_fc1_21", device, modality="multimodal")
            # collector.register_hook(layer.mlp.fc2, f"vision_fc2_21", device, modality="multimodal")
            collector.register_hook(layer.mlp.fc1, f"vision_fc1_22", device, modality="multimodal")
            # collector.register_hook(layer.mlp.fc2, f"vision_fc2_22", device, modality="multimodal")
            collector.register_hook(layer.mlp.fc1, f"vision_fc1_23", device, modality="multimodal")
            # collector.register_hook(layer.mlp.fc2, f"vision_fc2_23", device, modality="multimodal")

        # Register hooks for Idefics2 connector MLP
        # connector = model.model.connector.modality_projection
        # collector.register_hook(connector.gate_proj, "connector_gate_proj")
        # collector.register_hook(connector.up_proj, "connector_up_proj")
        # collector.register_hook(connector.down_proj, "connector_down_proj")

        # Register hooks for Idefics2 text model
        for layer_idx, layer in enumerate(model.model.text_model.layers):
            collector.register_hook(layer.mlp.gate_proj, f"text_gate_proj_29", device, modality="unimodal")
            collector.register_hook(layer.mlp.up_proj, f"text_up_proj_29", device, modality="unimodal")
            collector.register_hook(layer.mlp.down_proj, f"text_down_proj_29", device, modality="unimodal")
            collector.register_hook(layer.mlp.gate_proj, f"text_gate_proj_30", device, modality="unimodal")
            collector.register_hook(layer.mlp.up_proj, f"text_up_proj_30", device, modality="unimodal")
            collector.register_hook(layer.mlp.down_proj, f"text_down_proj_30", device, modality="unimodal")
            collector.register_hook(layer.mlp.gate_proj, f"text_gate_proj_31", device, modality="unimodal")
            collector.register_hook(layer.mlp.up_proj, f"text_up_proj_31", device, modality="unimodal")
            collector.register_hook(layer.mlp.down_proj, f"text_down_proj_31", device, modality="unimodal")

    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def collect_feedforward_activations(
    model, collector, dataloader, modality, model_type, device, num_batches=None, chunk_size=10, best_device="cuda:1"
):
    """
    Collect activations and compute importance scores batch-by-batch with chunked aggregation.
    Ensures consistent device allocation during aggregation.
    """
    collector.clear_activations()
    model.eval()
    num_batches = len(dataloader) if num_batches is None else num_batches

    # Initialize aggregated metrics
    aggregated_scores = defaultdict(lambda: defaultdict(list))

    # Choose a consistent aggregation device
    # aggregation_device = get_most_available_gpu()
    # aggregation_device = torch.device("cuda:1")
    aggregation_device = torch.device(best_device)
    for chunk_start in range(0, num_batches, chunk_size):
        chunk_end = min(chunk_start + chunk_size, num_batches)
        print(f"Processing chunk {chunk_start + 1} to {chunk_end}...")

        # Initialize chunk scores
        chunk_scores = defaultdict(lambda: defaultdict(list))

        try:
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx < chunk_start or batch_idx >= chunk_end:
                    continue

                print(f"Processing batch {batch_idx + 1}/{num_batches}...")

                # Prepare inputs
                if modality == "multimodal":
                    input_ids, attention_mask, pixel_values, labels = batch
                    inputs = {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "pixel_values": pixel_values,
                        "labels": labels,
                    }
                elif modality == "unimodal":
                    if model_type == "Llava":
                        input_ids, attention_mask, labels = batch
                        inputs = {
                            "input_ids": input_ids,
                            "attention_mask": attention_mask,
                            "labels": labels,
                        }
                    elif model_type == "Idefics2":
                        input_ids, attention_mask, labels = batch
                        inputs = {
                            "input_ids": input_ids,
                            "attention_mask": attention_mask,
                            # "labels": labels,
                        }

                # Forward pass
                with torch.no_grad():
                    if model_type == "Llava":
                        if modality == "multimodal":
                            device.unwrap_model(model)(**inputs)
                        elif modality == "unimodal":
                            device.unwrap_model(model).language_model(**inputs)

                    elif model_type == "Idefics2":
                        if modality == "multimodal":
                            device.unwrap_model(model)(**inputs)
                        elif modality == "unimodal":
                            device.unwrap_model(model).model.text_model(**inputs)

                # Compute metrics for each layer
                for layer_name in collector.list_collected_layers(modality=modality):
                    batch_activations = collector.get_activations(layer_name, modality=modality)
                    # print(f"Layer: {layer_name}, Batch {batch_idx + 1}, Activation Shape: {batch_activations.shape}")

                    # Compute metrics for the batch
                    clamped_activations = batch_activations.float().clamp(min=-1e3, max=1e3)
                    chunk_scores["I_abs"][layer_name].append(clamped_activations.abs().mean(dim=0))
                    chunk_scores["I_freq"][layer_name].append((clamped_activations.abs() > 1e-1).float().mean(dim=0))
                    chunk_scores["I_var"][layer_name].append(clamped_activations.std(dim=0))
                    chunk_scores["I_rms"][layer_name].append(torch.sqrt((clamped_activations**2).mean(dim=0)))

                    del clamped_activations, batch_activations
                    torch.cuda.empty_cache()

                collector.clear_activations()
                torch.cuda.empty_cache()

            # Aggregate chunk scores into the final scores
            print(f"Aggregating scores for chunk {chunk_start + 1} to {chunk_end}...")
            for metric, layer_dict in chunk_scores.items():
                for layer_name, scores in layer_dict.items():
                    # Concatenate scores for this chunk
                    scores = [score.to(aggregation_device) for score in scores]
                    # chunk_mean = torch.cat(scores).mean(dim=0).to(aggregation_device)
                    chunk_mean = torch.cat(scores).mean(dim=0).to(aggregation_device)
                    aggregated_scores[metric][layer_name].append(chunk_mean)

        except RuntimeError as e:
            print(f"Chunk aggregation failed due to OOM error for chunk {chunk_start + 1} to {chunk_end}: {e}")
            last_successful_batches = chunk_start
            print(f"Using aggregated scores from previous {last_successful_batches} batches.")
            break  # Skip this chunk and retain previously aggregated results

        del chunk_scores
        torch.cuda.empty_cache()

    # Finalize results by computing mean across all aggregated chunks
    final_scores = defaultdict(dict)
    try:
        for metric, layer_dict in aggregated_scores.items():
            for layer_name, chunk_means in layer_dict.items():
                # Concatenate chunk means and compute the final mean
                chunk_means = [score.to(aggregation_device) for score in chunk_means]
                final_scores[metric][layer_name] = torch.stack(chunk_means).mean(dim=0).to(aggregation_device)
    except RuntimeError as e:
        print(f"Final aggregation failed: {e}")
        return aggregated_scores  # Return the last successfully aggregated chunk

    return final_scores



def compute_combined_scores_incremental(forget_scores, retain_scores, weights=None, epsilon=1e-5):
    """
    Combine importance scores from forget and retain sets using weighted metrics.
    Args:
        forget_scores: Importance scores from the forget set.
        retain_scores: Importance scores from the retain set.
        weights: Dictionary of weights for each metric. Defaults to equal weights.
        epsilon: Small value to avoid division by zero.
    Returns:
        combined_scores: Combined importance scores for each layer.
    """
    if weights is None:
        weights = {"I_abs": 2.0, "I_freq":  0.0, "I_var":  2.0, "I_rms":  2.0}
        print("weights: ", weights)

    combined_scores = {}
    for metric in forget_scores:  # Iterate over all metrics (I_abs, I_freq, etc.)
        for layer_name in forget_scores[metric]:
            if layer_name not in retain_scores[metric]:
                raise KeyError(f"Layer {layer_name} not found in retain scores for metric {metric}.")

            # Extract per-neuron importance scores
            forget_importance = forget_scores[metric][layer_name]
            retain_importance = retain_scores[metric][layer_name]

            # Ensure both tensors are on the same device
            best_device = forget_importance.device  # Use the device of forget_importance
            forget_importance = forget_importance.to(best_device)
            retain_importance = retain_importance.to(best_device)

            # Combine scores for the given metric and layer
            combined_metric_score = weights[metric] * (
                (forget_importance / (retain_importance + epsilon)) - 1
            )

            # Accumulate the combined scores for each layer
            if layer_name not in combined_scores:
                combined_scores[layer_name] = combined_metric_score
            else:
                # Ensure the existing score is on the same device as the new score
                combined_scores[layer_name] = combined_scores[layer_name].to(combined_metric_score.device)
                combined_scores[layer_name] += combined_metric_score

    return combined_scores


def compute_top_k_pruning_mask(combined_scores_dict, top_k_percent):
    """
    Compute pruning masks for multiple layers based on the top-k percent of neurons.
    Args:
        combined_scores_dict: Dictionary of combined importance scores for multiple layers.
                              {layer_name: tensor_of_scores}
        top_k_percent: Float indicating the percentage of neurons to prune (e.g., 2 for top 2%).
    Returns:
        pruning_masks: Dictionary of binary masks (1 for pruned neurons, 0 for retained neurons) for each layer.
                       {layer_name: tensor_of_mask}
    """
    # Flatten all scores across layers to compute a global threshold
    all_scores = torch.cat([scores.flatten() for scores in combined_scores_dict.values()])

    # Determine the number of neurons to prune
    num_neurons = all_scores.numel()
    k = int((top_k_percent / 100) * num_neurons)

    # Find the global threshold score for the top-k neurons
    top_k_threshold, _ = torch.topk(all_scores, k, largest=True)
    threshold = top_k_threshold[-1]

    # Create a pruning mask for each layer based on the global threshold
    pruning_masks = {}
    for layer_name, scores in combined_scores_dict.items():
        pruning_masks[layer_name] = (scores >= threshold).float()

    return pruning_masks


def apply_structural_pruning(model, pruning_masks, model_type):
    """
    Apply mask-only pruning to the model by zeroing out pruned neurons without physically removing them.
    This maintains model architecture for loading while achieving pruning effects.

    Args:
        model: The model to prune
        pruning_masks: Dictionary of pruning masks for each layer
                      {layer_name: tensor_of_mask}
    Returns:
        applied_masks: Dictionary tracking the applied masks for potential iterative pruning
    """
    applied_masks = {}

    if model_type == "Llava":
        # For vision model layers
        for layer_idx, layer in enumerate(model.vision_tower.vision_model.encoder.layers):
            if hasattr(layer.mlp, 'fc1'):
                vision_fc1_key = f"vision_fc1_{layer_idx}"
                if vision_fc1_key in pruning_masks:
                    mask = pruning_masks[vision_fc1_key].to(layer.mlp.fc1.weight.device)
                    print(f"Applying mask to layer: {vision_fc1_key}")
                    apply_mask_to_layer(layer.mlp.fc1, mask)
                    applied_masks[vision_fc1_key] = mask

            if hasattr(layer.mlp, 'fc2'):
                vision_fc2_key = f"vision_fc2_{layer_idx}"
                if vision_fc2_key in pruning_masks:
                    mask = pruning_masks[vision_fc2_key].to(layer.mlp.fc2.weight.device)
                    print(f"Applying mask to layer: {vision_fc2_key}")
                    apply_mask_to_layer(layer.mlp.fc2, mask)
                    applied_masks[vision_fc2_key] = mask

        # For language model layers
        for layer_idx, layer in enumerate(model.language_model.model.layers):
            if hasattr(layer.mlp, 'gate_proj'):
                lang_gate_proj_key = f"lang_gate_proj_{layer_idx}"
                if lang_gate_proj_key in pruning_masks:
                    mask = pruning_masks[lang_gate_proj_key].to(layer.mlp.gate_proj.weight.device)
                    print(f"Applying mask to layer: {lang_gate_proj_key}")
                    apply_mask_to_layer(layer.mlp.gate_proj, mask)
                    applied_masks[lang_gate_proj_key] = mask

            if hasattr(layer.mlp, 'up_proj'):
                lang_up_proj_key = f"lang_up_proj_{layer_idx}"
                if lang_up_proj_key in pruning_masks:
                    mask = pruning_masks[lang_up_proj_key].to(layer.mlp.up_proj.weight.device)
                    print(f"Applying mask to layer: {lang_up_proj_key}")
                    apply_mask_to_layer(layer.mlp.up_proj, mask)
                    applied_masks[lang_up_proj_key] = mask

            if hasattr(layer.mlp, 'down_proj'):
                lang_down_proj_key = f"lang_down_proj_{layer_idx}"
                if lang_down_proj_key in pruning_masks:
                    mask = pruning_masks[lang_down_proj_key].to(layer.mlp.down_proj.weight.device)
                    print(f"Applying mask to layer: {lang_down_proj_key}")
                    apply_mask_to_layer(layer.mlp.down_proj, mask)
                    applied_masks[lang_down_proj_key] = mask


    elif model_type == "Idefics2":
        # For vision model layers
        for layer_idx, layer in enumerate(model.model.vision_model.encoder.layers):
            # Apply pruning mask to fc1
            if hasattr(layer.mlp, 'fc1'):
                vision_fc1_key = f"vision_fc1_{layer_idx}"
                if vision_fc1_key in pruning_masks:
                    mask = pruning_masks[vision_fc1_key].to(layer.mlp.fc1.weight.device)
                    print(f"Applying mask to layer: {vision_fc1_key}")
                    apply_mask_to_layer(layer.mlp.fc1, mask)
                    applied_masks[vision_fc1_key] = mask
            # Apply pruning mask to fc2
            if hasattr(layer.mlp, 'fc2'):
                vision_fc2_key = f"vision_fc2_{layer_idx}"
                if vision_fc2_key in pruning_masks:
                    mask = pruning_masks[vision_fc2_key].to(layer.mlp.fc2.weight.device)
                    print(f"Applying mask to layer: {vision_fc2_key}")
                    apply_mask_to_layer(layer.mlp.fc2, mask)
                    applied_masks[vision_fc2_key] = mask


        for layer_idx, layer in enumerate(model.model.text_model.layers):
            if hasattr(layer.mlp, 'gate_proj'):
                text_gate_proj_key = f"text_gate_proj_{layer_idx}"
                if text_gate_proj_key in pruning_masks:
                    mask = pruning_masks[text_gate_proj_key].to(layer.mlp.gate_proj.weight.device)
                    print(f"Applying mask to layer: {text_gate_proj_key}")
                    apply_mask_to_layer(layer.mlp.gate_proj, mask)
                    applied_masks[text_gate_proj_key] = mask
            if hasattr(layer.mlp, 'up_proj'):
                text_up_proj_key = f"text_up_proj_{layer_idx}"
                if text_up_proj_key in pruning_masks:
                    mask = pruning_masks[text_up_proj_key].to(layer.mlp.up_proj.weight.device)
                    print(f"Applying mask to layer: {text_up_proj_key}")
                    apply_mask_to_layer(layer.mlp.up_proj, mask)
                    applied_masks[text_up_proj_key] = mask
            if hasattr(layer.mlp, 'down_proj'):
                text_down_proj_key = f"text_down_proj_{layer_idx}"
                if text_down_proj_key in pruning_masks:
                    mask = pruning_masks[text_down_proj_key].to(layer.mlp.down_proj.weight.device)
                    print(f"Applying mask to layer: {text_down_proj_key}")
                    apply_mask_to_layer(layer.mlp.down_proj, mask)
                    applied_masks[text_down_proj_key] = mask
    return applied_masks


# def apply_mask_to_layer(layer, mask):
#     """
#     Helper function to apply pruning mask to a given layer's weights without changing architecture.
#
#     Args:
#         layer: The layer to mask
#         mask: The pruning mask tensor
#     """
#     if len(mask.shape) == 1:
#         # For 1D masks, expand to match weight matrix
#         expanded_mask = mask.view(-1, 1).expand_as(layer.weight.data)
#         layer.weight.data *= expanded_mask
#
#         # Apply to bias if it exists
#         if hasattr(layer, 'bias') and layer.bias is not None:
#             layer.bias.data *= mask
#     elif len(mask.shape) == 2:
#         # For 2D masks, apply directly
#         assert mask.shape == layer.weight.shape, f"Mask shape {mask.shape} doesn't match weight shape {layer.weight.shape}"
#         layer.weight.data *= mask
#
#         # Apply to bias if it exists
#         if hasattr(layer, 'bias') and layer.bias is not None:
#             # For 2D masks, we typically want to mask based on output neurons
#             bias_mask = mask.any(dim=1)  # If any input connection is kept, keep the bias
#             layer.bias.data *= bias_mask
#     else:
#         raise ValueError(f"Unsupported mask shape: {mask.shape}")

def apply_mask_to_layer(layer, mask):
    """
    Apply neuron-level pruning mask to a given layer's weights.
    Mask is 1D where each element represents whether to prune (1) or keep (0) a neuron.
    """
    if len(mask.shape) == 1:
        # Verify mask matches number of output neurons
        assert mask.shape[0] == layer.weight.shape[0], \
            f"Mask length {mask.shape[0]} doesn't match number of output neurons {layer.weight.shape[0]}"

        # Expand mask to cover all input connections for each neuron
        expanded_mask = mask.view(-1, 1).expand_as(layer.weight.data)
        # Zero out weights where mask is 1 (pruned neurons)
        layer.weight.data *= (1 - expanded_mask)  # Note the (1 - expanded_mask)

        # Also mask the bias if it exists
        if hasattr(layer, 'bias') and layer.bias is not None:
            layer.bias.data *= (1 - mask)  # Note the (1 - mask)

def apply_pruning(model, pruning_masks):
    """
    Apply pruning to the model based on provided pruning masks.

    Args:
        model: The model to prune.
        pruning_masks: Dictionary of pruning masks for each layer.
                       {layer_name: tensor_of_mask}
    """
    for name, param in model.named_parameters():
        if name in pruning_masks:  # Match parameter name directly to pruning mask
            mask = pruning_masks[name].to(param.device)
            if "weight" in name:
                # Ensure the mask has the same shape as the parameter
                if mask.shape != param.data.shape:
                    raise ValueError(f"Shape mismatch for mask and parameter {name}: {mask.shape} vs {param.data.shape}")
                param.data *= mask
            elif "bias" in name:
                # For biases, ensure mask shape matches
                if mask.shape != param.data.shape:
                    raise ValueError(f"Shape mismatch for bias mask and parameter {name}: {mask.shape} vs {param.data.shape}")
                param.data *= mask
    print("Pruning completed!")


def log_memory(prefix=""):
    """
    Log GPU and CPU memory usage.
    Args:
        prefix: String to identify the log context.
    """
    import torch
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    print(f"{prefix} | GPU Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB | "
          f"GPU Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")


def log_cpu_memory(prefix="CPU Memory"):
    # Get total, used, and free memory in GB
    memory = psutil.virtual_memory()
    total_memory = memory.total / 1e9
    used_memory = memory.used / 1e9
    free_memory = memory.available / 1e9

    # Print CPU memory usage
    print(f"{prefix} | Total: {total_memory:.2f} GB | Used: {used_memory:.2f} GB | Free: {free_memory:.2f} GB")

    # Get CPU utilization percentage
    cpu_percent = psutil.cpu_percent(interval=0.1)
    print(f"{prefix} | CPU Utilization: {cpu_percent:.2f}%")


def collect_feedforward_activations_single_batch(model, collector, dataloader, modality, model_type, device=None):
    """
    Collect activations for a single batch to debug OOM issues.
    Args:
        model: Multimodal model.
        collector: ActivationCollector instance.
        dataloader: PyTorch DataLoader for the specific modality.
        modality: "multimodal" or "unimodal".
        model_type: "Llava" or "Idefics2".
        device: Device to use for inference (default: None, uses model's device).
    Returns:
        collector.activations: Collected activations for the specified modality.
    """
    collector.clear_activations()  # Clear previous activations
    model.eval()  # Set the model to evaluation mode

    # Use the model's device if no specific device is provided
    device = device or next(model.parameters()).device

    # Process only the first batch
    batch = next(iter(dataloader))

    # Ensure inputs are on the correct device
    if modality == "multimodal":
        if model_type == "Llava":
            input_ids, attention_mask, pixel_values, labels = batch
            inputs = {
                "input_ids": input_ids.to(device),
                "attention_mask": attention_mask.to(device),
                "pixel_values": pixel_values.to(device) if pixel_values is not None else None,
                "labels": labels.to(device) if labels is not None else None,
            }
    elif modality == "unimodal":
        if model_type == "Llava":
            input_ids, attention_mask, labels = batch
            inputs = {
                "input_ids": input_ids.to(device),
                "attention_mask": attention_mask.to(device),
                "labels": labels.to(device) if labels is not None else None,
            }

    # Forward pass
    with torch.no_grad():
        if model_type == "Llava":
            if modality == "multimodal":
                model(**inputs)
            elif modality == "unimodal":
                model.language_model(**inputs)

    return collector.activations


def collect_feedforward_activations_multiple_batches(
    model, collector, dataloader, modality, model_type, device=None, num_batches=5
):
    """
    Collect activations for multiple batches with improved logic and memory management.
    Args:
        model: Multimodal model.
        collector: ActivationCollector instance.
        dataloader: PyTorch DataLoader for the specific modality.
        modality: "multimodal" or "unimodal".
        model_type: "Llava" or "Idefics2".
        device: Device to use for inference (default: None, uses model's device).
        num_batches: Number of batches to process (default: 5).
    Returns:
        collector.activations: Dictionary of collected activations for the specified modality.
    """
    collector.clear_activations()  # Clear any previous activations
    model.eval()  # Set the model to evaluation mode

    # Use the model's device if not explicitly provided
    # device = device or next(model.parameters()).device
    device = device or next(model.parameters()).device
    num_batches = len(dataloader) if num_batches is None else num_batches

    # Loop over the specified number of batches
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= num_batches:
            break  # Stop after processing the specified number of batches

        print(f"Processing batch {batch_idx + 1}/{num_batches}...")

        # Move batch data to the appropriate device
        if modality == "multimodal":
            if model_type == "Llava":
                input_ids, attention_mask, pixel_values, labels = batch
                inputs = {
                    "input_ids": input_ids.to(device),
                    "attention_mask": attention_mask.to(device),
                    "pixel_values": pixel_values.to(device) if pixel_values is not None else None,
                    "labels": labels.to(device) if labels is not None else None,
                }
        elif modality == "unimodal":
            if model_type == "Llava":
                input_ids, attention_mask, labels = batch
                inputs = {
                    "input_ids": input_ids.to(device),
                    "attention_mask": attention_mask.to(device),
                    "labels": labels.to(device) if labels is not None else None,
                }
        else:
            raise ValueError(f"Unsupported modality: {modality}")

        # Perform a forward pass to collect activations
        with torch.no_grad():
            if model_type == "Llava":
                if modality == "multimodal":
                    model(**inputs)
                elif modality == "unimodal":
                    model.language_model(**inputs)

        torch.cuda.empty_cache()  # Clear GPU cache

        # gc.collect()  # Trigger Python's garbage collector
        #
        # # Log memory usage after processing the batch
        # print(f"Batch {batch_idx + 1}/{num_batches} processed.")
        # log_memory(f"After batch {batch_idx + 1}")
        # log_cpu_memory("After batch processing")

    return collector.activations


def count_pruned_parameters(model, model_type, masks=None):
    """
    Count the percentage of pruned parameters in the model, either from existing weights
    or from provided masks.

    Args:
        model: The model to analyze
        masks: Optional dictionary of pruning masks

    Returns:
        stats_dict: Dictionary containing pruning statistics for each layer and overall
    """
    stats_dict = {}
    total_params = 0
    total_pruned = 0

    # Helper function to count zeros in a layer
    def count_zeros_and_total(weight_tensor, mask=None):
        if mask is not None:
            # Count based on mask
            if len(mask.shape) == 1:
                # For 1D masks, count effect on full weight matrix
                expanded_mask = mask.view(-1, 1).expand_as(weight_tensor)
                zeros = (expanded_mask == 0).sum().item()
            else:
                zeros = (mask == 0).sum().item()
            total = mask.numel()
        else:
            # Count based on actual weights
            zeros = (weight_tensor == 0).sum().item()
            total = weight_tensor.numel()
        return zeros, total

    # Vision model layers
    if model_type == "Llava":
        for layer_idx, layer in enumerate(model.vision_tower.vision_model.encoder.layers):
            if hasattr(layer.mlp, 'fc1'):
                vision_fc1_key = f"vision_fc1_{layer_idx}"
                mask = masks.get(vision_fc1_key) if masks else None
                zeros, total = count_zeros_and_total(layer.mlp.fc1.weight, mask)
                stats_dict[vision_fc1_key] = {
                    'total_params': total,
                    'pruned_params': zeros,
                    'pruned_percentage': (zeros / total) * 100 if total > 0 else 0
                }
                total_params += total
                total_pruned += zeros

            if hasattr(layer.mlp, 'fc2'):
                vision_fc2_key = f"vision_fc2_{layer_idx}"
                mask = masks.get(vision_fc2_key) if masks else None
                zeros, total = count_zeros_and_total(layer.mlp.fc2.weight, mask)
                stats_dict[vision_fc2_key] = {
                    'total_params': total,
                    'pruned_params': zeros,
                    'pruned_percentage': (zeros / total) * 100 if total > 0 else 0
                }
                total_params += total
                total_pruned += zeros

        # Language model layers
        for layer_idx, layer in enumerate(model.language_model.model.layers):
            if hasattr(layer.mlp, 'gate_proj'):
                lang_gate_proj_key = f"lang_gate_proj_{layer_idx}"
                mask = masks.get(lang_gate_proj_key) if masks else None
                zeros, total = count_zeros_and_total(layer.mlp.gate_proj.weight, mask)
                stats_dict[lang_gate_proj_key] = {
                    'total_params': total,
                    'pruned_params': zeros,
                    'pruned_percentage': (zeros / total) * 100 if total > 0 else 0
                }
                total_params += total
                total_pruned += zeros

            if hasattr(layer.mlp, 'up_proj'):
                lang_up_proj_key = f"lang_up_proj_{layer_idx}"
                mask = masks.get(lang_up_proj_key) if masks else None
                zeros, total = count_zeros_and_total(layer.mlp.up_proj.weight, mask)
                stats_dict[lang_up_proj_key] = {
                    'total_params': total,
                    'pruned_params': zeros,
                    'pruned_percentage': (zeros / total) * 100 if total > 0 else 0
                }
                total_params += total
                total_pruned += zeros

            if hasattr(layer.mlp, 'down_proj'):
                lang_down_proj_key = f"lang_down_proj_{layer_idx}"
                mask = masks.get(lang_down_proj_key) if masks else None
                zeros, total = count_zeros_and_total(layer.mlp.down_proj.weight, mask)
                stats_dict[lang_down_proj_key] = {
                    'total_params': total,
                    'pruned_params': zeros,
                    'pruned_percentage': (zeros / total) * 100 if total > 0 else 0
                }
                total_params += total
                total_pruned += zeros


    elif model_type == "Idefics2":
        # Vision model layers

        for layer_idx, layer in enumerate(model.model.vision_model.encoder.layers):
            if hasattr(layer.mlp, 'fc1'):
                vision_fc1_key = f"vision_fc1_{layer_idx}"
                mask = masks.get(vision_fc1_key) if masks else None
                zeros, total = count_zeros_and_total(layer.mlp.fc1.weight, mask)
                stats_dict[vision_fc1_key] = {
                    'total_params': total,
                    'pruned_params': zeros,
                    'pruned_percentage': (zeros / total) * 100 if total > 0 else 0
                }
                total_params += total
                total_pruned += zeros

            if hasattr(layer.mlp, 'fc2'):
                vision_fc2_key = f"vision_fc2_{layer_idx}"
                mask = masks.get(vision_fc2_key) if masks else None
                zeros, total = count_zeros_and_total(layer.mlp.fc2.weight, mask)
                stats_dict[vision_fc2_key] = {
                    'total_params': total,
                    'pruned_params': zeros,
                    'pruned_percentage': (zeros / total) * 100 if total > 0 else 0
                }
                total_params += total
                total_pruned += zeros

        # Text model layers
        for layer_idx, layer in enumerate(model.model.text_model.layers):

            if hasattr(layer.mlp, 'gate_proj'):
                text_gate_proj_key = f"text_gate_proj_{layer_idx}"
                mask = masks.get(text_gate_proj_key) if masks else None
                zeros, total = count_zeros_and_total(layer.mlp.gate_proj.weight, mask)
                stats_dict[text_gate_proj_key] = {
                    'total_params': total,
                    'pruned_params': zeros,
                    'pruned_percentage': (zeros / total) * 100 if total > 0 else 0

                }
                total_params += total
                total_pruned += zeros
            if hasattr(layer.mlp, 'up_proj'):
                text_up_proj_key = f"text_up_proj_{layer_idx}"
                mask = masks.get(text_up_proj_key) if masks else None
                zeros, total = count_zeros_and_total(layer.mlp.up_proj.weight, mask)
                stats_dict[text_up_proj_key] = {
                    'total_params': total,
                    'pruned_params': zeros,
                    'pruned_percentage': (zeros / total) * 100 if total > 0 else 0
                }
                total_params += total
                total_pruned += zeros
            if hasattr(layer.mlp, 'down_proj'):
                text_down_proj_key = f"text_down_proj_{layer_idx}"
                mask = masks.get(text_down_proj_key) if masks else None
                zeros, total = count_zeros_and_total(layer.mlp.down_proj.weight, mask)
                stats_dict[text_down_proj_key] = {
                    'total_params': total,
                    'pruned_params': zeros,
                    'pruned_percentage': (zeros / total) * 100 if total > 0 else 0
                }
                total_params += total
                total_pruned += zeros

    # Add overall statistics
    stats_dict['overall'] = {
        'total_params': total_params,
        'pruned_params': total_pruned,
        'pruned_percentage': (total_pruned / total_params) * 100 if total_params > 0 else 0
    }

    return stats_dict


def print_pruning_stats(stats_dict):
    """
    Pretty print the pruning statistics.

    Args:
        stats_dict: Dictionary of pruning statistics from count_pruned_parameters
    """
    print("\nPruning Statistics:")
    print("-" * 80)
    print(f"{'Layer':<30} {'Total Params':<15} {'Pruned Params':<15} {'Pruned %':<10}")
    print("-" * 80)

    for layer_name, stats in stats_dict.items():
        if layer_name != 'overall':
            print(
                f"{layer_name:<30} {stats['total_params']:<15} {stats['pruned_params']:<15} {stats['pruned_percentage']:.2f}%")

    print("-" * 80)
    overall = stats_dict['overall']
    print(
        f"{'Overall':<30} {overall['total_params']:<15} {overall['pruned_params']:<15} {overall['pruned_percentage']:.2f}%")


# def collect_feedforward_activations(
#     model, collector, dataloader, modality, model_type, device=None, num_batches=5
# ):
#     """
#     Collect activations and compute importance scores batch-by-batch.
#     """
#     collector.clear_activations()
#     model.eval()
#     device = device or next(model.parameters()).device
#     num_batches = len(dataloader) if num_batches is None else num_batches
#
#     # Store metrics for each batch
#     batch_metrics = defaultdict(lambda: defaultdict(list))
#
#     for batch_idx, batch in enumerate(dataloader):
#         if batch_idx >= num_batches:
#             break
#
#         print(f"Processing batch {batch_idx + 1}/{num_batches}...")
#
#         # Prepare inputs
#         if modality == "multimodal":
#             input_ids, attention_mask, pixel_values, labels = batch
#             inputs = {
#                 "input_ids": input_ids.to(device),
#                 "attention_mask": attention_mask.to(device),
#                 "pixel_values": pixel_values.to(device),
#                 "labels": labels.to(device) if labels is not None else None,
#             }
#         elif modality == "unimodal":
#             input_ids, attention_mask, labels = batch
#             inputs = {
#                 "input_ids": input_ids.to(device),
#                 "attention_mask": attention_mask.to(device),
#                 "labels": labels.to(device) if labels is not None else None,
#             }
#
#         # Forward pass
#         with torch.no_grad():
#             if model_type == "Llava":
#                 if modality == "multimodal":
#                     model(**inputs)
#                 elif modality == "unimodal":
#                     model.language_model(**inputs)
#
#         # Compute metrics for each layer
#         for layer_name in collector.list_collected_layers(modality=modality):
#             batch_activations = collector.get_activations(layer_name, modality=modality, to_cpu=True)
#             print(f"Layer: {layer_name}, Batch {batch_idx + 1}, Activation Shape: {batch_activations.shape}")
#
#             # Compute metrics for the batch
#             batch_metrics["I_abs"][layer_name].append(batch_activations.abs().mean(dim=0))
#             batch_metrics["I_freq"][layer_name].append((batch_activations.abs() > 1e-3).float().mean(dim=0))
#             batch_metrics["I_var"][layer_name].append(batch_activations.var(dim=0))
#             batch_metrics["I_rms"][layer_name].append(torch.sqrt((batch_activations**2).mean(dim=0)))
#
#         collector.clear_activations()
#         torch.cuda.empty_cache()
#
#     # Aggregate metrics across all batches
#     final_scores = {}
#     for metric, layer_dict in batch_metrics.items():
#         final_scores[metric] = {}
#         for layer_name, scores in layer_dict.items():
#             # print(f"Aggregating scores for {modality} in layer {layer_name}")
#             # print(scores)
#             final_scores[metric][layer_name] = torch.cat(scores).mean(dim=0)
#
#     for metric, layers in final_scores.items():
#         print(f"Checking {metric}")
#         for layer_name, values in layers.items():
#             print(f"Layer {layer_name}: Min {values.min()}, Max {values.max()}, Mean {values.mean()}")
#
#     return final_scores
# def compute_absolute_importance(collector, modality):
#     """
#     Compute absolute activation magnitude (I_abs) for a single batch.
#     Args:
#         collector: ActivationCollector instance with stored activations.
#         modality: "unimodal" or "multimodal".
#     Returns:
#         importance_scores: Dictionary of importance scores for each layer.
#     """
#     importance_scores = {}
#
#     # Iterate through all layers with collected activations
#     for layer_name in collector.list_collected_layers(modality=modality):
#         print(f"Computing importance for {modality} in layer {layer_name}")
#
#         # Compute mean absolute activation for the batch
#         batch_activations = torch.cat(collector.activations[modality][layer_name], dim=0).cpu()
#         importance_scores[layer_name] = batch_activations.abs().mean(dim=0)
#
#     return importance_scores

# def compute_frequency_importance(collector, modality):
#     """
#     Compute frequency-based activation (I_freq) for one batch of activations.
#     Args:
#         collector: ActivationCollector instance with stored activations.
#         modality: "unimodal" or "multimodal".
#     Returns:
#         importance_scores: Dictionary of frequency importance scores for each layer.
#     """
#     importance_scores = {}
#
#     for layer_name in collector.list_collected_layers(modality=modality):
#         print(f"Computing frequency importance for {modality} in layer {layer_name}")
#         # Retrieve activations for the batch
#         threshold = 1e-6  # Small value to filter near-zero activations
#         batch_activations = torch.cat(collector.activations[modality][layer_name], dim=0).cpu()
#         importance_scores[layer_name] = (batch_activations.abs() > threshold).float().mean(dim=0)
#         # Compute frequency of non-zero activations
#         # batch_activations = collector.activations[modality][layer_name][0].cpu()
#         # importance_scores[layer_name] = (batch_activations != 0).float().mean(dim=0)
#
#     return importance_scores
#
#
# def compute_variance_importance(collector, modality):
#     """
#     Compute variance of activations (I_var) for one batch of activations.
#     Args:
#         collector: ActivationCollector instance with stored activations.
#         modality: "unimodal" or "multimodal".
#     Returns:
#         importance_scores: Dictionary of variance importance scores for each layer.
#     """
#     importance_scores = {}
#
#     for layer_name in collector.list_collected_layers(modality=modality):
#         print(f"Computing variance importance for {modality} in layer {layer_name}")
#         # Retrieve activations for the batch
#         batch_activations = torch.cat(collector.activations[modality][layer_name], dim=0).cpu()
#         # batch_activations = collector.activations[modality][layer_name][0].cpu()
#         importance_scores[layer_name] = batch_activations.var(dim=0)
#
#     return importance_scores
#
#
# def compute_rms_importance(collector, modality):
#     """
#     Compute root mean square activation (I_rms) for one batch of activations.
#     Args:
#         collector: ActivationCollector instance with stored activations.
#         modality: "unimodal" or "multimodal".
#     Returns:
#         importance_scores: Dictionary of RMS importance scores for each layer.
#     """
#     importance_scores = {}
#
#     for layer_name in collector.list_collected_layers(modality=modality):
#         print(f"Computing RMS importance for {modality} in layer {layer_name}")
#         # Retrieve activations for the batch
#         # batch_activations = collector.activations[modality][layer_name][0].cpu()
#         batch_activations = torch.cat(collector.activations[modality][layer_name], dim=0).cpu()
#         importance_scores[layer_name] = torch.sqrt((batch_activations**2).mean(dim=0))
#
#     return importance_scores
