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
from transformers import BitsAndBytesConfig, LlavaForConditionalGeneration, AutoProcessor, get_scheduler, AdamW, \
    LlavaNextForConditionalGeneration, LlavaNextProcessor, Idefics2ForConditionalGeneration, \
    MllamaForConditionalGeneration, MllamaProcessor, AutoTokenizer
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
import json
from data_process.data_preprocess import Vanilla_LLaVA_Dataset, train_collate_fn_llava, train_collate_fn, \
    train_collate_fn_idefics
import matplotlib.pyplot as plt
from PIL import Image
from accelerate import Accelerator
from transformers import AutoProcessor
from transformers import BitsAndBytesConfig, LlavaForConditionalGeneration
import torch
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from transformers import Trainer, TrainingArguments
# from trl import SFTConfig, SFTTrainer
import random
from torch.utils.data import Subset

def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['multi_modal_projector', 'vision_model']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


# Example usage:
def load_model_and_processor(model_id):
    """
    Load the model and processor based on the provided model_id.
    Different models may require different loading methods, which are handled with conditional statements.
    """
    if model_id.startswith("llava"):
        # Load LLAVA model and processor
        print("Loading LLAVA model...")
        model = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            # quantization_config=bnb_config,
            cache_dir="/afs/crc.nd.edu/group/dmsquare/vol1/zliu29/mllm_unlearn/model/llava-1.5-7b-hf",
        )
        processor = AutoProcessor.from_pretrained(model_id)
        # Additional processor configuration if necessary
        processor.tokenizer.padding_side = "right"  # Ensure right padding
        processor.tokenizer.add_tokens(["<image>", "<pad>"], special_tokens=True)

    elif model_id.startswith("HuggingFaceM4"):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        # Load LLAVA Next model and processor
        print("Loading idefics2 model...")
        model = Idefics2ForConditionalGeneration.from_pretrained(
            "HuggingFaceM4/idefics2-8b",
            torch_dtype=torch.float16,
            device_map="auto",
            # quantization_config=bnb_config,
            low_cpu_mem_usage=True,
            cache_dir="/afs/crc.nd.edu/group/dmsquare/vol1/zliu29/mllm_unlearn/model/idfics2-8b",
        )
        processor = AutoProcessor.from_pretrained(
            "HuggingFaceM4/idefics2-8b",
            do_image_splitting=False
        )
        # Additional processor configuration if necessary
        processor.tokenizer.padding_side = "right"  # Ensure right padding
        processor.tokenizer.add_tokens(["<image>", "<pad>"], special_tokens=True)

    else:
        raise ValueError("Model ID not recognized or not supported. Please provide a valid model ID.")

    return model, processor



######################### Accelerate Version #################################
def main(args):
    # Load model and processor
    print("Trainer Status is ", args.trainer)
    model, processor = load_model_and_processor(args.model_id)
    print("Processor Tokenizer Length: ", len(processor.tokenizer)) #128257

    ################## Update #########################
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    print("Tokenizer Length: ", len(tokenizer))


    # Resize token embeddings to match the tokenizer
    # if args.models_id.startswith("meta-llama") == False:
    #     model.resize_token_embeddings(len(processor.tokenizer))

    if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        print("WARNING: Resizing the embedding matrix to match the tokenizer vocab size.")
        model.resize_token_embeddings(len(tokenizer))

    ################## Update #########################


    # LoRA configuration
    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        # target_modules=["q_proj", "v_proj"],
        target_modules=find_all_linear_names(model),
        init_lora_weights="gaussian",
    )

    print("getting peft model")
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # model.add_adapter(lora_config)
    # model.enable_adapters()
    if isinstance(model, PeftModel):
        print("This is a PEFT model.")
    else:
        print("This is NOT a PEFT model.")

    df = pd.read_parquet(args.data_dir)
    dataset = Vanilla_LLaVA_Dataset(df=df)


    if args.model_id.startswith("llava"):
        train_dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=lambda x: train_collate_fn_llava(x, processor, args)
        )
    elif args.model_id.startswith("HuggingFaceM4"):
        train_dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=lambda x: train_collate_fn_idefics(x, processor, args)
        )
    else:
        raise ValueError("Model ID not recognized or not supported. Please provide a valid model ID.")

    # Accelerator setup
    accelerator = Accelerator()
    if args.gradient_accumulation:
        print("Gradient accumulation enabled.")
        accumulation_steps = 4  # Adjust based on memory
        model.gradient_checkpointing_enable()
    else:
        print("Gradient accumulation disabled.")

    optimizer = AdamW(model.parameters(), lr=args.lr)

    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=len(train_dataloader) * args.num_epochs,
    )

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")
        if args.gradient_accumulation:
            for step, batch in enumerate(progress_bar):
                input_ids, attention_mask, pixel_values, labels = batch
                with accelerator.accumulate(model):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                                    pixel_values=pixel_values, labels=labels)
                    loss = outputs.loss / accumulation_steps
                    # loss = outputs.loss
                    accelerator.backward(loss)
                    if (step + 1) % accumulation_steps == 0:
                        accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)  #
                        optimizer.step()
                        optimizer.zero_grad()
                        lr_scheduler.step()
                total_loss += loss.item()
                progress_bar.set_postfix(loss=total_loss / len(progress_bar))
            # lr_scheduler.step()
            print(f"Epoch {epoch + 1} Loss: {total_loss / len(train_dataloader)}")
        else:
            for batch in progress_bar:
                input_ids, attention_mask, pixel_values, labels = batch
                outputs = model(input_ids=input_ids,
                                attention_mask=attention_mask,
                                pixel_values=pixel_values,
                                labels=labels)
                loss = outputs.loss
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()
                total_loss += loss.item()
                progress_bar.set_postfix(loss=total_loss / len(progress_bar))

            print(f"Epoch {epoch + 1} Loss: {total_loss / len(train_dataloader)}")

    # Save the final model
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    # if args.model_id.startswith("meta-llama") == False:
    unwrapped_model = unwrapped_model.merge_and_unload()
    unwrapped_model.save_pretrained(args.save_dir)
    print(f"Model saved to: {args.save_dir}")


if __name__ == "__main__":
    # Argument parser for different options
    parser = argparse.ArgumentParser(description="Fine-tune different models")
    parser.add_argument("--model_id", type=str, required=True, help="Pretrained model ID")
    parser.add_argument("--save_dir", type=str, default="./saved_model", help="Directory to save the model")
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory to save the model")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of epochs for training")
    parser.add_argument("--max_length", type=int, default=384, help="Maximum sequence length")
    parser.add_argument("--gradient_accumulation", type=bool, default=False, help="Enable gradient accumulation")
    parser.add_argument("--trainer", type=bool, default=False, help="Use HuggingFace Trainer")

    args = parser.parse_args()

    # Call main function
    main(args)

    # llava save: /afs/crc.nd.edu/group/dmsquare/vol1/zliu29/mllm_unlearn/saved_model/LLaVA_7b_hf_vanilla
    # llava hf: llava-hf/llava-1.5-7b-hf
    # llava cache: /afs/crc.nd.edu/group/dmsquare/vol1/zliu29/mllm_unlearn/model/llava-1.5-7b-hf
    # llava max_length 384

    # idefics2 save: /afs/crc.nd.edu/group/dmsquare/vol1/zliu29/mllm_unlearn/saved_model/idefics2_8b_vanilla
    # idefics2 hf: HuggingFaceM4/idefics2-8b
    # idefics2 cache: /afs/crc.nd.edu/group/dmsquare/vol1/zliu29/mllm_unlearn/model/idfics2-8b
    # idefics2 max_length 384

