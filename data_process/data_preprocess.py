import pandas as pd
import copy
import json
from typing import Any, Dict
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import AutoProcessor
import os
from io import BytesIO
from PIL import Image
import re
import torch
from torch.utils.data import DataLoader
import spacy


class Vanilla_LLaVA_Dataset(Dataset):
    """
    PyTorch Dataset for LLaVA fine-tuning. This class loads data directly from a DataFrame loaded
    from a Parquet file and returns them in a structure similar to Hugging Face datasets.
    """

    def __init__(self, df: pd.DataFrame, target_size=None, sort_json_key: bool = True):
        """
        Args:
            df (pd.DataFrame): DataFrame containing the Parquet data.
            target_size (tuple or None): The target size for resizing images (width, height). If None, retain the original size.
            sort_json_key (bool): Whether to sort the JSON keys when converting to tokens. Defaults to True.
        """
        super().__init__()
        self.df = df
        self.target_size = target_size  # Target size for resizing images (None means no resizing)
        self.sort_json_key = sort_json_key
        # Flatten the dataset to create a list of individual QA pairs with associated images
        self.dataset = self.flatten_dataset()

    def flatten_dataset(self):
        """
        Flatten the dataset such that each question-answer pair becomes a single item.
        Returns:
            flattened_data (list): List of dictionaries with image data and each QA pair.
        """
        flattened_data = []

        for idx, row in self.df.iterrows():
            # Extract the bytes from the 'image' dictionary
            image_data = row['image'].get('bytes')  # Access the image bytes

            # Convert the image bytes to a PIL Image
            try:
                image = Image.open(BytesIO(image_data)).convert("RGB")
            except Exception as e:
                print(f"Error loading image at index {idx}: {e}")
                continue

            # Safely load metadata as JSON
            try:
                metadata = json.loads(row['metadata'])  # Using json.loads to parse JSON safely
            except json.JSONDecodeError as e:
                print(f"Error decoding metadata at index {idx}: {e}")
                continue
            for qa_pair in metadata:
                question = qa_pair.get("Question", "")
                answer = qa_pair.get("Answer", "")

                if question and answer:
                    flattened_data.append({
                        "image": image,
                        "question": question,
                        "answer": answer
                    })
        # print(flattened_data)
        return flattened_data
    def resize_image(self, image):
        """
        Resizes the image to the target size if specified.
        Args:
            image (PIL.Image.Image): The input image to resize.
        Returns:
            PIL.Image.Image: The resized image if target_size is set, otherwise the original image.
        """
        if self.target_size is not None:
            return image.resize(self.target_size, Image.Resampling.LANCZOS)
        return image  # Return original image if target_size is None

    def __len__(self):
        return len(self.dataset)

    def json2token(self, obj: Any, sort_json_key: bool = True):
        """
        Converts a JSON object into a tokenized string sequence by recursively processing each key-value pair.
        """
        if isinstance(obj, dict):
            if len(obj) == 1 and "text_sequence" in obj:
                return obj["text_sequence"]
            else:
                output = ""
                keys = sorted(obj.keys(), reverse=True) if sort_json_key else obj.keys()
                for k in keys:
                    output += f"<s_{k}>" + self.json2token(obj[k], sort_json_key) + f"</s_{k}>"
                return output
        elif isinstance(obj, list):
            return "<sep/>".join([self.json2token(item, sort_json_key) for item in obj])
        else:
            return str(obj)

    def __getitem__(self, idx: int):
        """
        Returns one item from the dataset.

        Returns:
            dict: A dictionary containing:
                  - image: The preprocessed and resized image.
                  - question: The tokenized question.
                  - answer: The tokenized answer.
        """
        sample = self.dataset[idx]

        # Get the image and resize it if necessary
        image = self.resize_image(sample["image"])

        # Get the question and answer
        question = sample.get("question", "")
        answer = sample.get("answer", "")

        # Tokenize the question and answer
        tokenized_question = self.json2token(question, sort_json_key=self.sort_json_key)
        tokenized_answer = self.json2token(answer, sort_json_key=self.sort_json_key)

        return {
            "image": image,
            "question": tokenized_question,
            "answer": tokenized_answer
        }


class LLAVA_multimodal_Dataset(Dataset):
    """
    PyTorch Dataset for LLaVA fine-tuning. This class loads data directly from a DataFrame loaded
    from a Parquet file and returns them in a structure similar to Hugging Face datasets.
    """

    def __init__(self, df: pd.DataFrame, target_size=None, sort_json_key: bool = True):
        """
        Args:
            df (pd.DataFrame): DataFrame containing the Parquet data.
            target_size (tuple or None): The target size for resizing images (width, height). If None, retain the original size.
            sort_json_key (bool): Whether to sort the JSON keys when converting to tokens. Defaults to True.
        """
        super().__init__()
        self.df = df
        self.target_size = target_size  # Target size for resizing images (None means no resizing)
        self.sort_json_key = sort_json_key
        # Flatten the dataset to create a list of individual QA pairs with associated images
        self.dataset = self.flatten_dataset()

    def flatten_dataset(self):
        """
        Flatten the dataset such that each question-answer pair becomes a single item.
        Returns:
            flattened_data (list): List of dictionaries with image data and each QA pair.
        """
        flattened_data = []

        for idx, row in self.df.iterrows():
            # Extract the bytes from the 'image' dictionary
            image_data = row['image'].get('bytes')  # Access the image bytes

            # Convert the image bytes to a PIL Image
            try:
                image = Image.open(BytesIO(image_data)).convert("RGB")
            except Exception as e:
                print(f"Error loading image at index {idx}: {e}")
                continue

            # Safely load metadata as JSON
            try:
                metadata = json.loads(row['metadata'])  # Using json.loads to parse JSON safely
            except json.JSONDecodeError as e:
                print(f"Error decoding metadata at index {idx}: {e}")
                continue
            for qa_pair in metadata:
                question = qa_pair.get("Question", "")
                answer = qa_pair.get("Answer", "")

                if question and answer:
                    flattened_data.append({
                        "image": image,
                        "question": question,
                        "answer": answer
                    })
        # print(flattened_data)
        return flattened_data
    def resize_image(self, image):
        """
        Resizes the image to the target size if specified.
        Args:
            image (PIL.Image.Image): The input image to resize.
        Returns:
            PIL.Image.Image: The resized image if target_size is set, otherwise the original image.
        """
        if self.target_size is not None:
            return image.resize(self.target_size, Image.Resampling.LANCZOS)
        return image  # Return original image if target_size is None

    def __len__(self):
        return len(self.dataset)

    def json2token(self, obj: Any, sort_json_key: bool = True):
        """
        Converts a JSON object into a tokenized string sequence by recursively processing each key-value pair.
        """
        if isinstance(obj, dict):
            if len(obj) == 1 and "text_sequence" in obj:
                return obj["text_sequence"]
            else:
                output = ""
                keys = sorted(obj.keys(), reverse=True) if sort_json_key else obj.keys()
                for k in keys:
                    output += f"<s_{k}>" + self.json2token(obj[k], sort_json_key) + f"</s_{k}>"
                return output
        elif isinstance(obj, list):
            return "<sep/>".join([self.json2token(item, sort_json_key) for item in obj])
        else:
            return str(obj)

    def __getitem__(self, idx: int):
        """
        Returns one item from the dataset.

        Returns:
            dict: A dictionary containing:
                  - image: The preprocessed and resized image.
                  - question: The tokenized question.
                  - answer: The tokenized answer.
        """
        sample = self.dataset[idx]

        # Get the image and resize it if necessary
        image = self.resize_image(sample["image"])

        # Get the question and answer
        question = sample.get("question", "")
        answer = sample.get("answer", "")

        # Tokenize the question and answer
        tokenized_question = self.json2token(question, sort_json_key=self.sort_json_key)
        tokenized_answer = self.json2token(answer, sort_json_key=self.sort_json_key)

        return {
            "image": image,
            "question": tokenized_question,
            "answer": tokenized_answer
        }


class LLAVA_unimodal_Dataset(Dataset):
    """
    PyTorch Dataset for text-only inputs. This class loads data directly from a DataFrame
    and outputs question-answer pairs with all occurrences of "this person" replaced
    by the person's name.
    """
    def __init__(self, df: pd.DataFrame, sort_json_key: bool = True):
        """
        Args:
            df (pd.DataFrame): DataFrame containing the Parquet data.
            sort_json_key (bool): Whether to sort the JSON keys when converting to tokens. Defaults to True.
        """
        super().__init__()
        self.df = df
        self.sort_json_key = sort_json_key
        # Flatten the dataset to create a list of individual QA pairs
        self.dataset = self.flatten_dataset()

    def extract_person_name(self, text):
        """
        Extract the name of a person from the given text.
        Handles cases:
        - Extracts the name after the exact word "is"
        - Extracts the name after "named"
        - Returns the entire text if it directly contains the name
        """
        text = text.strip().rstrip(".")  # Remove leading/trailing spaces and trailing period

        # Handle the "named" case
        if "named" in text:
            name_start = text.index("named") + len("named")
            return text[name_start:].strip()

        # Handle the "is" case using regex for exact word match
        match = re.search(r"\bis\b (.+)", text)
        if match:
            return match.group(1).strip()

        # If no "is" or "named", assume the entire text is the name
        return text

    def replace_placeholders(self, text, person_name):
        """
        Replaces all placeholders in the text with the person's name.

        Args:
            text (str): The original text (question or answer).
            person_name (str): The person's name to insert.

        Returns:
            str: The updated text with placeholders replaced.
        """
        # Define placeholder rules
        placeholders = {
            "this person": person_name,
            "the person": person_name,
            "The person": person_name,
            "This person": person_name,
            "this person's": f"{person_name}'s",
            "This person's": f"{person_name}'s",
            "this individual": person_name,
            "This individual": person_name,
            "the individual": person_name,
            "The individual": person_name,
            "this profile": person_name,
        }

        # Replace placeholders dynamically
        for placeholder, replacement in placeholders.items():
            text = text.replace(placeholder, replacement)

        return text

    def flatten_dataset(self):
        """
        Flatten the dataset such that each question-answer pair becomes a single item.
        Modifies the question and answer to replace "this person" with the person's name.
        Returns:
            flattened_data (list): List of dictionaries with text-only question-answer pairs.
        """
        all_names = []
        total_count = 0  # To verify we have names for all JSON files
        flattened_data = []
        question_list = [
            "What is the name of the person in this profile?",
            "What is this person's name as stated in the biography?",
            "What is the person's name?",
            "What is the name of the person in the image?",
            "What is this person's name?",
            "What is the full name of this person?",
            "What is the full name of the person in the image?",
            "What is the name of the person in this biography?",
            "What is the name of the person in the profile?",
            "What is the name of the person in the biography?",
            "What is the person's name as stated in the profile?",
            "What is this person's full name?",
            "What is the name of this individual?",
            "What is the name of the individual in the profile?",
            "What is the name of this person?",
            "What is the name associated with this profile?",
            "What is the name of the individual in the image?"
        ]

        for idx, row in self.df.iterrows():
            # Safely load metadata as JSON
            try:
                metadata = json.loads(row['metadata'])  # Parse metadata as JSON
            except json.JSONDecodeError as e:
                print(f"Error decoding metadata at index {idx}: {e}")
                continue

            # Find the person's name from the corresponding QA pair
            person_name = None
            for qa_pair in metadata:
                question = qa_pair.get("Question", "").strip()
                answer = qa_pair.get("Answer", "").strip()

                # Extract the person's name if the question matches
                if question in question_list:
                    person_name = self.extract_person_name(answer)

                    # print(f"Person's name: {person_name}")
                    if person_name:
                        all_names.append(person_name)
                        total_count += 1
                        break  # Stop after finding the first matching question
            # Process each QA pair and replace "this person" with the person's name
            for qa_pair in metadata:
                question = qa_pair.get("Question", "").strip()
                answer = qa_pair.get("Answer", "").strip()

                # Skip this question-answer pair if it was used to extract the name
                if question in question_list and self.extract_person_name(answer) == person_name:
                    continue

                if question and answer:
                    # Replace "this person" in the question and answer with the person's name
                    # updated_question = (
                    #     question.replace("this person", person_name)
                    #     .replace("This person", person_name)
                    #     .replace("this person's", f"{person_name}'s")
                    #     .replace("This person's", f"{person_name}'s")
                    #     .replace("this individual", person_name)
                    #     .replace("the individual", person_name)
                    #     .replace("The individual", person_name)
                    #     .replace("This individual", person_name)
                    #     .replace("this profile", person_name)
                    # )
                    # updated_answer = (
                    #     answer.replace("this person", person_name)
                    #     .replace("This person", person_name)
                    #     .replace("this person's", f"{person_name}'s")
                    #     .replace("This person's", f"{person_name}'s")
                    #     .replace("this individual", person_name)
                    #     .replace("This individual", person_name)
                    #     .replace("the individual", person_name)
                    #     .replace("The individual", person_name)
                    #     .replace("this profile", person_name)
                    # )

                    updated_question = self.replace_placeholders(question, person_name)
                    updated_answer = self.replace_placeholders(answer, person_name)

                    flattened_data.append({
                        "question": updated_question,
                        "answer": updated_answer
                    })

        print(f"Total names found: {total_count}")
        # print(flattened_data)
        return flattened_data

    def json2token(self, obj: Any, sort_json_key: bool = True):
        """
        Converts a JSON object into a tokenized string sequence by recursively processing each key-value pair.
        """
        if isinstance(obj, dict):
            if len(obj) == 1 and "text_sequence" in obj:
                return obj["text_sequence"]
            else:
                output = ""
                keys = sorted(obj.keys(), reverse=True) if sort_json_key else obj.keys()
                for k in keys:
                    output += f"<s_{k}>" + self.json2token(obj[k], sort_json_key) + f"</s_{k}>"
                return output
        elif isinstance(obj, list):
            return "<sep/>".join([self.json2token(item, sort_json_key) for item in obj])
        else:
            return str(obj)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        """
        Returns one item from the dataset.

        Returns:
            dict: A dictionary containing:
                  - question: The tokenized question.
                  - answer: The tokenized answer.
        """
        sample = self.dataset[idx]

        # Get the question and answer
        question = sample.get("question", "")
        answer = sample.get("answer", "")

        # Tokenize the question and answer
        tokenized_question = self.json2token(question, sort_json_key=self.sort_json_key)
        tokenized_answer = self.json2token(answer, sort_json_key=self.sort_json_key)

        return {
            "question": tokenized_question,
            "answer": tokenized_answer
        }


def train_collate_fn(examples, processor, max_length):
    images, texts = [], []
    for image, question, rejected_sequence in examples:
        prompt = f"USER: <image>{question}\nASSISTANT: {rejected_sequence}"
        images.append(image)
        texts.append(prompt)

    batch = processor(text=texts, images=images, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    batch["labels"] = labels

    return batch["input_ids"], batch["attention_mask"], batch["pixel_values"], batch["labels"]



def train_collate_fn_idefics(examples, processor, args, modality="multimodal"):
    """
    A data collator function for IDEFICS that processes input text and images,
    adapting for different modalities: multimodal (text + images) and unimodal (text only).
    """
    if modality == "multimodal":
        texts = []
        images = []

        for example in examples:
            image = example.get("image")
            question = example.get("question", "")
            answer = example.get("answer", "")

            # Create the conversation prompt with the image token placeholder
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": question}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": answer}
                    ]
                }
            ]

            # Convert the conversation into a text template
            text = processor.apply_chat_template(messages, add_generation_prompt=False)

            # Append the image and text to respective lists
            texts.append(text.strip())
            images.append([image])

        if len(texts) == 0 or len(images) == 0:
            raise ValueError("Empty batch. No valid images or text in the examples provided.")

        # Use the processor to prepare the batch with both text and images
        batch = processor(
            text=texts,
            images=images,
            padding=True,
            truncation=True,
            # max_length=args.max_length,
            return_tensors="pt"
        )

        # Mask labels
        labels = batch["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100

        # Assign image token ID to the labels at the appropriate positions
        image_token_id = processor.tokenizer.additional_special_tokens_ids[
            processor.tokenizer.additional_special_tokens.index("<image>")
        ]
        labels[labels == processor.tokenizer.pad_token_id] = image_token_id

        batch["labels"] = labels

        if args.trainer:
            return {
                "input_ids": batch["input_ids"],
                "attention_mask": batch["attention_mask"],
                "pixel_values": batch["pixel_values"],
                "labels": batch["labels"]
            }
        else:
            return batch["input_ids"], batch["attention_mask"], batch["pixel_values"], batch["labels"]

    if modality == "unimodal":
        texts = []

        for example in examples:
            question = example.get("question", "")
            answer = example.get("answer", "")

            # Create the conversation prompt without the image
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": answer}
                    ]
                }
            ]

            # Convert the conversation into a text template
            text = processor.apply_chat_template(messages, add_generation_prompt=False)
            texts.append(text.strip())

        if len(texts) == 0:
            raise ValueError("Empty batch. No valid text in the examples provided.")

        # Use the tokenizer to prepare the batch with only text
        batch = processor.tokenizer(
            text=texts,
            padding=True,
            truncation=True,
            # max_length=args.max_length,
            return_tensors="pt"
        )

        # Mask labels
        labels = batch["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100
        batch["labels"] = labels

        if args.trainer:
            return {
                "input_ids": batch["input_ids"],
                "attention_mask": batch["attention_mask"],
                "labels": batch["labels"]
            }
        else:
            return batch["input_ids"], batch["attention_mask"], batch["labels"]



def train_collate_fn_llava(examples, processor, args, modality="multimodal"):
    # max_length = 384
    # MODEL_ID = "llava-hf/llava-1.5-7b-hf"
    # processor = AutoProcessor.from_pretrained(MODEL_ID)
    # processor.tokenizer.padding_side = "right"  # during training, one always uses padding on the right
    # print(f"train_collate_fn_llava called with modality: {modality}")
    if modality == "multimodal":
        images = []
        texts = []

        for example in examples:
            image = example.get('image')
            question = example.get('question')
            answer = example.get('answer')
            images.append(image)

            # Construct prompt with question and answer
            prompt = f"USER: <image>\n{question}\nASSISTANT: {answer}"
            texts.append(prompt)

        if len(texts) == 0 or len(images) == 0:
            raise ValueError("Empty batch. No valid images or text in the examples provided.")

        # Process the batch
        batch = processor(
            text=texts,
            images=images,
            padding=True,
            truncation=True,
            # max_length=args.max_length,
            return_tensors="pt"
        )
        # Mask labels
        labels = batch["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100
        batch["labels"] = labels

        if args.trainer:
            return {
                "input_ids": batch["input_ids"],
                "attention_mask": batch["attention_mask"],
                "pixel_values": batch["pixel_values"],
                "labels": batch["labels"]
            }
        else:
            return batch["input_ids"], batch["attention_mask"], batch["pixel_values"], batch["labels"]

    if modality == "unimodal":
        texts = []
        for example in examples:
            # image = example.get('image')
            question = example.get('question')
            answer = example.get('answer')
            # images.append(image)

            # Construct prompt with question and answer
            prompt = f"USER:{question}\nASSISTANT: {answer}"
            texts.append(prompt)

        if len(texts) == 0:
            raise ValueError("Empty batch. No valid images or text in the examples provided.")

        # Process the batch
        batch = processor.tokenizer(
            text=texts,
            padding=True,
            truncation=True,
            # max_length=args.max_length,
            return_tensors="pt"
        )
        # Mask labels
        labels = batch["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100
        batch["labels"] = labels

        if args.trainer:
            return {
                "input_ids": batch["input_ids"],
                "attention_mask": batch["attention_mask"],
                # "pixel_values": batch["pixel_values"],
                "labels": batch["labels"]
            }
        else:
            return batch["input_ids"], batch["attention_mask"], batch["labels"]

