import os
import json
import random
from PIL import Image
from tqdm import tqdm
import torch
from transformers import LlavaForConditionalGeneration, AutoProcessor, AutoTokenizer, Idefics2ForConditionalGeneration
import pandas as pd
import random
import json
from PIL import Image
from io import BytesIO
# from transformers import LlavaForConditionalGeneration, AutoProcessor, AutoTokenizer, Idefics2ForConditionalGeneration, MllamaProcessor, MllamaForConditionalGeneration
from rouge_score import rouge_scorer
from sklearn.model_selection import train_test_split
import argparse
import fnmatch
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


def load_and_combine_parquet_files(directory):
    # Get all Parquet files in the directory
    parquet_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.parquet')]

    # Read and concatenate all Parquet files
    combined_df = pd.concat([pd.read_parquet(file) for file in parquet_files], ignore_index=True)
    return combined_df

def save_ids_to_json(parquet_file, output_folder, filename="ids.json"):
    """
    Extract IDs from a Parquet file and save them to a JSON file in the specified folder.

    Args:
        parquet_file (str): Path to the Parquet file containing the data.
        output_folder (str): Path to the folder where the JSON file will be saved.
        filename (str): Name of the JSON file. Defaults to "ids.json".
    """
    # Load the Parquet file into a DataFrame
    df = pd.read_parquet(parquet_file)

    # Extract the unique IDs
    ids = df['ID'].unique().tolist()

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Construct the full path to the JSON file
    output_json_file = os.path.join(output_folder, filename)

    # Save the IDs to a JSON file
    with open(output_json_file, 'w') as f:
        json.dump(ids, f)

    print(f"Saved IDs to {output_json_file}")

def compute_bleu(ground_truth, predicted_answer):
    """
    Compute the BLEU score between a ground truth and predicted answer using simple whitespace tokenization.

    Args:
        ground_truth (str): The correct reference answer.
        predicted_answer (str): The predicted answer from the model.

    Returns:
        float: The BLEU score.
    """
    # Use .split() to tokenize based on spaces
    reference = [ground_truth.split()]  # Reference needs to be a list of tokenized words
    hypothesis = predicted_answer.split()  # Hypothesis (predicted answer) is also tokenized

    # Use smoothing to handle cases where BLEU score could be 0 for short texts
    smoothing_function = SmoothingFunction().method1

    # Compute the BLEU score
    bleu_score = sentence_bleu(reference, hypothesis, smoothing_function=smoothing_function)

    return bleu_score

def evaluate_from_ids(id_json_file, question_folder, filename_pattern="*"):
    """
    Load IDs from the JSON file and find their corresponding evaluation question files with a specific filename pattern,
    then return a list of the loaded JSON files.

    Args:
        id_json_file (str): Path to the JSON file containing the list of IDs.
        question_folder (str): Path to the folder containing evaluation question files.
        filename_pattern (str): Filename pattern to match (e.g., "*_question.json"). Default is "*" for any file.

    Returns:
        list: A list of loaded JSON files from the question folder.
    """
    # Load the list of IDs from the ID JSON file
    with open(id_json_file, 'r') as f:
        ids = json.load(f)

    json_files = []

    # Loop through the files in the question folder
    for filename in sorted(os.listdir(question_folder)):
        # Find files that match the ID and the filename pattern
        for id_ in ids:
            if filename.startswith(id_) and fnmatch.fnmatch(filename, filename_pattern):
                file_path = os.path.join(question_folder, filename)

                # Load the matching JSON file
                with open(file_path, 'r') as f:
                    json_files.append(json.load(f))
                break  # Move to the next file after finding the match

    return json_files

def formulate_prompt_with_options(question, options):
    """
    Formulate the prompt by combining the question and its options.

    Args:
        question (str): The question text.
        options (dict): The options for the question (e.g., {"A": "Option A", "B": "Option B"}).

    Returns:
        str: The formulated prompt combining the question and options.
    """
    # Combine the question with the options
    options_str = "\n".join([f"{key}: {value}" for key, value in options.items()])
    prompt = f"{question}\n{options_str}"
    return prompt


def formulate_prompt_with_options_llama(question, options):
    """
    Formulate the prompt by combining the question and its options.

    Args:
        question (str): The question text.
        options (dict): The options for the question (e.g., {"A": "Option A", "B": "Option B"}).

    Returns:
        str: The formulated prompt combining the question and options.
    """
    # Combine the question with the options
    options_str = "\n".join([f"{key}: {value}" for key, value in options.items()])
    prompt = f"{question}\n####Choices:\n{options_str}"
    return prompt
def split_dataset(original_dataset, forget_percentage=0.3):
    forget_set_size = int(len(original_dataset) * forget_percentage)
    retain_set_size = len(original_dataset) - forget_set_size
    forget_set, retain_set = train_test_split(original_dataset, test_size=retain_set_size, random_state=42)
    return forget_set, retain_set

def load_json_files(question_folder):
    """
    Load all JSON files from the given folder.
    """
    json_files = []
    for filename in sorted(os.listdir(question_folder)):
        if filename.endswith(".json"):
            with open(os.path.join(question_folder, filename), 'r') as f:
                json_files.append(json.load(f))
    return json_files

def load_image(image_folder, image_id):
    """
    Load an image, trying both .png and .jpg extensions.
    """
    possible_extensions = ['.png', '.jpg', '.jpeg']
    for ext in possible_extensions:
        image_path = os.path.join(image_folder, f"{image_id}{ext}")
        if os.path.exists(image_path):
            try:
                image = Image.open(image_path).convert("RGB")
                return image
            except Exception as e:
                print(f"Error loading image at {image_path}: {e}")
                return None
    print(f"Image not found for ID: {image_id}")
    return None


def load_random_test_image(image_folder, image_id):
    """
    Load a random image from a folder in 'test' mode.

    Args:
        image_folder: The folder where the image_id folder is stored.
        image_id: The ID of the folder containing multiple images.

    Returns:
        image: The randomly selected image (or None if not found or error occurs).
    """
    # In 'test' mode, image_id is a folder containing multiple images
    image_dir = os.path.join(image_folder, image_id)

    if not os.path.isdir(image_dir):
        print(f"Image folder not found for ID: {image_id}")
        return None

    # List the images inside the folder
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    # Filter to match the expected naming format: {image_id}_poseX_gen1.png
    image_files = [f for f in image_files if f.startswith(image_id) and 'pose' in f]

    if not image_files:
        print(f"No valid images found in folder: {image_dir}")
        return None

    # Randomly select one image from the folder
    selected_image = random.choice(image_files)
    image_path = os.path.join(image_dir, selected_image)

    try:
        image = Image.open(image_path).convert("RGB")
        print(f"Randomly selected image: {selected_image}")
        return image
    except Exception as e:
        print(f"Error loading image at {image_path}: {e}")
        return None

def evaluate_classification(parquet_file, few_shot_parquet_file, processor, tokenizer, model, args, id_list_file=None, mode="default", forget_parquet_file=None):
    """
    Evaluate classification task with/without few-shot samples based on the mode.

    Args:
        parquet_file: Path to the main Parquet file for evaluation.
        few_shot_parquet_file: Path to the Parquet file containing few-shot examples.
        processor: The processor for handling image and text inputs.
        tokenizer: The tokenizer for decoding model outputs.
        model: The model to use for classification.
        args: Arguments object containing model ID and other configurations.
        id_list_file: (Optional) Path to the JSON file containing the list of IDs. Default is None.
        mode: Mode that controls how few-shot samples are handled ('forget', 'retain_share', 'test', or others). Default is 'default'.
        forget_parquet_file: (Optional) Path to the forget Parquet file to filter IDs for test mode.

    Returns:
        dict: A dictionary with accuracy scores.
    """
    print("################################## Classification Task Starts ##############################################")
    print(f"############################## Evaluating {mode} Mode #########################################" )

    # Load the ID list from the JSON file if provided
    if id_list_file:
        with open(id_list_file, 'r') as f:
            id_list = json.load(f)
    elif mode == "test" and forget_parquet_file:
        # Load IDs from the forget Parquet file for filtering in test mode
        forget_df = pd.read_parquet(forget_parquet_file)
        id_list = forget_df['ID'].unique().tolist()
    else:
        # If no id_list_file is provided, load all IDs from the main Parquet file
        df = pd.read_parquet(parquet_file)
        id_list = df['ID'].unique().tolist()

    print(f"Loaded {len(id_list)} IDs from {id_list_file if id_list_file else 'parquet_file'}")

    total_image_textual_correct = 0
    total_image_textual_questions = 0
    total_pure_text_correct = 0
    total_pure_text_questions = 0

    # Randomly select few-shot examples based on the model
    if args.model_id.startswith("HuggingFaceM4"):
        selected_ids = random.sample(id_list, 1)
        print(f"Selected few-shot IDs: {selected_ids}")

        few_shot_image_prompts = []  # Stores few-shot prompts for image-textual questions
        few_shot_images = []
        few_shot_text_prompts = []
        few_shot_question_indices = {}  # Dictionary to track few-shot question indices

        # Load few-shot examples based on selected_ids from the few-shot Parquet file
        few_shot_df = pd.read_parquet(few_shot_parquet_file)
        few_shot_samples = few_shot_df[few_shot_df['ID'].isin(selected_ids)]

        for _, row in few_shot_samples.iterrows():
            classification_questions = row["Classification_Task"]
            image_data = row["image"]["bytes"]
            image = Image.open(BytesIO(image_data)).convert("RGB")

            # Initialize storage for this ID
            few_shot_question_indices[row["ID"]] = {
                "image_textual": [],
                "pure_text": []
            }

            # Get questions of each type
            image_textual_questions = classification_questions.get("Image_Textual_Questions", [])
            pure_text_questions = classification_questions.get("Pure_Text_Questions", [])

            # Add one image-textual question if available
            if len(image_textual_questions) > 0:
                idx = 0  # Take the first question
                question_data = image_textual_questions[0]
                few_shot_image_prompts.append({
                    "Question": question_data["Question"],
                    "Options": question_data["Options"],
                    "Correct Answer": question_data["Correct_Answer"]
                })
                few_shot_images.append(image)
                few_shot_question_indices[row["ID"]]["image_textual"].append(idx)

            # Add one pure-text question if available
            if len(pure_text_questions) > 0:
                idx = 0  # Take the first question
                question_data = pure_text_questions[0]
                few_shot_text_prompts.append({
                    "Question": question_data["Question"],
                    "Options": question_data["Options"],
                    "Correct Answer": question_data["Correct_Answer"]
                })
                few_shot_question_indices[row["ID"]]["pure_text"].append(idx)

        print(f"Loaded {len(few_shot_image_prompts)} few-shot image-textual prompts.")
        print(f"Loaded {len(few_shot_text_prompts)} few-shot pure-text prompts.")

        print("Selected few-shot prompts:")
        if few_shot_image_prompts:
            print("Image-Textual:", few_shot_image_prompts[0])
        if few_shot_text_prompts:
            print("Pure-Text:", few_shot_text_prompts[0])

    elif args.model_id.startswith("llava"):
        selected_ids = random.sample(id_list, 1)
        print(f"Selected few-shot IDs: {selected_ids}")

        few_shot_image_prompts = []  # Stores few-shot prompts for image-textual questions
        few_shot_images = []
        few_shot_text_prompts = []
        few_shot_question_indices = {}  # Dictionary to track few-shot question indices

        # Load few-shot examples based on selected_ids from the few-shot Parquet file
        few_shot_df = pd.read_parquet(few_shot_parquet_file)
        few_shot_samples = few_shot_df[few_shot_df['ID'].isin(selected_ids)]
        for _, row in few_shot_samples.iterrows():
            classification_questions = row["Classification_Task"]
            image_data = row["image"]["bytes"]
            image = Image.open(BytesIO(image_data)).convert("RGB")

            # Track the indices of few-shot questions for each ID
            few_shot_question_indices[row["ID"]] = {
                "image_textual": [],
                "pure_text": []
            }

            for idx, question_data in enumerate(classification_questions.get("Image_Textual_Questions", [])):
                few_shot_image_prompts.append({
                    "Question": question_data["Question"],
                    "Options": question_data["Options"],
                    "Correct Answer": question_data["Correct_Answer"]
                })
                few_shot_images.append(image)
                few_shot_question_indices[row["ID"]]["image_textual"].append(idx)

            for idx, question_data in enumerate(classification_questions.get("Pure_Text_Questions", [])):
                few_shot_text_prompts.append({
                    "Question": question_data["Question"],
                    "Options": question_data["Options"],
                    "Correct Answer": question_data["Correct_Answer"]
                })
                few_shot_question_indices[row["ID"]]["pure_text"].append(idx)

        print(f"Loaded {len(few_shot_image_prompts)} few-shot image-textual prompts.")
        print(f"Loaded {len(few_shot_text_prompts)} few-shot pure-text prompts.")

    # Load evaluation samples
    if mode == "test":
        if os.path.isdir(parquet_file):  # Check if it's a directory containing multiple Parquet files
            df = load_and_combine_parquet_files(parquet_file)
        else:
            df = pd.read_parquet(parquet_file)
        eval_samples = df[df['ID'].isin(id_list)]
    else:
        df = pd.read_parquet(parquet_file)
        eval_samples = df[df['ID'].isin(id_list)]

    # Process each evaluation sample
    for _, row in eval_samples.iterrows():
        classification_questions = row["Classification_Task"]

        # Randomly select one image if in test mode
        if mode == "test" and "images" in row:
            image_data = random.choice(row["images"])["bytes"]
        else:
            image_data = row["image"]["bytes"]

        image = Image.open(BytesIO(image_data)).convert("RGB")

        # Iterate through each image-textual question
        for idx, question_data in enumerate(classification_questions.get("Image_Textual_Questions", [])):
            if row["ID"] in few_shot_question_indices and idx in few_shot_question_indices[row["ID"]]["image_textual"]:
                continue  # Skip few-shot question

            question = question_data["Question"]
            options = question_data["Options"]
            correct_answer = question_data["Correct_Answer"]
            question_with_options = formulate_prompt_with_options(question, options)

            # Prepare few-shot prompt if applicable
            few_shot_prompt = ""
            if mode in ["forget", "retain_shared", "test"]:
                for i, few_shot_image in enumerate(few_shot_images):
                    few_shot_question = few_shot_image_prompts[i]["Question"]
                    few_shot_options = few_shot_image_prompts[i]["Options"]
                    few_shot_answer = few_shot_image_prompts[i]["Correct Answer"]
                    few_shot_prompt += (
                        f"USER: <image>\n"
                        f"Question: {few_shot_question}\n"
                        f"A: {few_shot_options['A']}\n"
                        f"B: {few_shot_options['B']}\n"
                        f"C: {few_shot_options['C']}\n"
                        f"D: {few_shot_options['D']}\n"
                        f"Correct Answer: {few_shot_answer}\n"
                    )

            prompt = (f"{few_shot_prompt}"
                      f"USER: <image>\n{question_with_options}\n"
                      f"Just give ONE letter representing the answer directly.\nASSISTANT:")

            # Model specific logic for generating answers
            if args.model_id.startswith("HuggingFaceM4"):
                inputs = processor(images=[*few_shot_images, image], text=prompt, return_tensors="pt").to("cuda")
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)
                generated_text = processor.decode(outputs[0][2:], skip_special_tokens=True)
            elif args.model_id.startswith("llava"):
                inputs = processor(images=[*few_shot_images, image], text=prompt, return_tensors="pt").to("cuda")
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)
                generated_text = processor.decode(outputs[0][2:], skip_special_tokens=True)

            assistant_response = generated_text.split("ASSISTANT:")[1].strip() if "ASSISTANT:" in generated_text else generated_text.strip()
            predicted_answer = assistant_response[0].upper() if assistant_response and assistant_response[0].upper() in options else None
            if predicted_answer == correct_answer:
                total_image_textual_correct += 1
            total_image_textual_questions += 1

        # Process Pure_Text_Questions
        for idx, question_data in enumerate(classification_questions.get("Pure_Text_Questions", [])):
            if row["ID"] in few_shot_question_indices and idx in few_shot_question_indices[row["ID"]]["pure_text"]:
                continue  # Skip few-shot question

            question = question_data["Question"]
            options = question_data["Options"]
            correct_answer = question_data["Correct_Answer"]
            question_with_options = formulate_prompt_with_options(question, options)

            few_shot_prompt = ""
            if mode in ["forget", "retain_shared", "test"]:
                for few_shot in few_shot_text_prompts:
                    few_shot_question = few_shot["Question"]
                    few_shot_options = few_shot["Options"]
                    few_shot_answer = few_shot["Correct Answer"]
                    few_shot_prompt += (
                        f"USER:\n"
                        f"Question: {few_shot_question}\n"
                        f"A: {few_shot_options['A']}\n"
                        f"B: {few_shot_options['B']}\n"
                        f"C: {few_shot_options['C']}\n"
                        f"D: {few_shot_options['D']}\n"
                        f"Correct Answer: {few_shot_answer}\n"
                    )

            prompt = (
                f"{few_shot_prompt}USER:\n{question_with_options}\n"
                f"Just give ONE letter representing the answer directly.\nASSISTANT:"
            )


            # Model specific logic
            if args.model_id.startswith("HuggingFaceM4"):
                inputs = tokenizer(prompt, return_tensors='pt').to("cuda")
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=5, do_sample=False)
                generated_text = tokenizer.decode(outputs[0][2:], skip_special_tokens=True)
            elif args.model_id.startswith("llava"):
                inputs = tokenizer(prompt, return_tensors='pt').to("cuda")
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)
                generated_text = tokenizer.decode(outputs[0][2:], skip_special_tokens=True)

            assistant_response = generated_text.split("ASSISTANT:")[1].strip() if "ASSISTANT:" in generated_text else generated_text.strip()
            predicted_answer = assistant_response[0].upper() if assistant_response and assistant_response[0].upper() in options else None
            if predicted_answer == correct_answer:
                total_pure_text_correct += 1
            total_pure_text_questions += 1

            print("Prompt: ", prompt)
            print("Model Answer: ", predicted_answer)
            print("Correct Answer: ", correct_answer)
            print("The model answer is: ", predicted_answer == correct_answer)
            print("\n")

    # Calculate accuracy
    image_textual_accuracy = (total_image_textual_correct / total_image_textual_questions) * 100 if total_image_textual_questions > 0 else 0
    pure_text_accuracy = (total_pure_text_correct / total_pure_text_questions) * 100 if total_pure_text_questions > 0 else 0

    print(f"Image-Textual Question Accuracy: {image_textual_accuracy:.2f}%")
    print(f"Pure Text Question Accuracy: {pure_text_accuracy:.2f}%")

    return {
        "Image-Textual Question Accuracy": image_textual_accuracy,
        "Pure Text Question Accuracy": pure_text_accuracy
    }


def evaluate_fill_in_the_blank(parquet_file, few_shot_parquet_file, processor, tokenizer, model, args, id_list_file=None, mode="default", forget_parquet_file=None):
    """
    Evaluate classification task with/without few-shot samples based on the mode.

    Args:
        parquet_file: Path to the main Parquet file for evaluation.
        few_shot_parquet_file: Path to the Parquet file containing few-shot examples.
        processor: The processor for handling image and text inputs.
        tokenizer: The tokenizer for decoding model outputs.
        model: The model to use for classification.
        args: Arguments object containing model ID and other configurations.
        id_list_file: (Optional) Path to the JSON file containing the list of IDs. Default is None.
        mode: Mode that controls how few-shot samples are handled ('forget', 'retain_share', or others). Default is 'default'.
        forget_parquet_file: (Optional) Path to the forget Parquet file to filter IDs for test mode.

    Returns:
        dict: A dictionary with accuracy scores.
    """
    print(
        "################################## Fill-in-the-blank Task Starts ##############################################")

    print(f"Evaluating {mode} Mode")
    # Load the ID list from the JSON file if provided
    if id_list_file:
        with open(id_list_file, 'r') as f:
            id_list = json.load(f)
    elif mode == "test" and forget_parquet_file:
        # Load IDs from the forget Parquet file for filtering in test mode
        forget_df = pd.read_parquet(forget_parquet_file)
        id_list = forget_df['ID'].unique().tolist()
    else:
        # If no id_list_file is provided, load all IDs from the Parquet file
        df = pd.read_parquet(parquet_file)
        id_list = df['ID'].unique().tolist()

    print(f"Loaded {len(id_list)} IDs from {id_list_file if id_list_file else 'parquet_file'}")

    total_image_textual_correct = 0
    total_image_textual_questions = 0
    total_pure_text_correct = 0
    total_pure_text_questions = 0

    # Randomly select few-shot examples based on the model
    if args.model_id.startswith("HuggingFaceM4"):
        selected_ids = random.sample(id_list, 1)
        print(f"Selected few-shot IDs: {selected_ids}")

        few_shot_image_prompts = []  # Stores few-shot prompts for image-textual questions
        few_shot_images = []
        few_shot_text_prompts = []
        few_shot_question_indices = {}  # Dictionary to track few-shot question indices

        # Load few-shot examples based on selected_ids from the few-shot Parquet file
        few_shot_df = pd.read_parquet(few_shot_parquet_file)
        few_shot_samples = few_shot_df[few_shot_df['ID'].isin(selected_ids)]

        for _, row in few_shot_samples.iterrows():
            fill_in_the_blank_questions = row["Mask_Task"]
            image_data = row["image"]["bytes"]
            image = Image.open(BytesIO(image_data)).convert("RGB")

            # Initialize storage for this ID
            few_shot_question_indices[row["ID"]] = {
                "image_textual": [],
                "pure_text": []
            }

            # Find first question of each type
            has_image_textual = False
            has_pure_text = False

            for idx, question_data in enumerate(fill_in_the_blank_questions):
                question_type = question_data["Type"]

                if question_type == "Image_Textual" and not has_image_textual:
                    question = question_data["Question"].replace("__",
                                                                 "[Blank]") + "\nPlease **ONLY** provide the correct answer that should replace the [Blank]."
                    few_shot_image_prompts.append({
                        "Question": question,
                        "Correct Answer": question_data["Ground_Truth"]
                    })
                    few_shot_images.append(image)
                    few_shot_question_indices[row["ID"]]["image_textual"].append(idx)
                    has_image_textual = True

                elif question_type == "Pure_Text" and not has_pure_text:
                    question = question_data["Question"].replace("__",
                                                                 "[Blank]") + "\nPlease **ONLY** provide the correct answer that should replace the [Blank]."
                    few_shot_text_prompts.append({
                        "Question": question,
                        "Correct Answer": question_data["Ground_Truth"]
                    })
                    few_shot_question_indices[row["ID"]]["pure_text"].append(idx)
                    has_pure_text = True

                if has_image_textual and has_pure_text:
                    break  # Stop once we have one of each type

        print(f"Loaded {len(few_shot_image_prompts)} few-shot image-textual prompts.")
        print(f"Loaded {len(few_shot_text_prompts)} few-shot pure-text prompts.")

        print("Selected few-shot prompts:")
        if few_shot_image_prompts:
            print("Image-Textual:", few_shot_image_prompts[0])
        if few_shot_text_prompts:
            print("Pure-Text:", few_shot_text_prompts[0])

    elif args.model_id.startswith("llava"):
        selected_ids = random.sample(id_list, 1)
        print(f"Selected few-shot IDs: {selected_ids}")

        few_shot_image_prompts = []  # Stores few-shot prompts for image-textual questions
        few_shot_images = []
        few_shot_text_prompts = []
        few_shot_question_indices = {}  # Dictionary to track few-shot question indices

        # Load few-shot examples based on selected_ids from the few-shot Parquet file
        few_shot_df = pd.read_parquet(few_shot_parquet_file)
        few_shot_samples = few_shot_df[few_shot_df['ID'].isin(selected_ids)]
        for _, row in few_shot_samples.iterrows():
            fill_in_the_blank_questions = row["Mask_Task"]
            image_data = row["image"]["bytes"]
            image = Image.open(BytesIO(image_data)).convert("RGB")

            # Track the indices of few-shot questions for each ID
            few_shot_question_indices[row["ID"]] = {
                "image_textual": [],
                "pure_text": []
            }

            for idx, question_data in enumerate(fill_in_the_blank_questions):
                question = question_data["Question"]
                ground_truth = question_data["Ground_Truth"]
                question_type = question_data["Type"]

                # Add few-shot examples for both question types
                if question_type == "Image_Textual":
                    # Prepare the prompt
                    question = question.replace("__",
                                                "[Blank]") + "\nPlease **ONLY** provide the correct answer that should replace the [Blank]."
                    few_shot_image_prompts.append({
                        "Question": question,
                        "Correct Answer": ground_truth
                    }
                        
                    few_shot_images.append(image)
                    # Record the index of this question as few-shot
                    few_shot_question_indices[row["ID"]]["image_textual"].append(idx)

                elif question_type == "Pure_Text":
                    # Prepare the prompt
                    question = question.replace("__",
                                                "[Blank]") + "\nPlease **ONLY** provide the correct answer that should replace the [Blank]."
                    few_shot_text_prompts.append({
                        "Question": question,
                        "Correct Answer": ground_truth
                    })
                    # Record the index of this question as few-shot
                    few_shot_question_indices[row["ID"]]["pure_text"].append(idx)

        print(f"Loaded {len(few_shot_image_prompts)} few-shot image-textual prompts.")
        print(f"Loaded {len(few_shot_text_prompts)} few-shot pure-text prompts.")
        print("Few-shot image prompts:", few_shot_image_prompts)

    # Load evaluation samples
    # Load the test set with multiple Parquet files if mode is "test"
    if mode == "test":
        if os.path.isdir(parquet_file):  # Check if it's a directory containing multiple Parquet files
            df = load_and_combine_parquet_files(parquet_file)
        else:
            df = pd.read_parquet(parquet_file)
        eval_samples = df[df['ID'].isin(id_list)]
    else:
        df = pd.read_parquet(parquet_file)
        eval_samples = df[df['ID'].isin(id_list)]

    # Process each evaluation sample
    for _, row in eval_samples.iterrows():
        fill_in_the_blank_questions = row["Mask_Task"]

        # Randomly select one image if in test mode
        if mode == "test" and "images" in row:
            image_data = random.choice(row["images"])["bytes"]
        else:
            image_data = row["image"]["bytes"]

        image = Image.open(BytesIO(image_data)).convert("RGB")

        # Iterate through each question in Mask_Task and skip if it's a few-shot question
        for idx, question_entry in enumerate(fill_in_the_blank_questions):
            question = question_entry["Question"]
            ground_truth = question_entry["Ground_Truth"]
            question_type = question_entry["Type"]
            question = question.replace("__", "[Blank]") + "\nPlease **ONLY** provide the correct answer that should replace the [Blank]."

            # Skip if this question was used for few-shot learning
            if row["ID"] in few_shot_question_indices:
                if question_type == "Image_Textual" and idx in few_shot_question_indices[row["ID"]]["image_textual"]:
                    continue  # Skip this image-textual question
                elif question_type == "Pure_Text" and idx in few_shot_question_indices[row["ID"]]["pure_text"]:
                    continue  # Skip this pure-text question

            # Combine few-shot examples with the current prompt (only if mode requires it)
            few_shot_prompt = ""
            if mode in ["forget", "retain_shared", "test", "retain_celebrity"]:
                if question_type == "Image_Textual":
                    for i, few_shot_image in enumerate(few_shot_images):
                        few_shot_prompt += (f"USER:<image>\n{few_shot_image_prompts[i]['Question']}\n"
                                            f"Correct Answer: {few_shot_image_prompts[i]['Correct Answer']}\n")
                elif question_type == "Pure_Text":
                    for i, few_shot_text in enumerate(few_shot_text_prompts):
                        few_shot_prompt += (f"USER:\n{few_shot_text['Question']}\n"
                                            f"Correct Answer: {few_shot_text['Correct Answer']}\n")

            prompt = (f"{few_shot_prompt}USER: "
                      f"<image>\n{question}\nASSISTANT:" if question_type == "Image_Textual" else
                      f"{few_shot_prompt}USER:\n{question}\nASSISTANT:")

            # Model specific logic
            if args.model_id.startswith("HuggingFaceM4"):
                inputs = processor(images=[*few_shot_images, image] if question_type == "Image_Textual" else None,
                                   text=prompt, return_tensors="pt").to("cuda")
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)
                generated_text = processor.decode(outputs[0][2:], skip_special_tokens=True)

            elif args.model_id.startswith("llava"):
                inputs = processor(images=[*few_shot_images, image] if question_type == "Image_Textual" else None,
                                   text=prompt, return_tensors="pt").to("cuda")
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)
                generated_text = processor.decode(outputs[0][2:], skip_special_tokens=True)
                print("Generated Text: ", generated_text)

            # Process the answer
            if "ASSISTANT:" in generated_text:
                assistant_response = generated_text.split("ASSISTANT:")[1].strip()
            elif "Answer:" in generated_text:
                assistant_response = generated_text.split("Answer:")[1].strip()
            else:
                assistant_response = generated_text.strip()

            print("Prompt: ", prompt)
            print("Model Answer: ", assistant_response)
            print("Correct Answer: ", ground_truth)
            print("The model answer is: ", ground_truth.lower() in assistant_response.lower())
            print("\n")
            # Evaluate if the generated answer contains the correct ground truth
            if question_type == "Image_Textual":
                if ground_truth.lower() in assistant_response.lower():
                    total_image_textual_correct += 1
                total_image_textual_questions += 1
            elif question_type == "Pure_Text":
                if ground_truth.lower() in assistant_response.lower():
                    total_pure_text_correct += 1
                total_pure_text_questions += 1

    # Calculate accuracy
    image_textual_accuracy = (total_image_textual_correct / total_image_textual_questions) * 100 if total_image_textual_questions > 0 else 0
    pure_text_accuracy = (total_pure_text_correct / total_pure_text_questions) * 100 if total_pure_text_questions > 0 else 0

    print(f"Image-Textual Question Accuracy: {image_textual_accuracy:.2f}%")
    print(f"Pure Text Question Accuracy: {pure_text_accuracy:.2f}%")

    return {
        "image_textual_accuracy": image_textual_accuracy,
        "pure_text_accuracy": pure_text_accuracy
    }


def evaluate_generation(parquet_file, processor, tokenizer, model, args, mode="default", forget_parquet_file=None):
    """
    Evaluate the generation tasks using the ROUGE and BLEU scores.

    Args:
        parquet_file: Path to the main Parquet file for evaluation.
        processor: The processor for handling text and images (e.g., from Hugging Face).
        tokenizer: The tokenizer for decoding model outputs.
        model: The model for answering the generation questions.
        args: Arguments object containing model ID and other configurations.
        file_name: Name of the file to save the evaluation results.
        mode: Mode to control which evaluation setup to use. Default is 'default'.
        forget_parquet_file: (Optional) Path to the forget Parquet file to filter IDs for test mode.

    Returns:
        dict: A dictionary containing average ROUGE and BLEU scores for Image_Textual and Pure_Text questions.
    """
    print("################################## Generation Task Starts ##############################################")

    # Initialize ROUGE scorer
    rouge_scorer_obj = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    # Initialize variables to store scores and question counts for both question types
    total_rouge1_img = total_rouge2_img = total_rougeL_img = total_bleu_img = total_image_textual_questions = 0
    total_rouge1_text = total_rouge2_text = total_rougeL_text = total_bleu_text = total_pure_text_questions = 0

    # Initialize list to store the results
    results = {
        "Generation_Questions": []
    }

    # Load the ID list from the forget Parquet file for filtering if mode is "test"
    if mode == "test" and forget_parquet_file:
        forget_df = pd.read_parquet(forget_parquet_file)
        id_list = forget_df['ID'].unique().tolist()
    else:
        # Load all IDs from the Parquet file if no filtering is needed
        df = pd.read_parquet(parquet_file)
        id_list = df['ID'].unique().tolist()

    # Load evaluation samples
    if mode == "test":
        if os.path.isdir(parquet_file):  # Check if it's a directory containing multiple Parquet files
            df = load_and_combine_parquet_files(parquet_file)
        else:
            df = pd.read_parquet(parquet_file)
        eval_samples = df[df['ID'].isin(id_list)]
    else:
        df = pd.read_parquet(parquet_file)
        eval_samples = df[df['ID'].isin(id_list)]

    # Loop through each person's data in the evaluation samples
    for _, row in tqdm(eval_samples.iterrows(), total=len(eval_samples)):
        image_id = row["ID"]
        generation_questions = row["Generation_Task"]

        # Randomly select one image if in test mode and multiple images are available
        if mode == "test" and "images" in row:
            image_data = random.choice(row["images"])["bytes"]
        else:
            image_data = row["image"]["bytes"]

        image = Image.open(BytesIO(image_data)).convert("RGB")

        # Process each generation question
        for question_data in generation_questions:
            question_type = question_data["Type"]
            question = question_data["Question"]
            ground_truth = question_data["Ground_Truth"]

            if question_type == "Image_Textual":
                prompt = f"USER: <image>\n{question}\nAnswer the question based on your trained knowledge in one sentence accurately in ENGLISH.\nASSISTANT: "

                if args.model_id.startswith("HuggingFaceM4"):
                    inputs = processor(images=[image], text=prompt, return_tensors="pt").to("cuda")
                elif args.model_id.startswith("llava"):
                    inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda")
                elif args.model_id.startswith("meta-llama"):
                    llama_prompt = f"<|image|><|begin_of_text|>### Question:{question}\n### Answer:"
                    inputs = processor(images=image, text=llama_prompt, return_tensors="pt")
                else:
                    raise ValueError("Model ID not supported")

                outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)
                generated_answer = processor.decode(outputs[0][2:], skip_special_tokens=True)

            else:  # Pure_Text case
                if args.model_id.startswith("meta-llama"):
                    llama_prompt = f"<|begin_of_text|>### Question: {question}\n### Answer:"
                    inputs = processor(text=llama_prompt, return_tensors="pt").to("cuda")
                else:
                    prompt = f"USER: {question}\nAnswer the question based on your trained knowledge in one sentence in ENGLISH.\nASSISTANT:"
                    inputs = tokenizer(prompt, return_tensors='pt').to("cuda")

                outputs = model.generate(**inputs, max_new_tokens=50, do_sample=False)
                generated_answer = tokenizer.decode(outputs[0][2:], skip_special_tokens=True)

            # Process the generated answer
            if "ASSISTANT:" in generated_answer:
                predicted_answer = generated_answer.split("ASSISTANT:")[1].strip()
            elif "Answer:" in generated_answer:
                predicted_answer = generated_answer.split("Answer:")[1].strip()
            else:
                predicted_answer = generated_answer.strip()

            # Print debug information
            print("###### Generation Question: ######", question)
            print("###### Generation Prompt: ######", prompt)
            print("###### Generation ASSISTANT: ######", predicted_answer)
            print("###### Generation Ground Truth: ######", ground_truth)

            # Save results for this question
            results["Generation_Questions"].append({
                "image_id": image_id,
                "question type": question_type,
                "question": question,
                "generated_answer": predicted_answer,
                "ground_truth": ground_truth
            })

            # Calculate ROUGE and BLEU scores
            bleu_score = compute_bleu(ground_truth, predicted_answer)
            rouge_scores = rouge_scorer_obj.score(ground_truth, predicted_answer)

            if question_type == "Image_Textual":
                # Accumulate scores for Image_Textual questions
                total_bleu_img += bleu_score
                total_rouge1_img += rouge_scores['rouge1'].fmeasure
                total_rouge2_img += rouge_scores['rouge2'].fmeasure
                total_rougeL_img += rouge_scores['rougeL'].fmeasure
                total_image_textual_questions += 1
            else:
                # Accumulate scores for Pure_Text questions
                total_bleu_text += bleu_score
                total_rouge1_text += rouge_scores['rouge1'].fmeasure
                total_rouge2_text += rouge_scores['rouge2'].fmeasure
                total_rougeL_text += rouge_scores['rougeL'].fmeasure
                total_pure_text_questions += 1

    # Save the results to a JSON file
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    with open(f'{args.output_folder}/{mode}_generation_results.json', 'w') as f:
        json.dump(results, f, indent=4)

    # Calculate and print average ROUGE and BLEU scores
    avg_scores = {}
    if total_image_textual_questions > 0:
        avg_scores.update({
            "Average ROUGE-1 (Image_Textual)": total_rouge1_img / total_image_textual_questions,
            "Average ROUGE-2 (Image_Textual)": total_rouge2_img / total_image_textual_questions,
            "Average ROUGE-L (Image_Textual)": total_rougeL_img / total_image_textual_questions,
            "Average BLEU (Image_Textual)": total_bleu_img / total_image_textual_questions
        })

    if total_pure_text_questions > 0:
        avg_scores.update({
            "Average ROUGE-1 (Pure_Text)": total_rouge1_text / total_pure_text_questions,
            "Average ROUGE-2 (Pure_Text)": total_rouge2_text / total_pure_text_questions,
            "Average ROUGE-L (Pure_Text)": total_rougeL_text / total_pure_text_questions,
            "Average BLEU (Pure_Text)": total_bleu_text / total_pure_text_questions
        })

    for metric, score in avg_scores.items():
        print(f"{metric}: {score}")

    return avg_scores


def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluate model on retain and forget sets.")

    parser.add_argument('--model_id', type=str, required=True, help='Model ID or path to the model.')
    parser.add_argument('--cache_path', type=str, required=True, help='Path to cache the trained model.')
    parser.add_argument('--data_split_folder', type=str, required=True, help='Path to the image folder.')
    parser.add_argument('--few_shot_data', type=str, required=True, help='Path to the image folder.')
    parser.add_argument('--test_data', type=str, required=True, help='Path to the image folder.')
    parser.add_argument('--celebrity_data', type=str, required=True, help='Path to real person image folder.')
    parser.add_argument('--output_folder', type=str, required=True, help='Path to real person image folder.')
    parser.add_argument('--output_file', type=str, required=True, help='Path to real person image folder.')
    parser.add_argument('--forget_ratio', type=int, default=5, help='Path to real person image folder.')
    parser.add_argument('--pretrain', type=bool, default=False, help="load pretrain model")
    return parser.parse_args()

def main():
    args = parse_arguments()
    # Construct folder paths for "forget" and "retain"
    forget_folder = os.path.join(args.data_split_folder, f"forget_{args.forget_ratio}")
    retain_folder = os.path.join(args.data_split_folder, f"retain_{100 - args.forget_ratio}")
    print("Forget Folder: ", forget_folder)
    print("Retain Folder: ", retain_folder)
    # Define paths to the Parquet files for "forget" and "retain" datasets
    forget_parquet_file = os.path.join(forget_folder, f"train-00000-of-00001.parquet")
    retain_parquet_file = os.path.join(retain_folder, f"train-00000-of-00001.parquet")
    # real_paraquet_file = os.path.join(args.celebrity_data)

    processor = AutoProcessor.from_pretrained(args.model_id)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    torch.cuda.empty_cache()
    if args.pretrain:
        if args.model_id.startswith("llava"):
            print("Loading LLAVA Pretrained model...")
            # Load LLAVA model and processor
            model = LlavaForConditionalGeneration.from_pretrained(
                args.model_id,
                # torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True,
                # quantization_config=bnb_config,
                cache_dir="/afs/crc.nd.edu/group/dmsquare/vol1/zliu29/mllm_unlearn/model/llava-1.5-7b-hf",
            )
        elif args.model_id.startswith("HuggingFaceM4"):
            print("Loading idefics2 Pretrained model...")
            model = Idefics2ForConditionalGeneration.from_pretrained(
                "HuggingFaceM4/idefics2-8b",
                torch_dtype=torch.float16,
                device_map="auto",
                # quantization_config=bnb_config,
                low_cpu_mem_usage=True,
                cache_dir="/afs/crc.nd.edu/group/dmsquare/vol1/zliu29/mllm_unlearn/model/idfics2-8b",
            )
    else:
        if args.model_id.startswith("llava"):
            print("Loading LLAVA Vanilla model...")
            print(f"Loading model from Cache: {args.cache_path}")
            model = LlavaForConditionalGeneration.from_pretrained(
                args.cache_path,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True,
                local_files_only=True
            )
        elif args.model_id.startswith("HuggingFaceM4"):
            print(f"Loading idefics2 model from {args.cache_path}...")
            model = Idefics2ForConditionalGeneration.from_pretrained(
                args.cache_path,
                torch_dtype=torch.float16,
                device_map="auto",
                low_cpu_mem_usage=True,
                local_files_only=True
            )


    # Evaluate Forget Set (from shared classification and generation folders)
    torch.cuda.empty_cache()
    print("### Evaluating Forget Set ###")
    forget_fill_in_the_blank_result = evaluate_fill_in_the_blank(parquet_file=forget_parquet_file,
        few_shot_parquet_file=args.few_shot_data,
        processor=processor,
        tokenizer=tokenizer,
        model=model,
        args=args,
        mode="forget")

    forget_classification_result = evaluate_classification(parquet_file=forget_parquet_file,
        few_shot_parquet_file=args.few_shot_data,
        processor=processor,
        tokenizer=tokenizer,
        model=model,
        args=args,
        mode="forget")


    forget_generation_result = evaluate_generation(parquet_file=forget_parquet_file,
                                                           processor=processor,
                                                           tokenizer=tokenizer,
                                                           model=model,
                                                           args=args,
                                                           mode="forget")

    print("### Evaluating Test Set ###")
    test_classification_result = evaluate_classification(parquet_file=args.test_data,
                                                                 few_shot_parquet_file=args.few_shot_data,
                                                                 processor=processor,
                                                                 tokenizer=tokenizer,
                                                                 model=model,
                                                                 args=args,
                                                                 mode="test",
                                                                 forget_parquet_file=forget_parquet_file)

    test_fill_in_the_blank_result = evaluate_fill_in_the_blank(parquet_file=args.test_data,
                                                                 few_shot_parquet_file=args.few_shot_data,
                                                                 processor=processor,
                                                                 tokenizer=tokenizer,
                                                                 model=model,
                                                                 args=args,
                                                                 mode="test",
                                                                 forget_parquet_file=forget_parquet_file)

    test_generation_result = evaluate_generation(parquet_file=args.test_data,
                                                   processor=processor,
                                                   tokenizer=tokenizer,
                                                   model=model,
                                                   args=args,
                                                   mode="test",
                                                 forget_parquet_file=forget_parquet_file)

    print("### Evaluating Retain Shared Set ###")
    retain_fill_in_the_blank_result = evaluate_fill_in_the_blank(parquet_file=retain_parquet_file,
                                                                 few_shot_parquet_file=args.few_shot_data,
                                                                 processor=processor,
                                                                 tokenizer=tokenizer,
                                                                 model=model,
                                                                 args=args,
                                                                 mode="retain_shared")

    retain_classification_result = evaluate_classification(parquet_file=retain_parquet_file,
                                                           few_shot_parquet_file=args.few_shot_data,
                                                           processor=processor,
                                                           tokenizer=tokenizer,
                                                           model=model,
                                                           args=args,
                                                           mode="retain_shared")

    retain_generation_result = evaluate_generation(parquet_file=retain_parquet_file,
                                                   processor=processor,
                                                   tokenizer=tokenizer,
                                                   model=model,
                                                   args=args,
                                                   mode="retain_shared")

    print("### Evaluating Real Celebrity Set ###")

    real_fill_in_the_blank_result = evaluate_fill_in_the_blank(parquet_file=args.celebrity_data,
                                                                 few_shot_parquet_file=args.few_shot_data,
                                                                 processor=processor,
                                                                 tokenizer=tokenizer,
                                                                 model=model,
                                                                 args=args,
                                                                 mode="retain_celebrity")

    real_classification_result = evaluate_classification(parquet_file=args.celebrity_data,
                                                           few_shot_parquet_file=args.few_shot_data,
                                                           processor=processor,
                                                           tokenizer=tokenizer,
                                                           model=model,
                                                           args=args,
                                                           mode="retain_celebrity")

    real_generation_result = evaluate_generation(parquet_file=args.celebrity_data,
                                                   processor=processor,
                                                   tokenizer=tokenizer,
                                                   model=model,
                                                   args=args,
                                                   mode="retain_celebrity") 

    # Output results
    print("Forget Set Results:")
    print(forget_classification_result)
    print(forget_generation_result)
    print(forget_fill_in_the_blank_result)

    print("Test Set Results:")
    print(test_fill_in_the_blank_result)
    print(test_classification_result)
    print(test_generation_result)

    print("Retain Set (shared dataset) Results:")
    print( retain_fill_in_the_blank_result)
    print(retain_classification_result)
    print(retain_generation_result)

    print("Retain Set (real person) Results:")
    print(real_fill_in_the_blank_result)
    print(real_classification_result)
    print(real_generation_result)

    output_file = f'{args.output_folder}/{args.output_file}_final_evaluation_results.json'
    # Prepare the data to be saved in JSON format
    results_data = {
        "Forget Set Results": {
            "fill_in_the_blank": forget_fill_in_the_blank_result,
            "classification": forget_classification_result,
            "generation": forget_generation_result
        },
        "Test Set Results": {
            "fill_in_the_blank": test_fill_in_the_blank_result,
            "classification": test_classification_result,
            "generation": test_generation_result,
        },
        "Retain Set (shared dataset) Results": {
            "fill_in_the_blank": retain_fill_in_the_blank_result,
            "classification": retain_classification_result,
            "generation": retain_generation_result
        },
        "Retain Set (real person) Results": {
            "fill_in_the_blank": real_fill_in_the_blank_result,
            "classification": real_classification_result,
            "generation": real_generation_result
        }
    }

    # Write the results to a local JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, ensure_ascii=False, indent=4)

    # Optionally print a message to indicate successful save
    print(results_data)
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()


