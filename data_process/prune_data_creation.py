import os
import json
import shutil
import pandas as pd
from datasets import Features, Value, Image, Dataset


# Define paths based on the uploaded files
data_split_path = "DATA_SPLIT_PATH"  # Corresponds to folders in image 1
train_data_path = "TRAIN_DATA_PATH"  # Corresponds to image 3
output_path = "PRUNE_DATA_FOLDER"  # Folder where the new structure will be saved
full_set_data_dir = "FULL_SET_DATA_DIR"  # Directory containing the original images

def create_prune_data(data_split_path, train_data_path, output_path):
    os.makedirs(output_path, exist_ok=True)

    # Iterate over the folders in Data_split
    for folder in os.listdir(data_split_path):
        folder_path = os.path.join(data_split_path, folder)
        if os.path.isdir(folder_path):
            # Create corresponding folder in the output directory
            output_folder_path = os.path.join(output_path, folder)
            os.makedirs(output_folder_path, exist_ok=True)

            # Copy the JSON files as they are from Data_split
            for file in os.listdir(folder_path):
                # Ensure we are processing files only
                file_path = os.path.join(folder_path, file)
                if os.path.isfile(file_path) and file.endswith(".json"):
                    # Copy the JSON file
                    shutil.copy(
                        file_path,
                        os.path.join(output_folder_path, file)
                    )
                    # Load the IDs from the JSON file
                    with open(file_path, "r") as json_file:
                        ids = json.load(json_file)  # `ids` will be correctly defined here
                    # print(f"Loaded IDs: {ids}")

                    # Create a subfolder for the corresponding files in Train_data
                    qa_subfolder = os.path.join(output_folder_path, "qa_files")
                    os.makedirs(qa_subfolder, exist_ok=True)

                    # Copy the corresponding QA files based on the IDs
                    for qa_file in ids:
                        qa_file_name = f"{qa_file}_qa.json"
                        qa_file_path = os.path.join(train_data_path, qa_file_name)
                        if os.path.exists(qa_file_path):
                            shutil.copy(qa_file_path, qa_subfolder)


def merge_to_parquet_with_image_bytes(output_path, full_set_data_dir):
    """
    Merge QA JSON files in each folder into a single Parquet file after preprocessing.
    The 'image' column will store image bytes instead of paths for compatibility with Hugging Face workflows.

    Args:
    - output_path (str): Path to the processed data directory containing folders with `qa_files`.
    - full_set_data_dir (str): Path to the directory containing original images.

    Returns:
    - str: Message indicating the result of the operation.
    """
    # Iterate over the folders in the processed data directory
    if os.path.exists(output_path):
        for folder in os.listdir(output_path):
            folder_path = os.path.join(output_path, folder)
            qa_folder_path = os.path.join(folder_path, "qa_files")
            if os.path.isdir(qa_folder_path):
                dataset_entries = []

                # Process each QA file in the folder
                for qa_file in os.listdir(qa_folder_path):
                    if qa_file.endswith("_qa.json"):
                        # Extract profile_id from the filename
                        profile_id = qa_file.split("_")[0]

                        # Check for image file in the original data directory
                        image_bytes = None
                        for extension in ['.jpg', '.png']:
                            profile_image_file = os.path.join(full_set_data_dir, "data_original",
                                                              f"{profile_id}{extension}")
                            if os.path.isfile(profile_image_file):
                                with open(profile_image_file, "rb") as img_file:
                                    image_bytes = {"bytes": img_file.read()}
                                break  # Stop after finding the first available image

                        if not image_bytes:
                            print(f"Warning: No image found for profile ID {profile_id}")

                        # Load metadata from the QA file
                        qa_file_path = os.path.join(qa_folder_path, qa_file)
                        with open(qa_file_path, "r") as f:
                            qa_data = json.load(f)

                        # Convert the metadata field to a JSON string
                        metadata_json = json.dumps(qa_data.get("metadata", []))

                        # Create an entry with image bytes, ID, and metadata
                        entry = {
                            "image": image_bytes,
                            "ID": profile_id,
                            "metadata": metadata_json
                        }

                        # Append the entry to dataset entries
                        dataset_entries.append(entry)

                # Convert the list of entries to a Hugging Face Dataset
                if dataset_entries:
                    # Create a Dataset with the 'image' column as an Image feature
                    features = Features({
                        "image": Image(),  # Image feature type
                        "ID": Value("string"),
                        "metadata": Value("string")
                    })
                    dataset = Dataset.from_dict(
                        {key: [entry[key] for entry in dataset_entries] for key in dataset_entries[0]},
                        features=features)

                    # Save the dataset as a Parquet file
                    parquet_file_name = f"train-00000-of-00001.parquet"
                    parquet_file_path = os.path.join(folder_path, parquet_file_name)
                    dataset.to_parquet(parquet_file_path)

        return "Parquet files with image bytes saved successfully at the same level as the forget/retain_id.json files."
    else:
        return "Output path does not exist. Ensure the required folders and files are in place."


print("Creating prune data...")
create_prune_data(data_split_path, train_data_path, output_path)

print("Merging data to Parquet...")
merge_to_parquet_with_image_bytes(output_path, full_set_data_dir)