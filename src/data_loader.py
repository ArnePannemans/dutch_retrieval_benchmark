import os
import logging
import json
from datasets import load_dataset
from dataset_registry import SUPPORTED_DATASETS
import config.translation_config as config

logger = logging.getLogger(__name__)

def ensure_directory_exists(directory_path: str) -> None:
    """
    Ensures that a directory exists, creating it if necessary.

    Args:
        directory_path (str): The directory path to ensure exists.
    """
    os.makedirs(directory_path, exist_ok=True)

def flatten_dataset(raw_dataset: dict) -> list:
    """
    Flatten the dataset by combining all split samples into a single list.

    Args:
        raw_dataset (dict): The raw dataset with splits.

    Returns:
        list: A list containing all samples from all splits.
    """
    return [sample for split in raw_dataset for sample in raw_dataset[split]]

def save_dataset_as_jsonl(dataset: list, file_path: str) -> None:
    """
    Saves the dataset as a JSONL file.

    Args:
        dataset (list): The dataset samples to save.
        file_path (str): The file path to save the dataset.
    """
    with open(file_path, 'w') as f:
        for sample in dataset:
            f.write(json.dumps(sample) + '\n')
    logger.info(f"Saved full dataset to {file_path}")

def load_or_download_dataset(dataset_name: str) -> list:
    """
    Load the dataset from 'data/english/{dataset_name}' if available, or download it from Hugging Face
    and save it locally in that directory.

    Args:
        dataset_name (str): The name of the dataset (e.g., 'ms_marco').

    Returns:
        list: A list of all dataset samples.
    """
    dataset_info = SUPPORTED_DATASETS.get(dataset_name)

    if not dataset_info:
        raise ValueError(f"Dataset '{dataset_name}' is not supported. Supported datasets: {list(SUPPORTED_DATASETS.keys())}")
    
    huggingface_path = dataset_info["huggingface_path"]
    config_name = dataset_info.get("config_name")

    output_dir = os.path.join(config.ENGLISH_DATA_DIR, dataset_name)
    ensure_directory_exists(output_dir)

    local_file = os.path.join(output_dir, f"{dataset_name}.jsonl")

    # Check if local English dataset exists
    if os.path.exists(local_file):
        logger.info(f"Loading local dataset from {local_file}")
        return load_dataset('json', data_files=local_file)['train']

    # Otherwise, download the dataset from Hugging Face
    logger.info(f"Downloading {dataset_name} from Hugging Face: {huggingface_path}")
    raw_dataset = load_dataset(huggingface_path, config_name) if config_name else load_dataset(huggingface_path)
    
    all_samples = flatten_dataset(raw_dataset)

    save_dataset_as_jsonl(all_samples, local_file)
    
    return all_samples
