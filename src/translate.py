import os
import logging
import argparse
import json
from openai import OpenAI
from dotenv import load_dotenv
from data_loader import load_or_download_dataset
import config.translation_config as config
from dataset_registry import DATASET_HANDLERS

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def ensure_directory_exists(directory_path: str) -> None:
    os.makedirs(directory_path, exist_ok=True)

def translate_text(text: str) -> str:
    try:
        response = client.chat.completions.create(
            model=config.TRANSLATION_MODEL,
            messages=[
                {"role": "system", "content": config.SYSTEM_PROMPT},
                {"role": "user", "content": f"Translate this text to Dutch: {text}"}
            ],
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"Translation failed for text: '{text}'. Error: {e}")
        return text  # In case of failure, return original text


def translate_sample(dataset_name: str, sample: dict) -> dict:
    """
    Translate the sample by applying dataset-specific processing and translation logic.
    """
    
    handler = DATASET_HANDLERS[dataset_name]["process_sample"]
    rebuild_handler = DATASET_HANDLERS[dataset_name]["rebuild_sample"]
    
    processed_sample = handler(sample)
    
    translated_sample = {}
    
    for field, value in processed_sample.items():
        if isinstance(value, list):  # If it's a list (like passages)
            translated_sample[field] = [translate_text(item) for item in value]
        else:  # Translate regular fields
            translated_sample[field] = translate_text(value)
    
    return rebuild_handler(translated_sample, sample)


def get_line_count(file_path: str) -> int:
    if not os.path.exists(file_path):
        return 0
    
    with open(file_path, 'r') as f:
        return sum(1 for _ in f)
    
def append_translated_sample(output_file: str, sample: dict) -> None:
    with open(output_file, 'a') as f:
        f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    

def translate_dataset(dataset_name: str, num_samples: int = config.DEFAULT_NUM_SAMPLES) -> None:
    """
    Translate a dataset from Huggingface to Dutch.

    Args:
        dataset_name (str): The name of the dataset (e.g., 'ms_marco').
        num_samples (int): The number of samples to translate (default: 1000).

    Returns:
        None
    """

    output_dir = os.path.join(config.DUTCH_DATA_DIR, dataset_name)
    ensure_directory_exists(output_dir)

    # File to save translated dataset (if it already exists, start from the last processed index)
    output_file = os.path.join(output_dir, f"{dataset_name}_dutch.jsonl")
    dataset = load_or_download_dataset(dataset_name)
    last_processed_index = get_line_count(output_file)

    for i, sample in enumerate(dataset):
        if i < last_processed_index:
            continue

        translated_sample = translate_sample(dataset_name, sample)
        append_translated_sample(output_file, translated_sample)

        if i + 1 - last_processed_index >= num_samples:
            break

    logger.info(f"Translation complete. Translated {i + 1 - last_processed_index} new samples.")


def push_to_huggingface(dataset_name: str):
    print(f"Pushing {dataset_name} to Hugging Face...")
    print("This functionality is not yet implemented.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Translate a dataset from Huggingface.')
    parser.add_argument('--dataset_name', type=str, required=True, help='The name of the dataset (e.g., ms_marco).')
    parser.add_argument('--num_samples', type=int, default=1000, help='The number of samples to translate (default: 1000).')
    parser.add_argument('--push_to_hub', action='store_true', help='Optional flag to push the dataset to Hugging Face after translation.')

    args = parser.parse_args()

    translate_dataset(args.dataset_name, args.num_samples)

    if args.push_to_hub:
        push_to_huggingface(args.dataset_name)
