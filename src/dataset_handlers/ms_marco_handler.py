# dataset_handlers/ms_marco_handler.py
def process_sample_ms_marco(sample: dict) -> dict:
    """
    Extracts and prepares the MS MARCO sample for translation.
    """
    processed_sample = {
        "query": sample.get("query", ""),
        "answers": sample.get("answers", []),
        "passages": sample.get("passages", {}).get("passage_text", [])
    }
    return processed_sample

def rebuild_sample_ms_marco(translated_sample: dict, original_sample: dict) -> dict:
    """
    Rebuilds the MS MARCO sample after translation.
    """
    rebuilt_sample = original_sample.copy()
    rebuilt_sample['query'] = translated_sample['query']
    rebuilt_sample['answers'] = translated_sample['answers']
    rebuilt_sample['passages']['passage_text'] = translated_sample['passages']
    return rebuilt_sample

