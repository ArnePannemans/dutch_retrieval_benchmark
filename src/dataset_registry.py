# src/dataset_registry.py
from dataset_handlers.ms_marco_handler import process_sample_ms_marco, rebuild_sample_ms_marco


SUPPORTED_DATASETS = {
    "ms_marco": {
        "huggingface_path": "microsoft/ms_marco",
        "config_name": "v1.1",
        "description": "MS MARCO dataset for passage retrieval."
    },
    "quora": {
        "huggingface_path": "beIR/quora",
        "config_name": None,
        "description": "Quora Question Pairs dataset for question similarity."
    },
}

DATASET_HANDLERS = {
    "ms_marco": {
        "process_sample": process_sample_ms_marco,
        "rebuild_sample": rebuild_sample_ms_marco,
    },
}
