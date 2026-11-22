
"""Download and cache pretrained models for software defect analysis.

Downloads the tokenizers and PyTorch model weights required by the
software defect analysis pipeline.

Supported models:
    - CodeBERT: microsoft/codebert-base
    - CodeT5-small: Salesforce/codet5-small

Features:
    - Sequential downloads to reduce network contention.
    - Persistent Hugging Face cache under data/models/cache.
    - Explicit PyTorch checkpoint loading to avoid automatic
      Safetensors conversion.
    - Reuse of previously downloaded files.
    - Individual error handling and download summaries.

Usage:
    uv run python scripts/download_tokenizers.py

Dependencies:
    transformers
    torch
"""

from pathlib import Path

from transformers import AutoModel, AutoTokenizer

# Configuration

"""Root directory for all downloaded model assets."""
MODEL_DIR = Path("data/models")

"""Hugging Face cache directory."""
CACHE_DIR = MODEL_DIR / "cache"

"""Models required by the software defect analysis pipeline."""
TOKENIZER_MODELS = {
    "CodeBERT": {
        "name": "microsoft/codebert-base",
        "type": "roberta",
    },
    "CodeT5-small": {
        "name": "Salesforce/codet5-small",
        "type": "t5",
    },
}


# Download utilities

def download_model(label: str, config: dict) -> None:
    """Download a tokenizer and its pretrained PyTorch model.

    Downloads the tokenizer and model associated with a Hugging Face
    repository. Both are stored in the shared local cache directory.

    The model is explicitly loaded using PyTorch checkpoint weights
    (pytorch_model.bin). Automatic Safetensors conversion is disabled
    to prevent unnecessary background downloads.

    Args:
        label: Human-readable model identifier.
        config: Model configuration containing:
            - name: Hugging Face repository identifier.
            - type: Model architecture identifier.

    Returns:
        None.

    Raises:
        KeyError: If a required configuration key is missing.
        OSError: If model files cannot be downloaded or loaded.
        ValueError: If the repository or model configuration is invalid.
    """
    hf_name = config["name"]
    model_type = config["type"]

    print(f"\n{'-' * 60}")
    print(f"[START] {label} ({hf_name})")
    print(f"{'-' * 60}")

    # Download tokenizer.
    print(f"[1/2] Downloading {label} tokenizer...")

    AutoTokenizer.from_pretrained(
        hf_name,
        cache_dir=CACHE_DIR,
    )

    print(f"[OK] {label} tokenizer ready.")

    # Configure model architecture.
    model_kwargs = {
        "cache_dir": CACHE_DIR,
        "use_safetensors": False,
    }

    if model_type == "roberta":
        model_kwargs["add_pooling_layer"] = False

    # Download and validate PyTorch model weights.
    print(f"[2/2] Downloading {label} model...")

    model = AutoModel.from_pretrained(
        hf_name,
        **model_kwargs,
    )

    # Release model memory after validation.
    del model

    print(f"[OK] {label} model ready.")


def download_all_models() -> None:
    """Download all configured models sequentially.

    Processes each model one at a time to avoid competing for limited
    network bandwidth.

    A failure in one model does not prevent other models from being
    downloaded.

    Returns:
        None.
    """
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    successful = []
    failed = []

    print("=" * 60)
    print("Hugging Face Model Downloader")
    print("=" * 60)
    print(f"Cache directory: {CACHE_DIR}")
    print(f"Configured models: {len(TOKENIZER_MODELS)}")
    print("Download mode: Sequential")
    print("Checkpoint format: PyTorch (.bin)")
    print("Safetensors conversion: Disabled")
    print("=" * 60)

    for label, config in TOKENIZER_MODELS.items():
        try:
            download_model(label, config)
            successful.append(label)

        except Exception as error:
            failed.append((label, str(error)))
            print(f"[FAIL] {label}: {error}")

    # Print final summary.
    print("\n" + "=" * 60)
    print("Download Summary")
    print("=" * 60)

    for label in successful:
        print(f"[OK] {label}")

    for label, error in failed:
        print(f"[FAIL] {label}: {error}")

    print("-" * 60)
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    print("=" * 60)


# Entry point

def main() -> None:
    """Execute the model download process."""
    download_all_models()


if __name__ == "__main__":
    main()
