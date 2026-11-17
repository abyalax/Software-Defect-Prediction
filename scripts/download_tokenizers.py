"""Script to download Hugging Face tokenizers before running analysis."""

from transformers import AutoTokenizer

TOKENIZER_MODELS = {
    "CodeBERT": "microsoft/codebert-base",
    "CodeT5-small": "Salesforce/codet5-small",
}


def download_tokenizers() -> None:
    """Download all required tokenizers."""
    print("Downloading tokenizers...")
    for model_label, hf_name in TOKENIZER_MODELS.items():
        print(f"  Downloading {model_label} ({hf_name})...")
        try:
            AutoTokenizer.from_pretrained(hf_name)
            print(f"  [OK] {model_label} downloaded successfully")
        except Exception as e:
            print(f"  [FAIL] Failed to download {model_label}: {e}")
    print("\nDownload complete.")


if __name__ == "__main__":
    download_tokenizers()
