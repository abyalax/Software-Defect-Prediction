"""CodeT5-small sequence classifier with the full encoder-decoder (``T5ForSequenceClassification`` style)."""

from __future__ import annotations

import torch
from torch import nn
from transformers import T5Config, T5Model

from software_defects.models.classifier_config import ClassifierConfig

DEFAULT_PRETRAINED_NAME = "Salesforce/codet5-small"


class CodeT5SmallClassifier(nn.Module):
    """Binary defect classifier on top of the whole T5 encoder-decoder.

    Like ``T5ForSequenceClassification``, the decoder reads the right-shifted input ids and the head works on the
    decoder hidden state at the last ``</s>`` token of every sequence, so the decoder has seen the full input.
    """

    def __init__(self, config: ClassifierConfig, backbone: T5Model | None = None) -> None:
        """Build the classifier.

        Args:
            config: Classifier settings.
            backbone: Ready-made T5 encoder-decoder. When None, ``config.pretrained_name`` is loaded from the
                local cache only.
        """
        super().__init__()
        self.config = config
        if backbone is None:
            model_kwargs = {"local_files_only": True}
            if config.cache_dir is not None:
                model_kwargs["cache_dir"] = config.cache_dir
            backbone = T5Model.from_pretrained(config.pretrained_name, **model_kwargs, use_safetensors=False)
        self.backbone = backbone
        hidden = backbone.config.d_model
        self.dropout = nn.Dropout(config.head_dropout)
        self.dense = nn.Linear(hidden, hidden)
        self.out_proj = nn.Linear(hidden, config.num_labels)
        if config.gradient_checkpointing:
            self.backbone.gradient_checkpointing_enable()
            # The T5 stack only checkpoints while training, and needs the cache off for it
            self.backbone.config.use_cache = False

    @classmethod
    def from_backbone_config(cls, config: ClassifierConfig, backbone_config: T5Config) -> CodeT5SmallClassifier:
        """Build a randomly initialised classifier from a T5 config (tests and smoke runs, no download)."""
        return cls(config, backbone=T5Model(backbone_config))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Return the logits of shape ``(batch, num_labels)``."""
        decoder_input_ids = self.backbone._shift_right(input_ids)
        hidden = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            use_cache=False,
        ).last_hidden_state
        features = self.dropout(self._last_eos_state(input_ids, attention_mask, hidden))
        features = torch.tanh(self.dense(features))
        return self.out_proj(self.dropout(features))

    def _last_eos_state(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, hidden: torch.Tensor
    ) -> torch.Tensor:
        """Pick the decoder state at the last ``</s>`` of each row (the last real token when none is present)."""
        positions = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0)
        is_eos = input_ids.eq(self.backbone.config.eos_token_id) & attention_mask.bool()
        last_real = (attention_mask.long() * (positions + 1)).argmax(dim=1)
        last_eos = (is_eos.long() * (positions + 1)).argmax(dim=1)
        index = torch.where(is_eos.any(dim=1), last_eos, last_real)
        return hidden[torch.arange(hidden.size(0), device=hidden.device), index]
