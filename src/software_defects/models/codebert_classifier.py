"""CodeBERT sequence classifier (RoBERTa encoder + classification head on the ``<s>`` token)."""

from __future__ import annotations

import logging

import torch
from torch import nn
from transformers import RobertaConfig, RobertaModel

from software_defects.models.classifier_config import ClassifierConfig

logging.getLogger("transformers").setLevel(logging.ERROR)

DEFAULT_PRETRAINED_NAME = "microsoft/codebert-base"


class CodeBertClassifier(nn.Module):
    """Binary defect classifier on top of a RoBERTa-style encoder.

    The head follows ``RobertaClassificationHead``: dropout, dense + tanh, dropout, projection, all on the
    hidden state of the first token.
    """

    def __init__(self, config: ClassifierConfig, backbone: RobertaModel | None = None) -> None:
        """Build the classifier.

        Args:
            config: Classifier settings.
            backbone: Ready-made encoder. When None, the pre-trained ``config.pretrained_name`` is loaded
                (from the local cache only, like the tokenizers).
        """
        super().__init__()
        self.config = config
        if backbone is None:
            model_kwargs = {"add_pooling_layer": False, "local_files_only": True}
            if config.cache_dir is not None:
                model_kwargs["cache_dir"] = config.cache_dir
            backbone = RobertaModel.from_pretrained(config.pretrained_name, **model_kwargs, use_safetensors=False)
        self.backbone = backbone
        hidden = backbone.config.hidden_size
        self.dropout = nn.Dropout(config.head_dropout)
        self.dense = nn.Linear(hidden, hidden)
        self.out_proj = nn.Linear(hidden, config.num_labels)
        if config.gradient_checkpointing:
            self.backbone.gradient_checkpointing_enable()

    @classmethod
    def from_backbone_config(cls, config: ClassifierConfig, backbone_config: RobertaConfig) -> CodeBertClassifier:
        """Build a randomly initialised classifier from an encoder config (tests and smoke runs, no download)."""
        return cls(config, backbone=RobertaModel(backbone_config, add_pooling_layer=False))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Return the logits of shape ``(batch, num_labels)``."""
        hidden = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        features = self.dropout(hidden[:, 0, :])
        features = torch.tanh(self.dense(features))
        return self.out_proj(self.dropout(features))
