# models/__init__.py
from .decoder_only_lm import DecoderOnlyLM
from .transformer_seq2seq import TransformerSeq2Seq

# 便于上层从 models 直接拿配置（转发顶层 config）
from config import ModelConfig as TransformerConfig

__all__ = ["DecoderOnlyLM", "TransformerSeq2Seq", "TransformerConfig"]
