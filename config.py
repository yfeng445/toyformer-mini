# config.py
from dataclasses import dataclass, field

@dataclass
class BaseAttnConfig:
    n_head: int = 4
    dropout: float = 0.0          # 注意力权重上的dropout（若启用）
    qkv_bias: bool = True
    out_bias: bool = True
    # 可逐步扩展：use_flash, n_kv_head, rope, window_size, ...

# 自注意力（可因果；Encoder/Decoder 共用此类，区别仅在 causal）
@dataclass
class SelfAttnConfig(BaseAttnConfig):
    causal: bool = False
    kv_dim: int | None = None

# 交叉注意力（无因果，只做 padding 掩码）
@dataclass
class CrossAttnConfig(BaseAttnConfig):
    kv_dim: int | None = None     # memory 的 KV 维（一般与 d_model 相等）

# FFN 与 AddNorm
@dataclass
class FFNConfig:
    d_ff: int | None = None
    activation: str = "gelu"
    gelu_approx: bool = False
    dropout: float = 0.0

@dataclass
class AddNormConfig:
    eps: float = 1e-5

@dataclass
class ModelConfig:
    vocab_size: int = 11770
    d_model: int = 256
    n_layer: int = 4
    n_ctx: int = 512
    dropout: float = 0.0

    self_attn: SelfAttnConfig = field(default_factory=lambda: SelfAttnConfig(causal=True))
    cross_attn: CrossAttnConfig = field(default_factory=CrossAttnConfig)
    ffn: FFNConfig = field(default_factory=FFNConfig)
    addnorm: AddNormConfig = field(default_factory=AddNormConfig)

    def __post_init__(self):
        # 默认 d_ff = 4 * d_model
        if self.ffn.d_ff is None:
            self.ffn.d_ff = 4 * self.d_model
