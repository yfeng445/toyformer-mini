# models/transformer_seq2seq.py
import torch
import torch.nn as nn
import config
from blocks import EncoderBlock, DecoderBlock
from layers import LMHead, SoftmaxLastDim


class TransformerSeq2Seq(nn.Module):
    """
    原始 Transformer：Encoder(N) + Decoder(N) → LMHead → (可选) Softmax
    约定：输入均为 embedding（外部完成 token/pos 两类嵌入）
    tie_output_to(): 这里采用“拷贝数值”而非真正 Parameter 共享，以规避重复注册冲突。
    """
    def __init__(self, cfg: config.ModelConfig, return_probs: bool = False):
        super().__init__()
        self.cfg = cfg
        self.encoder = nn.ModuleList([EncoderBlock(cfg) for _ in range(cfg.n_layer)])
        self.decoder = nn.ModuleList([DecoderBlock(cfg) for _ in range(cfg.n_layer)])

        # LM 头：d_model -> vocab_size
        self.head = LMHead(cfg, bias=False)

        self.prob = SoftmaxLastDim()
        self.d_model = cfg.d_model
        self.vocab_size = cfg.vocab_size
        self.return_probs = return_probs
        self._tied = False  # 此处仅拷贝权重数值，非真正 tying

    # -------- 查找 / 访问 LMHead 线性权重 --------
    def _locate_head_proj_param(self) -> tuple[torch.nn.Parameter, str]:
        """
        在 LMHead 中定位词表投影权重参数，并返回 (param, layout)：
        - layout = "VD"  表示形状 (vocab_size, d_model)
        - layout = "DV"  表示形状 (d_model, vocab_size) —— 你的当前 LMHead 是这种
        """
        V, D = self.vocab_size, self.d_model
        expect_elems = V * D

        # 1) 优先按常见命名快速路径
        # 你的实现：self.head.W  (D,V)
        if hasattr(self.head, "W") and isinstance(self.head.W, torch.nn.Parameter):
            w = self.head.W
            if w.ndim == 2 and w.numel() == expect_elems:
                if tuple(w.shape) == (V, D): return w, "VD"
                if tuple(w.shape) == (D, V): return w, "DV"

        # 兼容其他常见写法
        if hasattr(self.head, "weight") and isinstance(self.head.weight, torch.nn.Parameter):
            w = self.head.weight
            if w.ndim == 2 and w.numel() == expect_elems:
                if tuple(w.shape) == (V, D): return w, "VD"
                if tuple(w.shape) == (D, V): return w, "DV"
        if hasattr(self.head, "proj") and hasattr(self.head.proj, "weight"):
            w = self.head.proj.weight
            if w.ndim == 2 and w.numel() == expect_elems:
                if tuple(w.shape) == (V, D): return w, "VD"
                if tuple(w.shape) == (D, V): return w, "DV"

        # 2) 形状与元素数回退扫描
        for _, p in self.head.named_parameters(recurse=True):
            if p.ndim == 2 and p.numel() == expect_elems:
                if tuple(p.shape) == (V, D): return p, "VD"
                if tuple(p.shape) == (D, V): return p, "DV"

        raise RuntimeError("未在 LMHead 中找到符合 (V,D) 或 (D,V) 的投影权重。")

    def _get_head_weight(self) -> torch.nn.Parameter:
        w, _ = self._locate_head_proj_param()
        return w

    def _set_head_weight(self, shared_weight: torch.nn.Parameter):
        """
        仅做数值拷贝，不替换 Parameter 对象：
        - 如果 head 权重是 (V,D) 直接 copy_
        - 如果 head 权重是 (D,V) 则 copy_ shared_weight.t()
        """
        V, D = self.vocab_size, self.d_model
        if tuple(shared_weight.shape) != (V, D):
            raise ValueError(f"shape mismatch for tying: expect {(V, D)}, got {tuple(shared_weight.shape)}")

        with torch.no_grad():
            w, layout = self._locate_head_proj_param()
            if layout == "VD":
                w.copy_(shared_weight)
            elif layout == "DV":
                w.copy_(shared_weight.t())
            else:
                raise RuntimeError(f"unexpected layout: {layout}")



    @torch.no_grad()
    def tie_output_to(self, shared_weight: torch.nn.Parameter):
        """
        将输出层初始化为与 token embedding 相同的数值（非真正 Parameter-level 共享）。
        """
        expect = (self.vocab_size, self.d_model)
        if tuple(shared_weight.shape) != expect:
            raise ValueError(f"shape mismatch for tying: expect {expect}, got {tuple(shared_weight.shape)}")
        self._set_head_weight(shared_weight)
        self._tied = False  # 这里只做数值拷贝，不共享 Parameter 对象

    @property
    def is_tied(self) -> bool:
        return self._tied

    # -------- encode / decode / forward --------
    def encode(self, src_emb: torch.Tensor, *, src_kpm: torch.Tensor | None = None) -> torch.Tensor:
        h = src_emb
        for blk in self.encoder:
            h = blk(h, enc_kpm=src_kpm)
        return h

    def decode(
        self,
        tgt_emb: torch.Tensor,
        memory: torch.Tensor,
        *,
        tgt_kpm: torch.Tensor | None = None,
        src_kpm: torch.Tensor | None = None,
    ) -> torch.Tensor:
        y = tgt_emb
        for blk in self.decoder:
            y = blk(y, self_kpm=tgt_kpm, memory=memory, memory_kpm=src_kpm)
        return y

    def forward(
        self,
        src_emb: torch.Tensor,
        tgt_emb: torch.Tensor,
        *,
        src_kpm: torch.Tensor | None = None,
        tgt_kpm: torch.Tensor | None = None,
    ) -> torch.Tensor:
        mem = self.encode(src_emb, src_kpm=src_kpm)
        h = self.decode(tgt_emb, memory=mem, tgt_kpm=tgt_kpm, src_kpm=src_kpm)
        logits = self.head(h)
        return self.prob(logits) if self.return_probs else logits
