# train.py
from __future__ import annotations
import math, os, time, random
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import config
from models import TransformerSeq2Seq
from tokenizers import Tokenizer

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TOKENIZER_JSON = os.path.join("tokenizer", "corpus", "processed", "tokenizer.json")
tok = Tokenizer.from_file(TOKENIZER_JSON)
PAD_ID = tok.token_to_id("<pad>") if tok.token_to_id("<pad>") is not None else 0
BOS_ID = tok.token_to_id("<s>")   if tok.token_to_id("<s>")   is not None else 1
EOS_ID = tok.token_to_id("</s>")  if tok.token_to_id("</s>")  is not None else 2
UNK_ID = tok.token_to_id("<unk>") if tok.token_to_id("<unk>") is not None else 3

# ------------------------
# 最简 Token/Pos Embedding（独立于你已有的layers；不会冲突）
# ------------------------
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        nn.init.normal_(self.emb.weight, mean=0.0, std=0.02)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        # (B,T) -> (B,T,D)
        x = self.emb(ids)
        return x * (x.size(-1) ** 0.5)

class PositionalEmbedding(nn.Module):
    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        self.pe = nn.Embedding(max_len, d_model)
        nn.init.normal_(self.pe.weight, mean=0.0, std=0.02)

    def forward(self, B: int, T: int, device: torch.device) -> torch.Tensor:
        pos = torch.arange(T, device=device).unsqueeze(0).expand(B, T)  # (B,T)
        return self.pe(pos)  # (B,T,D)

# ------------------------
# 数据集：相邻行组成样本 (src=line_i, tgt=line_{i+1})
# ------------------------
class LinePairDataset(Dataset):
    def __init__(self, text_path: str, tokenizer: Tokenizer):
        with open(text_path, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f.readlines() if ln.strip()]
        pairs: List[Tuple[List[int], List[int]]] = []
        for i in range(len(lines) - 1):
            src = tokenizer.encode(lines[i]).ids
            tgt = tokenizer.encode(lines[i + 1]).ids
            pairs.append((src, tgt))
        self.data = pairs

    def __len__(self): return len(self.data)

    def __getitem__(self, idx: int):
        src_ids, tgt_ids = self.data[idx]
        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long)

# ------------------------
# 顶层 Collator（可 pickling）
# ------------------------
def _pad_1d(x: torch.Tensor, L: int, pad_id: int) -> torch.Tensor:
    if x.size(0) >= L:
        return x[:L]
    out = x.new_full((L,), pad_id)
    out[: x.size(0)] = x
    return out

class Seq2SeqCollator:
    """
    右侧 padding；长度上限可运行时调整（避免局部闭包）
    返回：
      src_ids:(B,Ls) src_mask:(B,Ls) 1/0
      tgt_ids:(B,Lt) tgt_mask:(B,Lt) 1/0
    """
    def __init__(self, pad_id: int, max_src_len: int, max_tgt_len: int):
        self.pad_id = pad_id
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len

    def set_caps(self, max_src_len: int, max_tgt_len: int):
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len

    def __call__(self, batch):
        src_list, tgt_list = zip(*batch)  # tuple of tensors
        Ls, Lt = self.max_src_len, self.max_tgt_len
        src_ids = torch.stack([_pad_1d(x, Ls, self.pad_id) for x in src_list], dim=0)
        tgt_ids = torch.stack([_pad_1d(x, Lt, self.pad_id) for x in tgt_list], dim=0)
        src_mask = (src_ids != self.pad_id).long()
        tgt_mask = (tgt_ids != self.pad_id).long()
        return src_ids, src_mask, tgt_ids, tgt_mask

# ------------------------
# 训练/验证工具
# ------------------------
def cyc_lr(it: int, warmup: int, base_lr: float) -> float:
    if it < warmup:
        return base_lr * (it + 1) / warmup
    return base_lr

def shift_tgt(tgt_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    teacher-forcing:
      y_in  = [BOS, y[:-1]]
      y_out = y
    这里直接用已有的 tgt_ids：假设 tokenizers 已含 <s> / </s>，我们做标准右移：
    """
    B, T = tgt_ids.size()
    y_in = tgt_ids[:, :-1].contiguous()
    y_out = tgt_ids[:, 1:].contiguous()
    return y_in, y_out

# ------------------------
# 构建模型与“数值对齐”LMHead
# ------------------------
@dataclass
class TrainCfg:
    d_model: int = 256
    n_layer: int = 4
    n_head: int = 4
    n_ctx: int = 512
    dropout: float = 0.0
    vocab_size: int = 11770

def build_modules() -> tuple[config.ModelConfig, TokenEmbedding, PositionalEmbedding, TransformerSeq2Seq]:
    cfg = config.ModelConfig(
        vocab_size=TrainCfg.vocab_size,
        d_model=TrainCfg.d_model,
        n_layer=TrainCfg.n_layer,
        n_ctx=TrainCfg.n_ctx,
        dropout=TrainCfg.dropout,
    )
    # 注意：n_head 在你的 SelfAttnConfig / ModelConfig 下游已使用
    # 若你的实现需显式设置，可在 cfg.self_attn / cfg.casualMHA 中补充

    tok_emb = TokenEmbedding(cfg.vocab_size, cfg.d_model).to(device)
    pos_emb = PositionalEmbedding(cfg.n_ctx, cfg.d_model).to(device)

    model = TransformerSeq2Seq(cfg, return_probs=False).to(device)

    # ---- tying（数值拷贝，兼容 LMHead (D,V) 存储）----
    @torch.no_grad()
    def _locate_head_param(m: TransformerSeq2Seq) -> torch.nn.Parameter:
        V, D = m.vocab_size, m.d_model
        expect = V * D
        cand_dv = None
        for _, p in m.head.named_parameters(recurse=True):
            if p.ndim == 2 and p.numel() == expect:
                if tuple(p.shape) == (V, D):
                    return p
                if tuple(p.shape) == (D, V):
                    cand_dv = p
        if cand_dv is not None:
            return cand_dv
        raise RuntimeError("LMHead 未找到 (V,D) 或 (D,V) 的二维权重参数。")

    with torch.no_grad():
        W_head = _locate_head_param(model)  # 可能是 (D,V)
        W_tok  = tok_emb.emb.weight         # (V,D)
        if tuple(W_head.shape) == (TrainCfg.vocab_size, TrainCfg.d_model):
            W_head.copy_(W_tok)            # (V,D) <- (V,D)
        elif tuple(W_head.shape) == (TrainCfg.d_model, TrainCfg.vocab_size):
            W_head.copy_(W_tok.t())        # (D,V) <- (D,V)
        else:
            raise RuntimeError(f"unexpected LMHead shape: {tuple(W_head.shape)}")

    return cfg, tok_emb, pos_emb, model

# ------------------------
# 主过程
# ------------------------
def main():
    BATCH_SIZE = 8
    GRAD_ACCUM = 4
    BASE_LR = 3e-4
    WARMUP_UPDATES = 200
    NUM_EPOCHS = 20
    MAX_SRC_LEN = TrainCfg.n_ctx
    MAX_TGT_LEN = TrainCfg.n_ctx // 2
    VAL_EVERY_EPOCHS = 2
    EARLY_PATIENCE = 3
    USE_AMP = True
    amp_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    print(f"[env] device={device.type} | amp={USE_AMP} ({amp_dtype})")

    cfg, tok_emb, pos_emb, model = build_modules()
    print(f"[cfg] d_model={cfg.d_model}, n_layer={cfg.n_layer}, vocab={cfg.vocab_size}, n_ctx={cfg.n_ctx}")

    ds_all = LinePairDataset(os.path.join("tokenizer", "corpus", "raw", "Reminiscences_of_a_Stock_Operator.txt"), tok)
    n_total = len(ds_all)
    n_val = max(1, int(0.05 * n_total))
    n_train = n_total - n_val
    ds_train, ds_val = torch.utils.data.random_split(ds_all, [n_train, n_val], generator=torch.Generator().manual_seed(SEED))

    collate = Seq2SeqCollator(PAD_ID, MAX_SRC_LEN, MAX_TGT_LEN)
    NUM_WORKERS = 4
    persistent_workers=True
    
    dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=(device.type=="cuda"), collate_fn=collate)
    dl_val   = DataLoader(ds_val,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=(device.type=="cuda"), collate_fn=collate)

    print(f"[data] train={len(ds_train)}, val={len(ds_val)}, steps/epoch≈{math.ceil(len(ds_train)/BATCH_SIZE)}, updates/epoch≈{math.ceil(len(ds_train)/(BATCH_SIZE*GRAD_ACCUM))}")

    # 优化器与损失
    opt = torch.optim.AdamW(model.parameters(), lr=BASE_LR, betas=(0.9, 0.95), weight_decay=0.1)
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_ID)

    # AMP
    scaler = torch.amp.GradScaler('cuda', enabled=(USE_AMP and device.type=="cuda" and amp_dtype in (torch.float16, torch.bfloat16)))

    # Early stop
    best_val = float("inf")
    epochs_no_improve = 0

    # 训练 Loop
    global_step = 0
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        t0 = time.time()
        run_loss = 0.0
        tokens_seen = 0

        # 如需动态 curriculum，可在每个 epoch 调整上限：
        # collate.set_caps(MAX_SRC_LEN, MAX_TGT_LEN)

        for it, (src_ids, src_mask, tgt_ids, tgt_mask) in enumerate(dl_train, start=1):
            src_ids = src_ids.to(device, non_blocking=True)
            tgt_ids = tgt_ids.to(device, non_blocking=True)
            src_mask = src_mask.to(device, non_blocking=True)
            tgt_mask = tgt_mask.to(device, non_blocking=True)

            # 构造 Embedding
            B, Ls = src_ids.size()
            _, Lt = tgt_ids.size()
            src_emb = tok_emb(src_ids) + pos_emb(B, Ls, device)
            # decoder 输入/输出（shift）
            y_in, y_out = shift_tgt(tgt_ids)
            _, Lt_in = y_in.size()
            tgt_emb = tok_emb(y_in) + pos_emb(B, Lt_in, device)

            # 学习率调度（warmup）
            lr = cyc_lr(global_step, WARMUP_UPDATES, BASE_LR)
            for pg in opt.param_groups: pg["lr"] = lr

            # 前向
            with torch.autocast(device_type="cuda" if device.type=="cuda" else "cpu", dtype=amp_dtype, enabled=USE_AMP):
                logits = model(src_emb, tgt_emb, src_kpm=src_mask, tgt_kpm=tgt_mask[:, :Lt_in])
                # logits: (B, Lt_in, V) ; 目标是 y_out: (B, Lt_in)
                loss = loss_fn(logits.reshape(-1, logits.size(-1)), y_out.reshape(-1))
                loss = loss / GRAD_ACCUM

            # 反向
            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # 累积
            if it % GRAD_ACCUM == 0:
                if scaler.is_enabled():
                    scaler.step(opt)
                    scaler.update()
                else:
                    opt.step()
                opt.zero_grad(set_to_none=True)
                global_step += 1

            run_loss += loss.item() * GRAD_ACCUM
            tokens_seen += (src_mask.sum().item() + tgt_mask.sum().item())

            if (it % (50)) == 0:
                dt = time.time() - t0
                tok_s = tokens_seen / max(dt, 1e-9)
                print(f"epoch {epoch} | update {global_step} | loss {run_loss / it:.4f} | lr {lr:.6f} | tok/s {tok_s:.1f}")

        # 验证
        if (epoch % VAL_EVERY_EPOCHS) == 0:
            model.eval()
            with torch.no_grad():
                val_loss, val_tokens = 0.0, 0
                for src_ids, src_mask, tgt_ids, tgt_mask in dl_val:
                    src_ids = src_ids.to(device, non_blocking=True)
                    tgt_ids = tgt_ids.to(device, non_blocking=True)
                    src_mask = src_mask.to(device, non_blocking=True)
                    tgt_mask = tgt_mask.to(device, non_blocking=True)

                    B, Ls = src_ids.size()
                    _, Lt = tgt_ids.size()
                    src_emb = tok_emb(src_ids) + pos_emb(B, Ls, device)

                    y_in, y_out = shift_tgt(tgt_ids)
                    _, Lt_in = y_in.size()
                    tgt_emb = tok_emb(y_in) + pos_emb(B, Lt_in, device)

                    with torch.autocast(device_type="cuda" if device.type=="cuda" else "cpu", dtype=amp_dtype, enabled=USE_AMP):
                        logits = model(src_emb, tgt_emb, src_kpm=src_mask, tgt_kpm=tgt_mask[:, :Lt_in])
                        loss = loss_fn(logits.reshape(-1, logits.size(-1)), y_out.reshape(-1))

                    val_loss += loss.item()
                    val_tokens += (src_mask.sum().item() + tgt_mask.sum().item())

                val_loss /= max(1, len(dl_val))
                ppl = math.exp(min(20, val_loss))  # clamp for stable print
                print(f"[val] epoch {epoch} | loss={val_loss:.4f} | ppl={ppl:.2f}")

                # early stop
                if val_loss + 1e-6 < best_val:
                    best_val = val_loss
                    epochs_no_improve = 0
                    os.makedirs("checkpoints", exist_ok=True)
                    torch.save({
                        "cfg": cfg,
                        "model": model.state_dict(),
                        "tok_emb": tok_emb.state_dict(),
                        "pos_emb": pos_emb.state_dict(),
                    }, os.path.join("checkpoints", "seq2seq_best.pt"))
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= EARLY_PATIENCE:
                        print(f"[early-stop] no improvement in {EARLY_PATIENCE} evals; best loss={best_val:.4f}")
                        break

    # 保存最终
    os.makedirs("checkpoints", exist_ok=True)
    torch.save({
        "cfg": cfg,
        "model": model.state_dict(),
        "tok_emb": tok_emb.state_dict(),
        "pos_emb": pos_emb.state_dict(),
    }, os.path.join("checkpoints", "seq2seq_final.pt"))
    print("[done] saved ⇒ checkpoints/seq2seq_final.pt")


if __name__ == "__main__":
    main()
