# generate.py —— 通用推理脚本（decoder-only / seq2seq） + 约束解码
import argparse, torch
from tokenizers import Tokenizer
from torch.serialization import add_safe_globals
import config as cfgmod

import config
from models import DecoderOnlyLM, TransformerSeq2Seq
from layers import TokenEmbedding, LearnedPositionalEmbedding

@torch.no_grad()
def top_k_sample(logits: torch.Tensor, temperature: float = 1.0, top_k: int = 0) -> int:
    # logits: (1, V)
    if temperature <= 0:
        return int(torch.argmax(logits, dim=-1).item())
    logits = logits / max(1e-8, temperature)
    if top_k and top_k > 0:
        v, ix = torch.topk(logits, k=min(top_k, logits.size(-1)))
        filt = torch.full_like(logits, float("-inf"))
        filt.scatter_(dim=-1, index=ix, src=v)
        logits = filt
    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())

def ban_repeating_ngrams(ids: torch.Tensor, logits: torch.Tensor, n: int):
    if n <= 0 or ids.size(1) < n: return
    prefix = tuple(ids[0, -(n-1):].tolist())
    seq = ids[0].tolist(); banned = set()
    for i in range(len(seq) - n + 1):
        gram = tuple(seq[i:i+n])
        if tuple(gram[:-1]) == prefix:
            banned.add(gram[-1])
    if banned:
        logits[0, list(banned)] = float("-inf")

def apply_repetition_penalty(ids: torch.Tensor, logits: torch.Tensor, penalty: float):
    if penalty == 1.0: return
    used = torch.unique(ids)
    vals = logits[:, used]
    logits[:, used] = torch.where(vals > 0, vals / penalty, vals * penalty)

def build_modules(ckpt_path: str, tokenizer_json: str, device: torch.device):
    add_safe_globals([
        cfgmod.ModelConfig,
        cfgmod.SelfAttnConfig,
        cfgmod.CrossAttnConfig,
        cfgmod.FFNConfig,
        cfgmod.AddNormConfig,
    ])
    ckpt = torch.load(ckpt_path, map_location="cpu")
    # 复原配置（容错：既支持 dict，也支持 pickled dataclass）
    if isinstance(ckpt["cfg"], dict):
        cfg = config.ModelConfig(**{k: v for k, v in ckpt["cfg"].items() if hasattr(config.ModelConfig, k)})
    else:
        cfg = ckpt["cfg"]  # 直接用 pickled 对象
    tok = Tokenizer.from_file(tokenizer_json)
    pad_id = tok.token_to_id("<pad>")

    # 构建模块（与训练一致：embedding 在模型外）
    tok_emb = TokenEmbedding(cfg, pad_id=pad_id).to(device).eval()
    pos_emb = LearnedPositionalEmbedding(cfg).to(device).eval()

    # 判别 checkpoint 是哪类模型（看 state_dict 的键最稳）
    model = None
    if "model" in ckpt:
        try:
            m = DecoderOnlyLM(cfg).to(device).eval()
            m.load_state_dict(ckpt["model"], strict=True)
            model = m; model_kind = "decoder"
        except Exception:
            m = TransformerSeq2Seq(cfg).to(device).eval()
            m.load_state_dict(ckpt["model"], strict=True)
            model = m; model_kind = "seq2seq"
    else:
        raise RuntimeError("checkpoint 缺少 'model' 键")

    # 加载嵌入权重（如无则跳过）
    if "tok_emb" in ckpt: tok_emb.load_state_dict(ckpt["tok_emb"], strict=False)
    if "pos_emb" in ckpt: pos_emb.load_state_dict(ckpt["pos_emb"], strict=False)

    return tok, cfg, tok_emb, pos_emb, model, model_kind, pad_id

@torch.inference_mode()
def generate_decoder_only(prompt: str, tok, cfg, tok_emb, pos_emb, model, pad_id,
                          max_new_tokens: int = 64, temperature: float = 1.0, top_k: int = 0,
                          min_new_tokens: int = 0, no_repeat_ngram_size: int = 0, repetition_penalty: float = 1.0,
                          device: torch.device = torch.device("cpu")) -> str:
    eos_id = tok.token_to_id("</s>")
    enc = tok.encode(prompt)               # TemplateProcessing 已加 <s> ... </s>
    ids = enc.ids
    if ids and ids[-1] == eos_id: ids = ids[:-1]  # 避免输入尾部就是 EOS
    ids = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)  # (1,T)

    for step in range(max_new_tokens):
        attn = torch.ones_like(ids, dtype=torch.long, device=device)       # (1,T) 1=valid
        x = pos_emb(tok_emb(ids))                                          # (1,T,D)
        logits = model(x, self_kpm=attn)                                   # (1,T,V)
        last = logits[:, -1, :]                                            # (1,V)

        # —— 约束解码 —— #
        apply_repetition_penalty(ids, last, repetition_penalty)
        ban_repeating_ngrams(ids, last, n=no_repeat_ngram_size)
        if step < min_new_tokens:
            last[:, eos_id] = float("-inf")

        next_id = top_k_sample(last, temperature, top_k)
        ids = torch.cat([ids, torch.tensor([[next_id]], device=device, dtype=torch.long)], dim=1)
        if next_id == eos_id or ids.size(1) >= cfg.n_ctx: break

    return tok.decode(ids[0].tolist(), skip_special_tokens=True)

@torch.inference_mode()
def generate_seq2seq(src: str, tok, cfg, tok_emb, pos_emb, model, pad_id,
                     max_new_tokens: int = 64, temperature: float = 1.0, top_k: int = 0,
                     min_new_tokens: int = 0, no_repeat_ngram_size: int = 0, repetition_penalty: float = 1.0,
                     device: torch.device = torch.device("cpu")) -> str:
    bos_id = tok.token_to_id("<s>"); eos_id = tok.token_to_id("</s>")

    es = tok.encode(src)
    src_ids  = torch.tensor(es.ids,            dtype=torch.long, device=device).unsqueeze(0)   # (1,S)
    src_mask = torch.tensor(es.attention_mask, dtype=torch.long, device=device).unsqueeze(0)   # (1,S)
    tgt_ids = torch.tensor([[bos_id]], dtype=torch.long, device=device)                        # (1,1)

    for step in range(max_new_tokens):
        tgt_mask = torch.ones_like(tgt_ids, dtype=torch.long, device=device)
        src_emb = pos_emb(tok_emb(src_ids))
        tgt_emb = pos_emb(tok_emb(tgt_ids))
        logits = model(src_emb, tgt_emb, src_kpm=src_mask, tgt_kpm=tgt_mask)                   # (1,T,V)
        last = logits[:, -1, :]

        # —— 约束解码 —— #
        apply_repetition_penalty(tgt_ids, last, repetition_penalty)
        ban_repeating_ngrams(tgt_ids, last, n=no_repeat_ngram_size)
        if step < min_new_tokens:
            last[:, eos_id] = float("-inf")

        next_id = top_k_sample(last, temperature, top_k)
        tgt_ids = torch.cat([tgt_ids, torch.tensor([[next_id]], device=device)], dim=1)
        if next_id == eos_id or tgt_ids.size(1) >= cfg.n_ctx: break

    return tok.decode(tgt_ids[0].tolist(), skip_special_tokens=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="path to .pt checkpoint")
    ap.add_argument("--tokenizer", default="tokenizer/corpus/processed/tokenizer.json")
    ap.add_argument("--mode", choices=["decoder", "seq2seq"], required=True)
    ap.add_argument("--prompt", type=str, help="decoder-only: prompt text")
    ap.add_argument("--src", type=str, help="seq2seq: source text")
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_k", type=int, default=0)
    # —— 解码约束参数 —— #
    ap.add_argument("--min_new_tokens", type=int, default=12)
    ap.add_argument("--no_repeat_ngram_size", type=int, default=3)
    ap.add_argument("--repetition_penalty", type=float, default=1.1)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    tok, cfg, tok_emb, pos_emb, model, model_kind, pad_id = build_modules(args.ckpt, args.tokenizer, device)

    if args.mode == "decoder":
        assert args.prompt is not None, "decoder 模式需要提供 --prompt"
        out = generate_decoder_only(
            args.prompt, tok, cfg, tok_emb, pos_emb, model, pad_id,
            max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_k=args.top_k,
            min_new_tokens=args.min_new_tokens, no_repeat_ngram_size=args.no_repeat_ngram_size,
            repetition_penalty=args.repetition_penalty, device=device
        )
    else:
        assert args.src is not None, "seq2seq 模式需要提供 --src"
        out = generate_seq2seq(
            args.src, tok, cfg, tok_emb, pos_emb, model, pad_id,
            max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_k=args.top_k,
            min_new_tokens=args.min_new_tokens, no_repeat_ngram_size=args.no_repeat_ngram_size,
            repetition_penalty=args.repetition_penalty, device=device
        )
    print(out)

if __name__ == "__main__":
    main()
