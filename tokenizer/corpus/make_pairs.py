# tokenizer/corpus/make_pairs.py
import os, random, re

RAW = "./raw/Reminiscences_of_a_Stock_Operator.txt"
OUT_DIR = "./processed"
MIN_CHARS = 8     # 过滤过短行
MAX_SAMPLES = None  # 设成整数可下采样

os.makedirs(OUT_DIR, exist_ok=True)

with open(RAW, "r", encoding="utf-8") as f:
    lines = [ln.strip() for ln in f if ln.strip()]

# 简单清洗 & 过滤
def ok(s: str) -> bool:
    s = re.sub(r"\s+", " ", s)
    return len(s) >= MIN_CHARS

lines = [re.sub(r"\s+", " ", s) for s in lines if ok(s)]

# 构造 (src=第i行, tgt=第i+1行)
pairs = [(lines[i], lines[i+1]) for i in range(len(lines) - 1)]
random.shuffle(pairs)
if MAX_SAMPLES is not None:
    pairs = pairs[:MAX_SAMPLES]

src_path = os.path.join(OUT_DIR, "train.src")
tgt_path = os.path.join(OUT_DIR, "train.tgt")

with open(src_path, "w", encoding="utf-8") as fs, open(tgt_path, "w", encoding="utf-8") as ft:
    for s, t in pairs:
        fs.write(s + "\n")
        ft.write(t + "\n")

print(f"wrote {len(pairs)} pairs to:\n  {src_path}\n  {tgt_path}")
