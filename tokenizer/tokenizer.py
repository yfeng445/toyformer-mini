from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.normalizers import NFKC
from tokenizers.processors import TemplateProcessing


# 1) 构建
tok = Tokenizer(BPE(unk_token="<unk>"))
tok.normalizer = NFKC()
tok.pre_tokenizer = ByteLevel()
tok.decoder = ByteLevelDecoder()

# 2) 训练（关键：special_tokens 顺序 & initial_alphabet）
trainer = BpeTrainer(
    vocab_size=16000,
    special_tokens=["<pad>", "<s>", "</s>", "<unk>"],   # pad 放在首位 ⇒ pad_id=0
    initial_alphabet=ByteLevel.alphabet()
)
tok.train(files=["./corpus/raw/Reminiscences_of_a_Stock_Operator.txt"], trainer=trainer)

# 3) post-processor：自动加 BOS/EOS
tok.post_processor = TemplateProcessing(
    single = "<s> $A </s>",
    pair   = "<s> $A </s> </s> $B </s>",
    special_tokens=[
        ("<s>", tok.token_to_id("<s>")),
        ("</s>", tok.token_to_id("</s>")),
    ],
)

# 4) 启用截断/补齐（与模型 n_ctx 对齐）
N_CTX = 2048
tok.enable_truncation(max_length=N_CTX)
tok.enable_padding(
    pad_id=tok.token_to_id("<pad>"),
    pad_token="<pad>"
)

tok.save("./corpus/processed/tokenizer.json")

tok = Tokenizer.from_file("./corpus/processed/tokenizer.json")

print("vocab_size =", tok.get_vocab_size())
print("pad_id =", tok.token_to_id("<pad>"))
print("bos_id =", tok.token_to_id("<s>"))
print("eos_id =", tok.token_to_id("</s>"))
print("unk_id =", tok.token_to_id("<unk>"))
print("post_processor =", tok.post_processor)

enc = tok.encode("This is a testing of the tokenizer.")
print("ids:", enc.ids)
print("mask:", enc.attention_mask)   # 1=valid, 0=pad
print("tokens:", enc.tokens)
print("decoded:", tok.decode(enc.ids))

