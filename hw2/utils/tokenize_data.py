from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
def tokenize_text(text):
    return tokenizer.tokenize(text)
def encode_text(text):
    return tokenizer.encode(text)
def decode_tokens(token_ids):
    return tokenizer.decode(token_ids)

if __name__ == "__main__":
    sample_text = "你好世界！"
    tokens = tokenize_text(sample_text)
    print("Tokens:", tokens)
    encoded = encode_text(sample_text)
    print("Encoded:", encoded)
    decoded = decode_tokens(encoded)
    print("Decoded:", decoded)
    print(tokenizer.vocab_size) # 21128
    print(tokenizer.pad_token_id) # [PAD]
    print(tokenizer.cls_token_id) # [CLS]
    print(tokenizer.sep_token_id) # [SEP]
    print(tokenizer.unk_token_id) # [UNK]
    print(tokenizer.mask_token_id) # [MASK]