from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
def tokenize_text(text):
    return tokenizer.tokenize(text)
def encode_text(text):
    return tokenizer.encode(text, return_tensors='pt')
def decode_tokens(token_ids):
    return tokenizer.decode(token_ids)

if __name__ == "__main__":
    sample_text = "Hello, how are you? I'm fine, thank you!"
    tokens = tokenize_text(sample_text)
    print("Tokens:", tokens)
    encoded = encode_text(sample_text)
    print("Encoded:", encoded)
    decoded = decode_tokens(encoded[0])
    print("Decoded:", decoded)