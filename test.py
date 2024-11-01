import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import BigramLanguageModel

batch_size = 16
block_size = 500
max_iters = 50000000
eval_interval = 444
learing_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'

eval_iters = 444
n_embd = 64
n_head = 4
n_layer = 4
dropout = 0.0
torch.manual_seed(1337)

file_path = 'data/japanese_train.jsonl'
file_path_val = 'data/japanese_train.jsonl'

def read_data(file_path):
    """
    Read data from jsonl file
    Args:
        file_path: str

    Returns:
        texts: list of strings
        summaries: list of strings
    """
    texts = []
    summaries = []

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            json_data = json.loads(line)
            text = json_data.get("text")
            summary = json_data.get("summary")
            if text and summary:
                texts.append(text)
                summaries.append(summary)
    return texts, summaries

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Tokenizer:

    @staticmethod
    def create_vocab(dataset):
        """
        Create vocabulary from dataset

        Args:
            dataset: list of strings
        """
        vocab = {
                token: index
                for index, token in enumerate(sorted(list(set(dataset))))
        }
        # adding unknown token
        vocab["<unk>"] = len(vocab)

        return vocab

    def __init__(self, vocab):
        self.vocab_encode = {str(k) : int(v) for k,v in vocab.items()}
        self.vocab_decode = {v:k for k,v in self.vocab_encode.items()}

    def encode(self, text):
        """
        Encode text to list of indices

        Args:
            text: string

        Returns:
            list of indices
        """
        return [self.vocab_encode.get(char, self.vocab_encode["<unk>"]) for char in text]

    def decode(self, indices):
        """
        Decode list of indices to text

        Args:
            indices: list of indices
        
        Returns:
            text: string
        """

        return "".join([self.vocab_decode.get(idx, "<unk>") for idx in indices])



# make train data
print("Make train data")
texts,summaries = read_data(file_path)
train_data = ""

for text,summary in zip(texts,summaries):
    train_data = train_data + "<BOS>" + text + "<SUMMARY>" + summary + "<EOS>"

vocab = Tokenizer.create_vocab(train_data)
tokenizer = Tokenizer(vocab)
vocab_size = len(vocab)
print("Vocab size: ",vocab_size)

# make validate data
print("Make validate data")
texts, summaries = read_data(file_path_val)
val_data = ""
for text,summary in zip(texts,summaries):
    val_data = val_data + "<BOS>" + text + "<SUMMARY>" + summary + "<EOS>"

train_data = torch.tensor(tokenizer.encode(train_data), dtype=torch.long)
val_data = torch.tensor(tokenizer.encode(val_data), dtype=torch.long)
print("successfully read data")

block_size = 500
def get_batch(split="train"):
    data = train_data if split == "train" else val_data
    ix = torch.randint(0, data.size(0) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+1+block_size] for i in ix])

    x,y = x.to(device), y.to(device)
    return x,y

model = BigramLanguageModel(vocab_size, n_embd, n_head, n_layer, dropout)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

m = model.to(device)

for iter in range(max_iters):
    x,y = get_batch('train')
    logits, loss = model(x, y)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()

    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"iter: {iter}, loss: {losses['train']:.4f}, val_loss: {losses['val']:.4f}")

         # test generation
        texts, _ = read_data(file_path_val)
        val_input_text = "<BOS>" + texts[0] + "<SUMMARY>" # <SUMMARY>の続きを生成させる
        context = torch.tensor(tokenizer.encode(val_input_text)).unsqueeze(0)
        out_text = tokenizer.decode(m.generate(context, max_new_tokens=2000)[0].tolist())
        print("Context: ", out_text.split("<SUMMARY>")[0])
        print("Generated Summary: ", out_text.split("<SUMMARY>")[1].split("<EOS>")[0])