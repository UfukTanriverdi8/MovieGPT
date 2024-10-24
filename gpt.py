import torch
import torch.nn as nn
from torch.nn import functional as F
from art import text2art
from datetime import datetime
import argparse
from tqdm import tqdm
import wandb
from torch.amp import GradScaler, autocast
import os

#os.environ["CUDA_VISIBLE_DEVICES"] = "1"


# Argument parsing
parser = argparse.ArgumentParser(description="Movie Plot GPT")
parser.add_argument('--checkpoint', type=str, default=None, help='Path to your checkpoint file')
parser.add_argument('--training_data', type=str, default='dataset/movie_descriptions.txt', help='Path to the training data file')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
parser.add_argument('--context_length', type=int, default=256, help='Context length for predictions')
parser.add_argument('--eval_iters', type=int, default=100, help='Number of iterations to use for evaluation of the loss')
parser.add_argument('--lr', type=float, default=2.5e-4, help='Learning rate for the optimizer')
parser.add_argument('--n_embd', type=int, default=768, help='Number of dimensions for the embeddings')
parser.add_argument("--n_head", type=int, default=8, help="Number of heads for an attention block")
parser.add_argument("--n_layer", type=int, default=8, help="Number of attention layers to use in the model")
parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
parser.add_argument('--max_new_tokens', type=int, default=500, help='Maximum number of tokens to generate')
parser.add_argument('--seed', type=int, default=7331, help='Seed for reproducibility')
parser.add_argument('--device', type=str, default='cuda', help='Device to use for training')
parser.add_argument('--temperature', type=float, default=1.0, help='Temperature for sampling')
parser.add_argument('--top_k', type=int, default=None, help='Top-k sampling')
parser.add_argument('--logging_steps', type=int, default=50, help='Logging steps for wandb')
parser.add_argument('--epochs', type=int, default=5, help='Number of epochs for training')
parser.add_argument('--disable-typewriter', action='store_false', help='Disable the typewriter mode for generation')
args = parser.parse_args()


# hyperparameters
batch_size = args.batch_size # how many independent sequences will we process in parallel? DROP IT DOWN IF CPU
block_size = args.context_length # context length DROP IT DOWN IF CPU
learning_rate = args.lr # learning rate for the optimizer
device = args.device
eval_iters = args.eval_iters # how many iters to use for evaluation of the loss
n_embd = args.n_embd # how many dimensions to use for the embeddings DROP IT DOWN IF CPU
n_head = args.n_head # how many heads for an attention block DROP IT DOWN IF CPU
n_layer = args.n_layer # how many attention blocks to use in the model DROP IT DOWN IF CPU
dropout = args.dropout # dropout rate
max_new_tokens = args.max_new_tokens # how many new tokens to generate
epochs = args.epochs
# ------------

# for mixed precision training
scaler = GradScaler(device)

# seed for reproducibility
torch.manual_seed(args.seed)

with open(args.training_data, 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    # getting random starting indexes for our blocks
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    # y is just x but shifted 1 to right
    # we want to know the next element of each char in x
    # so we take the shifted version of x to store next element for each char
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    # we need to move our data to the device that we will use for computation
    x, y = x.to(device), y.to(device)
    return x, y

# skipping gradient calculation for evaluation
# even though we don't call the backward function, PyTorch will still
# track the operations happening in the forward pass becuase those tensors
# have requires_grad=True. To prevent this tracking we can use this decorator
@torch.no_grad()
def estimate_loss():
    out = {}
    # we need to set the model to evaluation mode
    # dropout layers behave differently during evaluation
    # batch norm layers also behave differently during evaluation
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in tqdm(range(eval_iters), desc=f"Estimating loss for {split} set"):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        avg_loss = losses.mean()
        out[split] = avg_loss
        if split == 'val':
            perplexity = torch.exp(avg_loss)
            out['val_perplexity'] = perplexity
    # set the model back to training mode
    model.train()
    return out

# one head of the self attention mechanism
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        #self.head_size = head_size
        
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # buffers are not modified by gradient updates
        # but they will be moved to the device that the model is on
        # and also they will be a part of the state dict of the model
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        B, T, C = x.shape
        q = self.query(x) # B,T,head_size
        k = self.key(x) # B,T,head_size
        
        
        attn = q @ k.transpose(-2, -1) * C ** (-0.5) # B,T,T
        attn = attn.masked_fill(self.tril[:T,:T] == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        v = self.value(x) # B,T,head_size
        out = attn @ v # B,T,head_size
        return out 
    
class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        head_out = torch.cat([h(x) for h in self.heads], dim=-1) # (B, T, n_heads * head_size)
        head_out = self.dropout(self.proj(head_out))
        return head_out
    
class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)
    
# communication followed by computation
class Block(nn.Module):
    def __init__(self, n_embd, n_heads):
        super().__init__()
        # n_heads * head_size = n_embd
        self.sa = MultiHeadAttention(n_heads, n_embd//n_heads)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    def forward(self, x):
        # these cumulative operations are called residual connections
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token's value will represent the meaning of that token
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        # Each positional embedding tells the model where the token is in the sequence
        # without these model couldn't know the position of the token in the sequence
        self.positional_embedding = nn.Embedding(block_size, n_embd)
        # we will use the self attention mechanism to learn the relationships between tokens
        # here's our attention blocks
        self.blocks = nn.Sequential(*[Block(n_embd, n_heads=n_head) for _ in range(n_layer)])
        # this is where we get the logits for the next token out of meaning of the current token
        # For more info about logits check the simple_bigram.py
        self.lm_head = nn.Linear(n_embd, vocab_size)
        # final layer normalization
        self.ln_f = nn.LayerNorm(n_embd)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # fetching the learned token embeddings for each token in the sequence
        token_embs = self.token_embedding_table(idx) # (B,T,n_embd)
        # fetching the learned positional embeddings for each position in the sequence
        pos_embs = self.positional_embedding(torch.arange(T, device=idx.device)) # (T,n_embd)
        # adding the token and positional embeddings together
        x = token_embs + pos_embs # (B,T,n_embd)
        # applying the self attention mechanism to the embeddings
        x = self.blocks(x)
        x = self.ln_f(x)
        # getting the logits for the next token out of the embeddings which represent the meaning
        logits = self.lm_head(x) # (B,T,C=vocab_size)

        if targets is None:
            # during generation we will not have targets
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            # the line below will first apply softmax to our logits,
            # turning our logits into a probability distribution with a sum of 1
            # then we will take the the correct next token with the value
            # we have from the targets
            # then we will take the -log of the likelihood of the true next char
            # this will be our loss value
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the block size
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, _ = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            
            # apply temperature scaling
            if temperature <= 0.0:
                print("Temperature should be greater than 0.0. Setting it to 0.1")
                temperature = max(temperature, 0.1)
            elif temperature > 2.0:
                print("Temperature ideally should be between 0.0 and 2.0. More temperature means more randomness in the generated text. A value that is not in the ideal range will still work but it will produce unpredicted results.")
            logits = logits / temperature
            
            # apply top-k sampling
            if top_k is not None:
                values, indices = torch.topk(logits, top_k)
                logits = torch.full_like(logits, float('-inf'))
                logits.scatter_(1, indices, values)
            
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
            if args.disable_typewriter:
                print(decode([idx_next.item()]), end='', flush=True)
        
        return idx


model = GPTLanguageModel()
m = model.to(device)

# print the number of parameters in the model
print(text2art(text="Movie GPT", font="graffiti"))
width = 61
print("="*61)
params = f"{sum(p.numel() for p in m.parameters())/1e6:.2f}" + 'M parameters'
print(params.center(width))
print("="*61)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

if args.checkpoint is not None:
    print("Custom checkpoint file found: " + args.checkpoint)
    model.load_state_dict(torch.load(args.checkpoint, map_location=torch.device(device), weights_only=True))
    print("Model loaded from the checkpoint")
    print(str(max_new_tokens) + " chars will be generated. You can change this with command line arguments.")
    str_input = input("Enter the starting string: ")
    start_str = encode(str_input)
    start_idx = torch.tensor(start_str, dtype=torch.long).unsqueeze(0).to(device)
    output = decode(model.generate(start_idx, max_new_tokens=max_new_tokens, temperature=args.temperature, top_k=args.top_k)[0].tolist())
    print(output)
    with open(f"outputs/output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt", "w") as f:
        f.write(output)

else:
    print("No checkpoints found. Training from scratch")
    print(args.training_data + " will be used for training")
    print(f"Training will be done for {epochs} epochs")
    # Initialize wandb
    wandb.init(project="movie-gpt", config={
        "batch_size": batch_size,
        "context_length": block_size,
        "learning_rate": learning_rate,
        "n_embd": n_embd,
        "n_head": n_head,
        "n_layer": n_layer,
        "dropout": dropout,
        "device": device,
        "max_new_tokens": max_new_tokens,
        "seed": args.seed,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "logging_steps": args.logging_steps,
        "epochs": epochs
    })
    # training loop with epochs
    steps_per_epoch = len(train_data) // (block_size * batch_size)
    eval_interval = steps_per_epoch // 5  # Evaluate approximately 5 times per epoch
    for epoch in range(epochs):
        for iter in tqdm(range(steps_per_epoch), desc=f"Epoch {epoch + 1}/{epochs}"):

            # Every once in a while evaluate the loss on train and val sets
            if iter % eval_interval == 0 and iter > 0:
                losses = estimate_loss()
                print(f"Epoch {epoch}, step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
                wandb.log({"train_loss": losses['train'], "val_loss": losses['val'], "val_perplexity": losses['val_perplexity'], "step": iter, "epoch": epoch})


            # Sample a batch of data
            xb, yb = get_batch('train')

            # Forward pass with autocast for mixed precision
            with autocast(device_type=device):
                logits, loss = model(xb, yb)

            # Zero gradients, backward pass, and update weights using scaler
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
 
            # log per iteration
            if iter % args.logging_steps == 0 and iter > 0:
                wandb.log({"train_loss_step": loss.item(), "step": iter, "epoch": epoch})



        checkpointname = f"checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}_epoch{epoch + 1}.pth"
        torch.save(model.state_dict(), checkpointname)
        print(f"Checkpoint saved: {checkpointname}")

    print("Training finished")

    str_input = input("Enter the starting string: ")
    start_str = encode(str_input)
    start_idx = torch.tensor(start_str, dtype=torch.long).unsqueeze(0).to(device)
    output = decode(model.generate(start_idx, max_new_tokens=max_new_tokens, temperature=args.temperature, top_k=args.top_k)[0].tolist())
    print(output)
    with open(f"outputs/output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt", "w") as f:
        f.write(output)