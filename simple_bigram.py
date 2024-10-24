import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2 #0.01
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
# ------------

# seed for reproducibility
torch.manual_seed(7331)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('dataset/letterboxd/movie_descriptions_small.txt', 'r', encoding='utf-8') as f:
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
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    # set the model back to training mode
    model.train()
    return out

# super simple bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token's value in this embedding table
        # gives us the unnormalized logits for the next possible tokens
        # so for example the value 'A' in this table represents the
        # logits for the next chars that can come after the value 'A'
        # we have 88 chars so the possible chars are 88 also
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        # idx and targets are both (B,T) tensor of integers
        # B = batch dimension, index of the current batch
        # T = time dimension, index of the char in the current block/context
        # C = channels dimension, 88 logit values for the next possible char
        logits = self.token_embedding_table(idx) # (B,T,C)

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

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to last dimension to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # tokens are chosen based on the probability distribution
            # given by their likelihoods. this is called sampling. For example:
            # A=0.7,B=0.2,C=0.1
            # A would have 70% chance of being chosen
            # B would have 20%
            # C would have 10%
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = BigramLanguageModel()
m = model.to(device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    # pytorch accumulates the gradients by default
    # to prevent that we need to set the gradients to zero
    # additionally we're setting them to none for optimization
    optimizer.zero_grad(set_to_none=True)

    # calculating the gradients from the loss tensor
    loss.backward()

    # after calculating the gradients we take one step in the direction of them
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))