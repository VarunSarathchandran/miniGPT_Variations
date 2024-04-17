import numpy as np
import matplotlib.pyplot as plt
import torch 
import torch.nn as nn
from torch.nn import functional as F
from Tokenizers.tokenizer import BasicTokenizer, RegexTokenizer
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()



#--------
"""
Experiment Parameters

"""

include_tokenizer=True
dataset='shakespeare' 
#dataset='enwik8'
attention="dot prod"
#attention="additive"
block="traditional"
#block="weighted"
RPR=True

#--------
#load checkpoint
load_checkpoint = False
checkpoint_name= 'SavedCheckpoints/checkpoint_ft_5300.pth'
#enter your checkpoint name here
#--------

#hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 10000
eval_interval = 100
learning_rate = 3e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.3
data_fraction= 0.1
#if we are working with enwik, then take only a fraction of the whole data.
# ------------



# load models and vocabularies
basic_tokenizer = BasicTokenizer()
regex_tokenizer = RegexTokenizer()

#basic_tokenizer.load("basic.model")
regex_tokenizer.load("Tokenizers/regex.model")




torch.manual_seed(1337)


#--------
# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
if dataset=='shakespeare':
    filename= 'Datasets/input.txt'
else :
    filename= 'Datasets/enwik8.txt'
with open(filename, 'r', encoding='utf-8') as f:
    text = f.read()



#for tokenizer
if include_tokenizer==True:
    vocab_size = len(regex_tokenizer.vocab)
#create a mapping from characters to integers

else: 
#for character level model
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    # create a mapping from characters to integers
    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }
    encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
    decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string



# atte
# Train and test splits
#if fine tune is true, load shakespare, else load enwik
# fine_tune=True
# if fine_tune==False:
#  data = torch.tensor(regex_tokenizer.encode(text[:int(len(text)*data_fraction)]), dtype=torch.long)
#     #for character level model
#  #data =  torch.tensor(encode(text), dtype=torch.long)
# else:
#  data = torch.tensor(regex_tokenizer.encode(shak_text), dtype=torch.long)
#  #data =  torch.tensor(encode(shak_text), dtype=torch.long)

if dataset=='shakespeare':
    if include_tokenizer==True: 
        data = torch.tensor(regex_tokenizer.encode(text), dtype=torch.long)
    else:
        data = torch.tensor(encode(text), dtype=torch.long)
else :
    if include_tokenizer==True:
        data = torch.tensor(regex_tokenizer.encode(text[:int(len(text)*data_fraction)]), dtype=torch.long)
    else:
        data= torch.tensor(encode(text[:int(len(text)*data_fraction)]), dtype=torch.long)


n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]



# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

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

class RelativePosition(nn.Module):

    def __init__(self, num_units, max_relative_position):
        super().__init__()
        self.num_units = num_units
        self.max_relative_position = max_relative_position
        self.embeddings_table = nn.Parameter(torch.Tensor(max_relative_position * 2 + 1, num_units))
        nn.init.xavier_uniform_(self.embeddings_table)

    def forward(self, length_q, length_k):
        range_vec_q = torch.arange(length_q)
        range_vec_k = torch.arange(length_k)
        distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
        distance_mat_clipped = torch.clamp(distance_mat, -self.max_relative_position, self.max_relative_position)
        final_mat = distance_mat_clipped + self.max_relative_position
        final_mat = torch.LongTensor(final_mat).to(device)
        embeddings = self.embeddings_table[final_mat].to(device)

        return embeddings


class Head_Relative_Positions(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size,additive):
        #additive is accepted just to keep the same interface as the other head. It is not used since it is incompatible with relative position representations
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.max_relative_position = 3
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.q_layer= nn.Linear(head_size,block_size,bias=False)
        self.k_layer= nn.Linear(head_size,block_size,bias=False)
        self.tanh=nn.Tanh()
        self.register_buffer('tril', torch.tril(torch.ones(batch_size,n_head,block_size,block_size)))
        self.register_buffer('att_type',torch.Tensor([additive]))
        self.dropout = nn.Dropout(dropout)
        self.head_size=head_size

        self.scale = torch.sqrt(torch.FloatTensor([head_size])).to(device)
        self.register_buffer('att_type',torch.Tensor([additive]))
        self.relative_position_k = RelativePosition(head_size, self.max_relative_position)
        self.relative_position_v = RelativePosition(head_size, self.max_relative_position)
    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C= x.shape
        k= self.key(x)
        q= self.query(x)
        v= self.value(x)

        r_q1= q.view(B,-1,1,self.head_size).permute(0,2,1,3)
        #B,1,T,head_size
        r_k1 = k.view(B,-1,1,self.head_size).permute(0,2,1,3)
        attn1= torch.matmul(r_q1, r_k1.permute(0,1,3,2))
        #B,1,T,T

        r_q2= q.permute(1,0,2).view(T,B,self.head_size)
        #T,B,head_size
        r_k2= self.relative_position_k(T,T)
        attn2= torch.matmul(r_q2, r_k2.transpose(1, 2)).transpose(0, 1)
        #B,T,T
        attn2= attn2.view(B,1,T,T)
        #B,1,T,T
        attn= (attn1 + attn2)/self.scale
        attn= self.dropout(torch.softmax(attn.masked_fill(self.tril[:B,:1,:T,:T]==0,float('-inf')),dim=-1))
        #B,1,T,T
        r_v1= v.view(B,-1,1,self.head_size).permute(0,2,1,3)
        w1= torch.matmul(attn, r_v1)
        #B,1,T,head_size
        r_v2= self.relative_position_v(T,T)
        w2= attn.permute(2, 0, 1, 3).view(T, B,T)
        w2= torch.matmul(w2, r_v2)
        w2=w2.transpose(0, 1).view(B,1,T,self.head_size)
        #B,1,T,head_size
        out= w1+w2
        out= out.permute(0,2,1,3)
        #B,T,1,head_size
        out= out.view(B,T,self.head_size)
        return out


class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size,additive):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.q_layer= nn.Linear(head_size,block_size,bias=False)
        self.k_layer= nn.Linear(head_size,block_size,bias=False)
        self.tanh=nn.Tanh()
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.register_buffer('att_type',torch.Tensor([additive]))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        k = self.key(x)   # (B,T,hs)
        q = self.query(x) # (B,T,hs)
        # compute attention scores ("affinities")
        #computes additive attention is slef.additive is True, else uses scaled dot product attention
        if self.att_type==1:
            q_w= self.q_layer(q)
            k_w= self.k_layer(k)
            wei = self.tanh(q_w+k_w)
            #if training does not work, consider scaling.
        else:
            wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,hs)
        out = wei @ v # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size,additive):
        super().__init__()
        if(RPR==True):
         self.heads = nn.ModuleList([Head_Relative_Positions(head_size,additive) for _ in range(num_heads)])
        else:
            self.heads = nn.ModuleList([Head(head_size,additive) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

#remember to add additive attention later
class BranchedMultiAttentionBlock(nn.Module):
    """ Implementation of the branched attention in Weighted Transformer paper"""
    def __init__(self, n_embd,num_heads,head_size,additive):
        super().__init__()
        if RPR==True:
            self.heads = nn.ModuleList([Head_Relative_Positions(head_size,additive) for _ in range(num_heads)])
        else: 
            self.heads = nn.ModuleList([Head(head_size,additive=additive) for _ in range(num_heads)]) #list of heads
        self.linears = nn.ModuleList([nn.Linear(n_embd, n_embd) for _ in range(num_heads)]) #list of linear layers
        self.FFNs = nn.ModuleList([FeedFoward(n_embd) for _ in range(num_heads)]) #list of feed forward layers
        self.w_ks = nn.ParameterList([nn.Parameter(torch.randn(1)) for _ in range(num_heads)])
        self.w_as = nn.ParameterList([nn.Parameter(torch.randn(1)) for _ in range(num_heads)])
        self.ln1= nn.LayerNorm(n_embd)
        self.num_heads = num_heads
        
    def forward(self, x):
        x= self.ln1(x) #layer norm
        out= [h(x) for h in self.heads]
        out = [self.linears[i](out[i]) for i in range(self.num_heads)]
        w_ks_norm= F.softmax(torch.Tensor(self.w_ks),dim=0)
        #not sure this will train correctly
        out = [w_ks_norm[i]*out[i] for i in range(self.num_heads)]
        out = [self.FFNs[i](out[i]) for i in range(self.num_heads)]
        w_as_norm= F.softmax(torch.Tensor(self.w_as),dim=0)
        out = [w_as_norm[i]*out[i] for i in range(self.num_heads)]
        out = sum(out)
        out = out +x #residual connection
        return out
class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head,additive):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head 
        self.sa = MultiHeadAttention(n_head, head_size,additive)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        if attention=="dot prod":
            additive=False
        if attention=="additive":
            additive=True

        if RPR==True:
            if additive==True:
                additive=False
                print("Additive attention is not supported with relative position representations, forcing dot-product attention")
        if block=="traditional":
            self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head,additive=additive) for _ in range(n_layer)])
        if block=="weighted":
            self.blocks = nn.Sequential(*[BranchedMultiAttentionBlock(n_embd,n_head,n_embd,additive=additive) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # better init, not covered in the original GPT video, but important, will cover in followup video
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            #print(idx_cond.shape)
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = GPTLanguageModel()

m = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')
print(f"attention: {attention}, block: {block}, RPR: {RPR}")

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
loss_history=[]
start_epoch=0
if load_checkpoint:
    if device == 'cpu':
        checkpoint = torch.load(checkpoint_name, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(checkpoint_name)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    loss_history.append(checkpoint['loss'])
    start_epoch = checkpoint['epoch']

for iter in range(start_epoch,start_epoch+max_iters):

    # every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss()
            loss_history.append(losses)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            #saving the model checkpoint
            torch.save({'epoch': iter, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'loss': losses}, f'SavedCheckpoints/checkpoint_{iter}.pth')

    # sample a batch of data
        xb, yb = get_batch('train')

    # evaluate the loss
        with autocast():
            logits, loss = model(xb, yb)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
# generate from the model

if block=="traditional":
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
if block=="weighted":
    context = torch.zeros((1, block_size), dtype=torch.long, device=device)
if include_tokenizer==True:
    output = regex_tokenizer.decode(np.array(m.generate(context, max_new_tokens=1000)[0].tolist()))
else:
    output = decode(np.array(m.generate(context, max_new_tokens=1000)[0].tolist()))

print(output)
with open('GeneratedData/output.txt', 'w') as file:
    print(output, file=file)
#open('more.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))

train_losses=[i['train'] for i in loss_history]
test_loseses=[i['val'] for i in loss_history]
plt.plot(train_losses, label='train')
plt.plot(test_loseses, label='val')
plt.xlabel('Iteration(Steps of 500)')
plt.ylabel('Loss')
plt.title('Train and Validation Loss')
plt.legend()
