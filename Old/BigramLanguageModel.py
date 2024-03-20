import torch
import torch.nn as nn
from torch.nn import functional as F
import Old.tokenizer as t
import random
#for reproducability
torch.manual_seed(0)

eval_iters = 300

#everything that happens in this function does not call backward()
#more efficient
@torch.no_grad()
def estimate_loss():
    out = {}
    m.eval()
    for split in ['training', 'validating']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = t.get_batch(split)
            _,loss = m(X,Y)
            losses[k] - loss.time()
        out[split] = losses.mean()
    m.train()
    return out

class BigramLanguageModel(nn.Module):
    #creates embedding table of vocab_size by vocab_size
    #has random vectors for each character in vocab_size
        #not trained
    def __init__(self, vocab_size):
        #initializes the Module from pytorch
        super().__init__()
        #creates a sort of look up table
        #61 x 61
        #Each row is associated with a character
            #ex: row 0 is associated with \n
            #you can look up these characters
        #each row has a vector of length vocab_size with numbers in it
            #numbers are random right now
            #since it is not trained
        #first param: how many vectors to make
        #second param: how long the vectors should be
        self.table = nn.Embedding(vocab_size, vocab_size)

    #takes in a training data and the targets
    #returns the logits and loss of comparing the logits to the targets
    def forward(self, idx, targets=None):
        #takes in 4 (batch size, B) tensors with 8 (block size, T) chars in them
            #chars come as ints
        #idx is (B, T) so (4, 8)
        #will return a tensor of size B (4)
        #each tensor will contain another T (8) tensors that include the embeddings of each char in T
        #so the logits is just a tensor that includes all embeddings for the 4 batches of training data
        logits = self.table(idx)

        #targets do not have to be specified
        #if there are none, loss is 0
        if targets is not None:
            B, T, C = logits.shape
            #must reshape the tensor since loss takes in logits of (C) or (N,C)
            #streches out the tensor
            #now is a tensor of length B*T (32) which represents the total amount of training characters we send in
            #The elements of this tensor are the embedding associated with the character
            logits = logits.view(B*T, C)
            #targets must match the shape of logits
            targets = targets.view(B*T)
            #loss measures the error in the training data compared to the validating data
            #does this by doing a shit ton of math 
            #returns a tensor that has the loss
            loss = F.cross_entropy(logits, targets)
        else:
            loss = 0

        return logits, loss
    
    #generates a string of length max_new_token
    #takes in a tensor of the starting char idx
    def generate(self, idx, max_new_token):
        #creates max_new_token tokens
        #remember _ means we don't care about that value
            #from haskell!!!!
        for _ in range(max_new_token):
            #the size of the logits will be 1x_x61
            #where _ is the number of token that the for loop is on
            #the B is only 1 since we are only looking at batch
            logits, loss = self(idx)

            #will only focus on the latest char added
            #will become 1x61 vector
            logits = logits[:,-1]
            
            #makes a vector of size 61
            #contains the probability of each elements based off the logits
            #every probability will add up to 1
            soft = F.softmax(logits, dim=-1)

            #picks an index from the vector given
            #will pick based off of the probability given in the vector
            #picks num_samples number of samples
            next = torch.multinomial(soft, num_samples=1)
            
            #adds to tensors together
            #dim=1 means that it will add it to the next dimension
            idx = torch.concat((idx, next), dim=1)
        return idx

    
m = BigramLanguageModel(t.vocab_size)
m = m.to(t.device)
x, y = t.get_batch('training')
logits, loss = m(x,y)
#print(logits[:,-1])

#print(t.decode(m.generate(torch.zeros((1,1), dtype=torch.long),1000)[0].tolist()))

optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

batch_size = 32
print(loss)
for steps in range(300):
    xb,yb = t.get_batch('training')
    logits, loss = m(xb, yb)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
print(loss)
print(t.decode(m.generate(torch.zeros((1,1), dtype=torch.long,device=t.device),1000)[0].tolist()))

#print(F.softmax(logits, dim=0))