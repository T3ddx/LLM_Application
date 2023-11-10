import torch
import torch.nn as nn
from torch.nn import functional as F
import tokenizer as t
import random
#for reproducability
torch.manual_seed(0)

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
    def forward(self, idx, targets):
        #takes in 4 (batch size, B) tensors with 8 (block size, T) chars in them
            #chars come as ints
        #idx is (B, T) so (4, 8)
        #will return a tensor of size B (4)
        #each tensor will contain another T (8) tensors that include the embeddings of each char in T
        #so the logits is just a tensor that includes all embeddings for the 4 batches of training data
        logits = self.table(idx)
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

        return logits, loss
    
    #generates a string of length max_new_token
    #takes in a tensor of the starting char idx
    def generate(self, idx, max_new_token):
        #creates max_new_token tokens
        #remember _ means we don't care about that value
            #from haskell!!!!
        for _ in range(max_new_token):
            pass

    
m = BigramLanguageModel(t.vocab_size)
logits, loss = m(t.x,t.y)
logits = logits[:,-1, :]