#torch is used a lot
#need this
import torch

#for reproducability
torch.manual_seed(0)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#opens file of text
file = open('text.txt', 'r')
#gets string
text = file.read()

#makes a sorted list of each char used
#set removes all duplicates
#list makes it to a list that is interable
#sorted sorts the list
chars = sorted(list(set(text)))
#gets number of unique chars in the text
vocab_size = len(chars)

#makes a dictionary where the number of the character is the key
#and the character is the value
#string to integer
stoi = { num:ch for ch, num in enumerate(chars)}
#makes a dictionary where the character is the key
#and the number of the character is the value
#integer to string
itos = { ch:num for ch,num in enumerate(chars)}

#function
#takes in a string and makes every character in it into an integer
#returns a list
encode = lambda s : [stoi[char] for char in s]
#function
#takes in a list and makes every integer into a char
#joins the list
#returns a string
decode = lambda s: ''.join([itos[num] for num in s])

#creates a tensor of encode(text)
#encode(text) returns a list of every character turned to an int in the text
#a tensor is a sort of list
#looks like: tensor([...])
#tensor of data type long
data = torch.tensor(encode(text), dtype=torch.long)

#finds how many words 
n = int(.9*len(text))
training_data = data[n:]
validating_data = data[:n]

batch_size = 4
block_size = 8

def get_batch(type):
    data = training_data if type == 'training' else validating_data
    rand = torch.randint(len(data) - block_size, (batch_size,))
    inputs = torch.stack([data[i : i+block_size] for i in rand])
    targets = torch.stack([data[i+1 : i+1+block_size] for i in rand])
    inputs, targets = inputs.to(device), targets.to(device)
    return inputs, targets


