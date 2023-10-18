import torch
import matplotlib.pyplot as plt

file = open('names.txt', 'r')

names_not_real = file.readlines()
#makes every word into a list then adds the start and end symbol to that list
#So you have a list of lists
names = [["<S>"] + list(x.strip('\n')) + ["<E>"] for x in names_not_real]


#gets a list of every character we use with no duplicates (b/c of set)
#not including newline character
chars = sorted(list(set(''.join(names_not_real))))[1:]

#string to integer
#enumerate returns a tuple of the object and its index
#makes a dictionary of every object and its index
stoi = {char:num for num, char in enumerate(chars)}
stoi['<S>'] = 54
stoi['<E>'] = 55

#integer to string
#takes the items in stoi and reverses them
itos = {num:char for char,num in stoi.items()}

b = {}
for name in names:
    for x,y in zip(name,name[1:]):
        #sets bigram to the tuple of the 2 char
        bigram = (x,y)
        #adds the bigram to the set if not already in it with the value of 0 + 1
        #if it is, set the value to previous value + 1
        b[bigram] = b.get(bigram, 0) + 1

#sorted takes in an iterable then creates a sorted list of the elements
#takes in iterable b.items which is an iterable of tuples
#sorts it using the key
#for us it is the second element of b.items
#reverse it so we see the largest number first
sorted_names = sorted(b.items(), key = lambda s : s[1], reverse=True)

#zip pairs the elements of two iterators together in a tuple
#x,y in zip() unpacks it
#cool set builder notation btw
#[(x,y) for name in names[:10] for x,y in zip(name, name[1:])])

#tensor of all zeros
#56 b/c we have 54 characters + start char + end char
#making it int b/c we are tracking the counts
N = torch.zeros((56,56), dtype=torch.int32)

#finds the bigram of the names using zip
#finds the integer associated with each char in the bigram
#adds an instance (+1) in the row and column associated with the integers of each char
for name in names:
    for x,y in zip(name, name[1:]):
        ix = stoi[x]
        iy = stoi[y]
        N[ix,iy] += 1

#shows the graph using matplotlib
#the #%% creates a cell so we can see the plot
#took this from Andrej Karapathy
#to visualize our tensor
# %%
plt.figure(figsize=(32,32))
plt.imshow(N, cmap='Blues')
for i in range(56):
    for j in range(56):
        chstr = itos[i] + itos[j]
        plt.text(j, i, chstr, ha='center', va='bottom', color='gray')
        plt.text(j, i, N[i,j].item(), ha='center', va='top', color='gray')
plt.axis('off')
#plt.imshow(N)