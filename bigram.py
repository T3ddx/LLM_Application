import torch

file = open('names.txt', 'r')

names_not_real = file.readlines()
names = [["<S>"] + list(x.strip('\n')) + ["<E>"] for x in names_not_real]

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