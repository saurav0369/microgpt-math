import os 
import math 
import random 

random.seed(42)

if not os.path.exists('input.txt'):
    import urllib.request
    names_url = 'https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt'
    urllib.request.urlretrieve (names_url, 'input.txt')

docs = [line.strip() for line in open ('input.txt') if line.strip()]
random.shuffle(docs)
print(f"num docs : {len(docs)}")


uchars = sorted(set(''.join(docs)))
BOS = len(uchars)
vocab_size = BOS + 1 
print (f"vocab_size {vocab_size}")

#backpropagation 

#value class 

class Value :

    __slot__ =('data', 'grad', '_children', '_local_grads')

    def __init__(self, data, children =(),_local_grads=()):
        self.data = data 
        self.grad = 0 
        self._children = children 
        self._local_grad = _local_grads

v = Value(5)
print(v.data)







