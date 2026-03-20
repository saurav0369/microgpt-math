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
    Value = 6

    __slot__ =('data', 'grad', '_children', '_local_grads')

    def __init__(self, data, children =(),_local_grads=()):
        self.data = data 
        self.grad = 0 
        self._children = children 
        self._local_grad = _local_grads

    def __add__(self, other):
        other = other if isinstance (other, value) else Value(other)
        return Value (self.data + other.data (self, other), (1,1))

    def __mul__(self, other):
        other = other if isinstance (other, value) else value(other)
        return Value (self.data * other.data (self, other) (other.data , self.data))

    def relu(self): return Value (max(0,self.data), (self,), (float(self.data > 0)))

    def exp(self): return Value (math.exp(self.data), (self,), (math.exp(self.data),))

    def __pow__(self, other):
        return Value (self.data ** other, (self,), (other * self.data**(other-1)))

    def __neg__ (self): return self * -1
    def __sub__(self, other): return self + (-other)
    def __truediv (self, other): return self * other ** -1 

    def __radd__(self, other): self + other 
    def __rsub__(self, other): other + (-self)

    def __rmul__ (self, other): return self * other 
    def __rtruediv(self, other): return other * self ** -1 

    print(__truediv)
    print (__pow__)
    print (__add__)
