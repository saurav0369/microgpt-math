import os 
import math 
import random 

random.seed(42)

if not os.path.exists('input.txt'):
    import urllib.request
    names_url = 'https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt'
    urllib.request.urlretrieve (names_url, 'input.txt')

    print(names_url)

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


    def backward(self):
        topo =[]
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)

                for child in v._children:
                    build_topo(child)
                topo.append(v)

        build_topo (self)
        self.grad = 1 
        for v in reversed (topo):
            for child, _local_grads in zip (v._children, v._local_grads):
                child.grad += _local_grads * v.grad 


# paramters 
n_layer = 1 
n_embd = 16
block_size = 16
n_head = 4 
head_dim = n_embd//n_head

print(f"num of head dimensions {head_dim}")


def matrix(nout, nin, std =0.08 ):
    return [[Value (random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]

state_dict ={ 'wte': matrix(vocab_size, n_embd), 'wpe': matrix(block_size, n_embd),
'lm_head': matrix(vocab_size, n_embd)}

for i in range (n_layer):
    state_dict[f'layer{i}.attn_wq'] = matrix (n_embd, n_embd)
    state_dict[f'layer{i}.attn.wk'] = matrix (n_embd, n_embd)
    state_dict[f'layer{i}.attn.wv'] = matrix (n_embd, n_head)
    state_dict[f'layer{i}.attn.wo'] = matrix (n_embd, n_embd)

    state_dict[f'layer{i}.mlp_fc1'] = matrix(4 * n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc2'] = matrix (n_embd, 4 * n_embd)

params = []

for mat in state_dict.values():
    for row in mat :
        for p in row :

            params.append(p)

print (f'params numbers{len(params)}')            

        

