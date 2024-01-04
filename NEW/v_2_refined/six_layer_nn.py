from multiprocessing.dummy import current_process
from operator import truediv
import torch
import torch.nn.functional as F
import random

names = open("shakespeare.txt", "r").read().splitlines()

chrs = sorted(list(set(''.join(names))))
stoi = {s:i+1 for i,s in enumerate(chrs)}
stoi['|'] = 0
itos = {i:s for s,i in stoi.items()}

block_size = 20

def build_dataset (names):
    t_in, t_out = [], []

    for name in names:
        context = [0] * block_size
        for chr in name + '|':
            int_val = stoi[chr]
            t_in.append(context)
            t_out.append(int_val)
            context = context [1:] + [int_val]

    t_in = torch.tensor(t_in)
    t_out = torch.tensor(t_out)
    return t_in, t_out

random.seed(42)
random.shuffle(names)
n1 = int(0.8*len(names))
n2 = int(0.9*len(names))

t_in_train, t_out_train = build_dataset(names[:n1])
t_in_dev, t_out_dev = build_dataset(names[n1:n2])
t_in_test, t_out_test = build_dataset(names[n2:])

class linear_layer:

    def __init__(self, fan_in, fan_out, bias=True):
        self.weight = torch.randn((fan_in, fan_out), generator=rand_gen) / fan_in**0.5
        if bias:
            self.bias = torch.zeros(fan_out)
        else:
            self.bias = None

    def __call__ (self, input):
        self.output = input @ self.weight
        if self.bias is not None:
            self.output += self.bias
        return self.output
    
    def params (self):
        return [self.weight] + ([] if self.bias is None else [self.bias])

class batch_norm_layer:

    def __init__ (self, dimension, avg_momentum=0.01):
        self.avg_momentum = avg_momentum
        self.train = True

        self.gaus_weight = torch.ones(dimension)
        self.gaus_bias = torch.zeros(dimension)

        self.running_mean = torch.zeros(dimension)
        self.running_std = torch.ones(dimension)

    def __call__ (self, input):
        if self.train:
            input_mean = input.mean(0, keepdim=True)
            input_std = input.std(0, keepdim=True)

        else:
            input_mean = self.running_mean
            input_std = self.running_std
        
        input_gaus = (input - input_mean) / input_std
        self.output = self.gaus_weight * input_gaus + self.gaus_bias

        if self.train:
            with torch.no_grad():
                self.running_mean = (1 - self.avg_momentum) * self.running_mean + self.avg_momentum * input_mean
                self.running_std = (1 - self.avg_momentum) * self.running_std + self.avg_momentum * input_std
        
        return self.output
    
    def params (self):
        return [self.gaus_weight, self.gaus_bias]

class tanh_layer:

    def __call__ (self, input):
        self.output = torch.tanh(input)
        return self.output
    
    def params (self):
        return []

batch_size = 40
num_embed = 10
num_hid = 100
vocab_size = len(stoi)

rand_gen = torch.Generator().manual_seed(2147483647)

lookup = torch.randn((vocab_size, num_embed), generator=rand_gen)

neural_net_layers = [
    linear_layer(num_embed * block_size, num_hid, bias=False), batch_norm_layer(num_hid), tanh_layer(),
    linear_layer(num_hid, num_hid, bias=False),     batch_norm_layer(num_hid),      tanh_layer(),
    linear_layer(num_hid, num_hid, bias=False),     batch_norm_layer(num_hid),      tanh_layer(),
    linear_layer(num_hid, num_hid, bias=False),     batch_norm_layer(num_hid),      tanh_layer(),
    linear_layer(num_hid, num_hid, bias=False),     batch_norm_layer(num_hid),      tanh_layer(),
    linear_layer(num_hid, vocab_size, bias=False),  batch_norm_layer(vocab_size)
]

with torch.no_grad():
  neural_net_layers[-1].gaus_weight *= 0.1
  
  for layer in neural_net_layers[:-1]:
    if isinstance(layer, linear_layer):
      layer.weight *= 5/3

params = [lookup] + [param for layer in neural_net_layers for param in layer.params()]
for param in params:
    param.requires_grad = True

num_tests = 100000
percentage = 0

for i in range(num_tests):

    batch = torch.randint(0, t_in_train.shape[0], (batch_size,))

    embedding = lookup[t_in_train[batch]]

    curr = embedding.view(embedding.shape[0], -1)

    for layer in neural_net_layers:
        curr = layer(curr)

    loss = F.cross_entropy(curr, t_out_train[batch])

    for param in params:
        param.grad = None

    loss.backward()

    if i < num_tests/2:
        learning_rate = 0.1
    else:
        learning_rate = 0.01

    for param in params:
        param.data += -learning_rate * param.grad
    
    new_percentage = int(i/num_tests*100)
    if new_percentage > percentage:
        print("%2d%% --> %f loss" % (new_percentage, loss.item()))
        percentage = new_percentage

for layer in neural_net_layers:
  layer.train = False

def run(dataset_in, dataset_out):
    embedding = lookup[dataset_in]
    curr = embedding.view(embedding.shape[0], -1)

    for layer in neural_net_layers:
        curr = layer(curr)

    return (F.cross_entropy(curr, dataset_out)).item()

print()

print("train set loss:", run(t_in_train, t_out_train))
print("dev set loss:", run(t_in_dev, t_out_dev))
print("test set loss:", run(t_in_test, t_out_test))

print()

rand_gen_2 = torch.Generator().manual_seed(2147483647+10)

for i in range(10):
    out = []
    context = [0] * block_size
    while True:
        embedding = lookup[torch.tensor([context])]
        curr = embedding.view(embedding.shape[0], -1)

        for layer in neural_net_layers:
            curr = layer(curr)
        
        probs = F.softmax(curr, dim=1)
        index = torch.multinomial(probs, num_samples=1, generator=rand_gen_2).item()
        context = context[1:] + [index]
        if index == 0:
            break
        out.append(index)
    
    print(''.join(itos[j] for j in out))