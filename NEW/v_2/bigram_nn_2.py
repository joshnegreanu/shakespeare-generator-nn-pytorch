from operator import truediv
import torch
import torch.nn.functional as F
import random

names = open("names.txt", "r").read().splitlines()

rand_gen = torch.Generator().manual_seed(2147483647)

chrs = sorted(list(set(''.join(names))))
stoi = {s:i+1 for i,s in enumerate(chrs)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}

block_size = 20
embed_dim = 10
tanh_num = 300
batch_size = 30

def build_dataset (names):
    t_in, t_out = [], []

    for name in names:
        context = [0] * block_size
        for chr in name + '.':
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

lookup = torch.randn((len(stoi), embed_dim), generator=rand_gen)

weights_1 = torch.randn((embed_dim*block_size, tanh_num), generator=rand_gen) * (5/3)/((embed_dim*block_size)**0.5)
#biases_1 = torch.randn(tanh_num, generator=rand_gen) * 0.01
weights_2 = torch.randn((tanh_num, len(stoi)), generator=rand_gen) * 0.01
biases_2 = torch.randn(len(stoi), generator=rand_gen) * 0

batch_norm_gain = torch.ones((1, tanh_num))
batch_norm_bias = torch.zeros((1, tanh_num))
batch_norm_mean = torch.ones((1, tanh_num))
batch_norm_std = torch.ones((1, tanh_num))

params = [lookup, weights_1, weights_2, biases_2, batch_norm_gain, batch_norm_bias]

for param in params:
    param.requires_grad = True

num_tests = 10000
percentage = 0

for i in range(num_tests):

    mini_batch = torch.randint(0, t_in_train.shape[0], (batch_size,))

    embedding = lookup[t_in_train[mini_batch]]

    """
    nice way to do it but there's a more efficient way
    torch.cat(torch.unbind(embedding, 1), 1)
    """

    h = torch.tanh(embedding.view(-1, embed_dim*block_size) @ weights_1)

    bnm_i = h.mean(0, keepdim=True)
    bns_i = h.std(0, keepdim=True)
    
    with torch.no_grad():
        batch_norm_mean = 0.999 * batch_norm_mean + 0.001 * bnm_i
        batch_norm_std = 0.999 * batch_norm_std + 0.001 * bns_i

    h = batch_norm_gain * (h - bnm_i) / bns_i + batch_norm_bias

    logits = h @ weights_2 + biases_2

    """
    nice but not needed
    counts = logits.exp()
    probs = counts / counts.sum(1, keepdim=True)
    loss = -probs[torch.arrange(t_in.nelement()), t_out].log().mean()
    """

    loss = F.cross_entropy(logits, t_out_train[mini_batch])

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

def run(dataset_in, dataset_out):
    embedding = lookup[dataset_in]
    h = torch.tanh(embedding.view(-1, embed_dim*block_size) @ weights_1)
    h = batch_norm_gain * (h - batch_norm_mean) / batch_norm_std + batch_norm_bias
    logits = h @ weights_2 + biases_2
    return (F.cross_entropy(logits, dataset_out)).item()

print("train set loss:", run(t_in_train, t_out_train))
print("dev set loss:", run(t_in_dev, t_out_dev))
print("test set loss:", run(t_in_test, t_out_test))

rand_gen_2 = torch.Generator().manual_seed(2147483647+10)

for i in range(50):
    out = []
    context = [0] * block_size
    while True:
        emb = lookup[torch.tensor([context])]
        h = torch.tanh(emb.view(1, -1) @ weights_1)
        h = batch_norm_gain * (h - batch_norm_mean) / batch_norm_std + batch_norm_bias
        logits = h @ weights_2 + biases_2
        probs = F.softmax(logits, dim=1)
        index = torch.multinomial(probs, num_samples=1, generator=rand_gen_2).item()
        context = context[1:] + [index]
        if index == 0:
            break
        out.append(index)
    
    print(''.join(itos[j] for j in out))