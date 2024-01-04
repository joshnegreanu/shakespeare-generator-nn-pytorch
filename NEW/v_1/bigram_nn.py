import torch
import torch.nn.functional as F

names = open("code.txt", "r").read().splitlines()

chrs = sorted(list(set(''.join(names))))
stoi = {s:i+1 for i,s in enumerate(chrs)}
stoi['?'] = 0
itos = {i:s for s,i in stoi.items()}

t_in, t_out = [], []

for name in names:
  chars = ['?'] + list(name) + ['?']
  for char_1, char_2 in zip(chars, chars[1:]):
    
    int_1 = stoi.get(char_1)
    int_2 = stoi.get(char_2)

    t_in.append(int_1)
    t_out.append(int_2)
    
t_in = torch.tensor(t_in)
t_out = torch.tensor(t_out)

rand_gen = torch.Generator().manual_seed(2147483647)
weights_1 = torch.rand(len(stoi), len(stoi), generator=rand_gen, requires_grad=True)

num_trains = 10000

for i in range(num_trains):

    one_hots = F.one_hot(t_in, num_classes=len(stoi)).float()

    logits = one_hots @ weights_1
    counts = logits.exp()
    probs = counts / counts.sum(1, keepdims=True)
    loss = -probs[torch.arange(t_in.nelement()), t_out].log().mean() #+ 0.01*(weights**2).mean()

    print(i, loss.item())
  
    weights_1.grad = None
    loss.backward()
  
    weights_1.data += -10 * weights_1.grad

print("")

num_names = 10

for i in range(num_names):
  
    output = []
    int_val = 0
    while True:
      
        one_hots = F.one_hot(torch.tensor([int_val]), num_classes=len(stoi)).float()
        logits = one_hots @ weights_1
        counts = logits.exp()
        probs = counts / counts.sum(1, keepdims=True)
    
        int_val = torch.multinomial(probs, num_samples=1, replacement=True, generator=rand_gen).item()
        if int_val == 0:
            break
        output.append(itos.get(int_val))

    print(''.join(output))