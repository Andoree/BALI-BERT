import torch
from torch import nn, binary_cross_entropy_with_logits
from torch.nn import BCELoss

t1 = [1,2,3,4,5]
t2 = [1,2,3,6,7]


t1 = torch.LongTensor(t1)
t2 = torch.LongTensor(t2)


mask = t1 == t2
print(mask)


labels = (t1 == t2).type(torch.float32)

print(labels)


logits = torch.ones(size=(7, ), dtype=torch.float32) * 50
labels = [1, 1, 1, 1, 1, 1, 1]
labels = torch.FloatTensor(labels)
sigm = nn.Sigmoid()
bce = BCELoss()
# bce = binary_cross_entropy_with_logits()
output = bce(sigm(logits), labels)
print("o", output)
