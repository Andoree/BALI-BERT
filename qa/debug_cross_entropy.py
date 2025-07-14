import torch
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

num_labels = 2

logits = torch.zeros(size=(14, ), dtype=torch.float32)


labels = torch.zeros(size=(7, ), dtype=torch.long)

print("logits.view(-1, num_labels)", logits.view(-1, num_labels).size())

loss_fct = CrossEntropyLoss()
loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))

