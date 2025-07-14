import torch

t = [
    [1, 10, 100],
    [2, 20, 200],
    [3, 30, 300]
]
t = torch.FloatTensor(t)
print("t.size()", t.size())
tt = [
    [4, 40, 400],
    [5, 50, 500],
    [6, 60, 600]
]
tt = torch.FloatTensor(tt)
print("tt.size()", tt.size())

ttt = torch.stack([t, tt])
print("ttt.size()", ttt.size())

index_1 = [1, 1, 0, 0]
index_2 = [0, 1, 2, 1]
index_3 = [0, 1, 2, 2]
tttt = ttt[index_1, index_2, index_3]

print(tttt)
