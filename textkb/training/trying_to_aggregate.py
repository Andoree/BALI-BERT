import torch
from torch_geometric.nn import SimpleConv

t = [
    [1, 1, 1],
    [10, 10, 10],
    [100, 100, 100],
    [1000, 1000, 1000],
    [10000, 10000, 100000]
]

t = torch.FloatTensor(t)

token_index = torch.LongTensor([0, 1, 2, 3, 4, 4, 4])
#
# # 3 entities
entity_index = torch.LongTensor([0, 2, 0, 1, 1, 0, 3])
# token_index = torch.LongTensor([0, 1, 2, 3])
#
# # 3 entities
# entity_index = torch.LongTensor([0, 1, 2, 3])


edge_index = torch.stack((token_index, entity_index))


aggr_conv = SimpleConv(aggr='sum')
entity_embs = torch.zeros(size=(4, 3), dtype=torch.float32)
# Aggregating tokens into sentence-based entity embeddings - <num_entities, h>

ttt = aggr_conv(edge_index=edge_index, x=(t, entity_embs))

for a in ttt:
    print(list([x.item() for x in a]))

###################
# tt = torch.zeros(size=(5, 4), dtype=torch.float32)
# tt[token_index, entity_index] = 1
# print("tt", tt)
# # tt = tt.unsqueeze(-1).repeat(1, 1, 3)
# # tt = tt.unsqueeze(-1).unsqueeze(1)
# tt = tt.unsqueeze(0)
#
#
# print("t", t.size())
# print("tt", tt.size())
# # print("tt", tt)
# token_embeddings = t[token_index]
# # token_embeddings = token_embeddings.unsqueeze(0).unsqueeze(-2)
# token_embeddings = token_embeddings.unsqueeze(-1)
# print("Token embddings:", token_embeddings.size())
#
# ttt = torch.matmul(tt, token_embeddings).sum(0).sum(0)
# print(ttt.size())
#
# for a in ttt:
#     print(list([x.item() for x in a]))

###########################

# for a in token_embeddings:
#     print(list([x.item() for x in a]))
# (7 x 3) * (3, 3) -> (4, 3)
# (5, 4, 3) * (7, 3) -> (4, 3)
# t = t.unsqueeze(0)
# tt = tt.unsqueeze(-1)
# print("t.size()", t.size())
# print("tt.size()", tt.size())
#
# ttt = torch.matmul(t, tt)
#
# print("ttt", ttt.size())
# ttt = list(ttt)

#
# for a in ttt:
#     print(list([x.item() for x in a]))

d = {"a": 1,
     "b": 2,
     "c": 3,
     "d": 4}

length1 = len(d)
length2 = len(d.keys())

print("length1", length1)
print("length2", length2)