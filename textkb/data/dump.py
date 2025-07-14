# # if self.t2hr_adjacency_lists.get(trg_c_id) is not None:
# #     m = 1
# #     target_concept_neighbors = self.t2hr_adjacency_lists(trg_c_id)
# #     pos_link = random.choice(target_concept_neighbors)
# #     pos_t, pos_r = pos_link[0], pos_link[1]
# #     pos_triples.append((trg_c_id, pos_r, pos_t))
# # else:
# #     m = 0
#
# # TODO: Здесь условие должно быть другим
# if self.h2rt_adjacency_lists.get(trg_c_id) is not None:
#     m = 1
#     target_concept_neighbors = self.h2rt_adjacency_lists[trg_c_id]
#     # TODO: А что, если ребра нет?
#     # TODO: КОРРАПЧУ НАЧАЛО
#     pos_link = random.choice(target_concept_neighbors)
#     pos_t, pos_r = pos_link[0], pos_link[1]
#     pos_triples.append((trg_c_id, pos_r, pos_t))
#
#     if batch_type == "head":
#         inverse_neighbors = self..get(trg_c_id)
#         # TODO: Кажется, всё-таки не так! Мне нельзя избавляться от головы, голова всегда есть
#         # TODO: Надо использовать tr2h
#
#         nni = []
#         while len(nni) < self.link_negative_sample_size:
#             neg_cands = [random.sample(range(self.num_nodes), self.link_negative_sample_size * 2)]
#             for nc in neg_cands:
#                 if self.TODO.get(nc) is None:
#                     nni.append(nc)
#                 else:
#                     if trg_c_id not in self.tr2h.get((nc, pos_r), []):
#                         neg_cands.append(nc)
#
#     elif batch_type == "tail":
#         # Corrupting target node's neighbors
#         r_pos_triples = tuple((t[0] for t in target_concept_neighbors if t[1] == pos_r))
#         n_n_ids = tuple((idx for idx in random
#                         .sample(range(self.num_nodes), self.link_negative_sample_size
#                                 + len(target_concept_neighbors)) \
#                          if idx not in r_pos_triples))[:self.link_negative_sample_size]
#         neg_node_ids.append(n_n_ids)
# else:
#     m = 0