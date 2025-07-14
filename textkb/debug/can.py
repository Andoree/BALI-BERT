# sep = '\t'
# for fname in os.listdir(args.tokenized_data_dir):
#     if fname == "config.txt":
#         continue
#     logging.info(f"Processing {fname}")
#     fpath = os.path.join(args.tokenized_data_dir, fname)
#     with open(fpath, 'r', encoding="utf-8") as inp_file:  # encoding="ascii") as inp_file:
#         print(fname)
#         for line in inp_file:
#             attrs = line.strip().split(sep)
#             data = tuple(map(int, line.strip().split(',')))
#             inp_ids_end, token_mask_end, ei_tok_idx_end, ei_ent_idx_end = data[:4]
#
#
#             sent_tokens = transformer_tokenizer.convert_ids_to_tokens(input_ids)
#             sentence = "".join((x.strip("#") if x.startswith("#") else f" {x}" for x in sent_tokens))
#
#             assert len(input_ids) == len(token_ent_binary_mask)
#
#             sent_tokens = [token for m, token in zip(token_ent_binary_mask, sent_tokens) if m == 1]
#             sampled_tokens = [sent_tokens[token_id] for token_id in edge_index_token_idx]
#             # TODO "".join((x.strip("#") if x.startswith("#") else f" {x}" for x in t
#             sampled_node_terms = [transformer_tokenizer.convert_ids_to_tokens(node_id2tokenized_terms[node_id][0])
#                                   for node_id in edge_index_entity_idx]
#             sampled_node_terms = ["".join((x.strip("#") if x.startswith("#") else f" {x}" for x in t)) for t in
#                                   sampled_node_terms]
#             assert len(sampled_tokens) == len(sampled_node_terms)
#             # TODO: UNCOMMENT!!!
#             for token, term in zip(sampled_tokens, sampled_node_terms):
#                 print(f"{token} ---> {term}")
#             print('---')
import torch
from transformers import AutoTokenizer

# p = 0.7241
# r = 0.1175
# f = (2 * p * r) / (p + r)
#
# print(f)
#
# p = 0.6667
# r = 0.0699
# f = (2 * p * r) / (p + r)
# print(f)
#

t = [
    [[1, 1, 1], [2, 2, 2], [3, 3, 3]],
    [[4, 4, 4], [5, 5, 5], [6, 6, 6]],
    [[7, 7, 7], [8, 8, 8], [9, 9, 9]]
]
t = torch.LongTensor(t)

row_index = torch.LongTensor([0, 1, 2, 0, 2, 1])
col_index = torch.LongTensor([0, 1, 2, 2, 0, 2])
tt = t[row_index, col_index]
print("tt", tt)

row_index = [0, 1, 2, 0, 2, 1]
col_index = [0, 1, 2, 2, 0, 2]
tt = t[row_index, col_index, :]
print("tt", tt)


s = "Parkinson's disease (PD) is a multifactorial disorder of the nervous system in which there is a progressive loss of dopaminergic neurons."
# microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext/
tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
tok = tokenizer.tokenize(s)
print(tok)
