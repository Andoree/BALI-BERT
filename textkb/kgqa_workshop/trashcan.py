# # groundTruthAnswerEntity, answerEntity
# wikidata_id2name: Dict[str, str] = {}

# for question_graph in graphs:
#   print("question_graph", question_graph.keys())
#   for node_dict in question_graph["nodes"]:
#       wikidata_id = node_dict["name_"]
#       wikidata_name = node_dict["label"]
#       wikidata_id2name[wikidata_id] = wikidata_name
# return wikidata_id2name

# def map_id2name_map(row):
#     graph_dict = row["graph"]
#     cand_answer_id = row["answerEntity"]
#     true_answer_id = row["groundTruthAnswerEntity"]
#     node_id2name = {node_dict["name_"]: node_dict["label"] \
#                     for node_dict in graph_dict["nodes"]}
#     # for k, v in row.items():
#     #   print(f"{k}:{v}")
#     # print('--')
#     # print(node_id2name)
#     # print(cand_answer_id)
#     # print(true_answer_id)
#     cand_answer_id = node_id2name[cand_answer_id]
#     true_answer_id = node_id2name[true_answer_id]
#
#     return cand_answer_id, true_answer_id