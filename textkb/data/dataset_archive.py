# import random
# from typing import List, Tuple, Dict, Iterable, Sized
#
# import torch
# import torch_geometric.data
# from torch.utils.data import Dataset
# from torch_geometric.data import Data, Batch
# from torch_geometric.loader.dataloader import Collater
# from transformers import BatchEncoding
#
# from textkb.data.entity import Entity
# import spacy
#
#
# class TextGraphDataset(Dataset):
#     BATCH_ORDER = {
#         "SUBTOKEN2ENTITY_GRAPH": -1,
#     }
#     MASKING_MODES = (
#         "text",
#         "graph",
#         "both",
#         "random"
#     )
#     TEXT_GRAPH_MASKING_OPTIONS = ((False, True), (True, False), (True, True))
#
#     def __init__(self, tokenizer, sentence_input_ids: List[int], token_ent_binary_mask: List[int],
#                  edge_index_token_idx: List[int], edge_index_entity_idx: List[int], entity_node_ids: List[int],
#                  node_id2terms: Dict[int, List[str]], node_id2adjacency_list: Dict[int, Tuple[Tuple[int, int]]],
#                  node_id2input_ids: Dict[int, List[int]], max_n_neighbors: int, mask_entities: bool,
#                  mask_mentioned_concept_node: bool, masking_mode: str, sentence_max_length: int,
#                  concept_name_max_length: int):
#         assert (len(sentence_input_ids) == len(token_ent_binary_mask)
#                 == len(edge_index_token_idx) == len(edge_index_entity_idx))
#         # TODO: Узнать, PubMedBert - cased or uncased
#         # TODO: Что, если CODER - cased и потому хорош? У меня-то модели uncased!
#         # TODO: Reformat everything that is Dict[int, ....] to List
#
#         self.bert_tokenizer = tokenizer
#         self.sentence_input_ids = sentence_input_ids
#         self.token_ent_binary_mask = token_ent_binary_mask
#         self.edge_index_token_idx = edge_index_token_idx
#         self.edge_index_entity_idx = edge_index_entity_idx
#         # self.sentences = sentences
#         # self.entities = entities
#         self.entity_node_ids = entity_node_ids
#         # TODO: Возможно, всё-таки переделать обработку в потоковую, чтобы можно было распараллелить
#         # TODO: токенизацию и т.п.
#         # self.validate_entities()
#         # TODO: На выходе мне надо:
#         # Список спанов сабтокенов сущностей каждого предложения: List[List[span_start, span_end]]
#         # Список номеров концептов в словаре (TODO: Что делать с CUI-less?)
#         # Нужен граф в виде списка рёбер
#         # TODO: Наверное, текстов слишком много для того, чтобы предварительно токенизировать все
#         self.node_id2terms = node_id2terms
#         self.node_id2input_ids = node_id2input_ids
#         self.node_id2adjacency_list = node_id2adjacency_list
#         self.sentence_max_length = sentence_max_length
#         self.concept_name_max_length = concept_name_max_length
#         self.max_n_neighbors = max_n_neighbors
#         self.mask_entities = mask_entities
#         self.mask_mentioned_concept_node = mask_mentioned_concept_node
#         assert masking_mode in TextGraphDataset.MASKING_MODES
#         self.masking_mode = masking_mode
#         # self.graph_data_collator = Collater(follow_batch, exclude_keys)
#         self.MASK_TOKEN_ID: int = self.bert_tokenizer.mask_token_id
#         self.CLS_TOKEN_ID: int = self.bert_tokenizer.cls_token_id
#         self.SEP_TOKEN_ID: int = self.bert_tokenizer.sep_token_id
#         self.PAD_TOKEN_ID: int = self.bert_tokenizer.pad_token_id
#
#     # TODO: Учесть, что entity должны быть токенизированы в виде питоновских списков и уже оттранкейчены
#     def pad_input_ids(self, input_ids: List[List[int]], max_length) -> Tuple[torch.LongTensor, torch.FloatTensor]:
#         att_masks = torch.FloatTensor(
#             [[1, ] * len(lst) + [0, ] * (max_length - len(lst)) for lst in input_ids])
#         input_ids = torch.LongTensor(
#             [lst + [self.PAD_TOKEN_ID, ] * (max_length - len(lst)) for lst in input_ids])
#         return input_ids, att_masks
#
#     def sample_node_neighors_subgraph(self, node_ids_list: List[int], mask_trg_nodes, use_rel=False):
#         if use_rel:
#             raise NotImplementedError(f"use_rel : {use_rel}")
#         num_target_concepts = len(node_ids_list)
#         init_node_id = 0
#         cum_neigh_sample_size = 0
#         edge_trg_index = []
#         src_nodes_input_ids: List[int] = []
#         trg_nodes_input_ids: List[Tuple[int]] = []
#         # TODO: Кажется, при токенизации предложений я могу запомнить global_concept_id, а потом профильтровать
#         # TODO
#
#         # TODO: УЧЕСТЬ! Что если я одновременно не маскирую и concept name целевой ноды, и mention, ТО БУДЕТ ЛИК!!!
#         for trg_node_counter, target_node_id in enumerate(node_ids_list):
#             ####################################
#             # #### Processing neighbor nodes ###
#             ####################################
#             node_neighbor_ids: Tuple[Tuple[int, int]] = self.node_id2adjacency_list[target_node_id]
#             neigh_sample_size = min(self.max_n_neighbors, len(node_neighbor_ids))
#             # TODO: А как я потом узнаю batch_size? Надо эту циферку тоже сохранить!
#             neigh_input_ids_list = (random.choice(self.node_id2input_ids[t[1]])
#                                     for t in random.sample(node_neighbor_ids, neigh_sample_size))
#
#             # neigh_input_ids, neigh_att_masks = zip(*neigh_tok_out_list)
#             src_nodes_input_ids.extend(neigh_input_ids_list)
#
#             cum_neigh_sample_size += neigh_sample_size
#
#             ####################################
#             # ##### Processing target node #####
#             ####################################
#             # trg_node_input_ids, trg_node_att_mask = random.choice(self.node_id2tokenizer_output[target_node_id])
#             if mask_trg_nodes:
#                 trg_num_tokens = len(random.choice(self.node_id2input_ids[target_node_id])) - 2
#                 trg_nodes_input_ids.append(
#                     (self.CLS_TOKEN_ID,) + (self.MASK_TOKEN_ID,) * trg_num_tokens + (self.SEP_TOKEN_ID,))
#             else:
#                 trg_nodes_input_ids.append(random.choice(self.node_id2input_ids[target_node_id]))
#             edge_trg_index.extend([trg_node_counter, ] * neigh_sample_size)
#             # TODO: MASK TARGET CONCEPT OR ZERO TENSOR OPTION!!! IMPORTANT!!!!
#             init_node_id += neigh_sample_size
#             # TODO: я точно правильно стакаю два набора src?
#
#         edge_src_index = torch.arange(cum_neigh_sample_size)
#         graph_data_src_neighbors = torch_geometric.data.Data(x=torch.arange(cum_neigh_sample_size),
#                                                              edge_src_index=edge_src_index)
#
#         edge_trg_index = torch.LongTensor(edge_trg_index)
#         assert edge_src_index.size() == edge_trg_index.size()
#
#         graph_data_trg_nodes = torch_geometric.data.Data(x=torch.arange(num_target_concepts),
#                                                          edge_trg_index=edge_trg_index)
#         return src_nodes_input_ids, trg_nodes_input_ids, graph_data_src_neighbors, graph_data_trg_nodes
#
#     def label_entity_words(self, sent_tokens, sent_entities: Tuple[Entity]) -> Tuple[List[str], List[int]]:
#         labels = []
#         words = []
#         entity_length_checksum = sum(len(e.mention_str) for e in sent_entities)
#         # TODO: Как я буду предусматривать, что сущности могут быть truncated?
#         labeled_tokens_length = 0
#         for token in sent_tokens:
#             word_text = token.text
#             word_span_start, word_span_end = token.idx, token.idx + len(word_text)
#             words.append(word_text)
#             # TODO: Узнать, PubMedBert - cased or uncased
#             # TODO: Что, если CODER - cased и потому хорош? У меня-то модели uncased!
#             label = -1
#             for entity in sent_entities:
#                 e_s_start, e_s_end = entity.span_start, entity.span_end
#                 if word_span_start >= e_s_start and word_span_end <= e_s_end:
#                     assert word_text in entity.mention_str
#                     label = entity.node_id
#                     labeled_tokens_length += len(token.text)
#                     break
#                 else:
#                     assert word_text not in entity.mention_str
#
#             labels.append(label)
#         assert len(labels) == len(sent_tokens)
#         assert entity_length_checksum == labeled_tokens_length
#
#         return words, labels
#
#     def bert_tokenize_and_preserve_labels(self, sentence_words: List[str], word_concept_ids: List[int],
#                                           mask_entities):
#
#         input_ids = [self.CLS_TOKEN_ID, ]
#         # token_c_ids = [-1, ]
#         token_ent_binary_mask = [0, ]
#
#         edge_index_entity_idx = []
#         num_entity_subtokens = 0
#         current_concept_global_id = -1
#         current_concept_local_id = -1
#         actual_concept_ids: List[int] = []
#         # TODO: Кажется, я теряю информацию о соседних сущностях! Надо было и может быть всё ещё надо переделать:
#         # TODO: Токенизировать отдельно сущности, а отдельно контекст между ними??
#         # TODO: Если сущность пропала после truncation, надо это вернуть в результате работы функции
#         # TODO: В результате токенизации надо получить реальные сущности, их concept id и токены
#         for word, concept_id in zip(sentence_words, word_concept_ids):
#             assert isinstance(concept_id, int)
#             # Tokenize the word and count # of subwords
#             word_inp_ids = self.bert_tokenizer.tokenize(word, add_special_tokens=False, return_attention_mask=False,
#                                                         return_token_type_ids=False)["input_ids"]
#             n_subwords = len(word_inp_ids)
#             # TODO: Verify output manually
#             if len(input_ids) + len(word_inp_ids) > self.sentence_max_length - 1:
#                 # TODO: Вот здесь нужно что-то сделать!
#                 # TODO: А давай не будем сущности обрезать вообще? Но это риск - вдруг у меня сущность размером с
#                 # TODO: целое предложение
#                 # TODO: НАПИСАТЬ СКРИПТ, КОТОРЫЙ НАЙДЁТ САМУЮ ДЛИННУЮ СУЩНОСТЬ!
#                 if concept_id == -1:
#                     break
#             # TODO: Другой вариант - пронумеровать все trg вершины, а потом модифицировать sequence. ДОРОГО!!!!
#             # Add the same label to the new list of labels `n_subwords` times
#             # TODO: Если не нужно, удалить
#             # token_c_ids.extend([concept_id, ] * n_subwords)
#             if concept_id == -1:
#                 m = 0
#             else:
#                 if mask_entities:
#                     word_inp_ids = [self.MASK_TOKEN_ID, ] * n_subwords
#                 m = 1
#                 if current_concept_global_id != concept_id:
#                     current_concept_global_id = concept_id
#                     current_concept_local_id += 1
#                 # TODO: НА САМОМ ДЕЛЕ СЛЕВА - ЧИСЛО ENTITY-ТОКЕНОВ
#                 # TODO: СПРАВА - ЧИСЛО ИМЕНОВАННЫХ СУЩНОСТЕЙ
#                 # TODO: А ПОТОМ С ЭТИМ СЕТАПОМ ПРОВЕРИТЬ ВСЁ В КОЛАБЕ
#                 edge_index_entity_idx.extend([current_concept_local_id, ] * n_subwords)
#                 num_entity_subtokens += n_subwords
#             input_ids.extend(word_inp_ids)
#             token_ent_binary_mask.extend([m, ] * n_subwords)
#
#         input_ids.append(self.SEP_TOKEN_ID)
#
#         # token_c_ids.append(-1)
#         token_ent_binary_mask.append(0)
#         n_sent_tokens = len(input_ids)
#
#         edge_index_subtoken_idx = torch.arange(len(edge_index_entity_idx))
#         edge_index_entity_idx = torch.LongTensor(edge_index_entity_idx)
#
#         assert edge_index_subtoken_idx.size() == edge_index_entity_idx.size() == num_entity_subtokens
#         subtoken2entity_edge_index = torch.stack((edge_index_subtoken_idx, edge_index_entity_idx), dim=0)
#
#         tok_ent_tok_graph_data = Data(edge_index=subtoken2entity_edge_index)
#         # TODO: А ведь это только лишь один граф, а как быть, когда графов будет много?
#         sent_tok = BatchEncoding({
#             "input_ids": input_ids,
#         })
#         d = {
#             "tokenized_sentence": sent_tok,
#             # "token_concept_ids": token_c_ids,
#             "token_is_entity_mask": token_ent_binary_mask,
#             "tok_ent_tok_graph_data": tok_ent_tok_graph_data
#         }
#         return d
#
#     def __getitem__(self, idx):
#         if self.masking_mode == "text":
#             mask_entities, mask_nodes = True, False
#         elif self.masking_mode == "graph":
#             mask_entities, mask_nodes = False, True
#         elif self.masking_mode == "both":
#             mask_entities, mask_nodes = True, True
#         elif self.masking_mode == "random":
#             mask_entities, mask_nodes = random.choice(TextGraphDataset.TEXT_GRAPH_MASKING_OPTIONS)
#         else:
#             raise ValueError(f"Invalid masking mode: {self.masking_mode}")
#         # TODO:
#         """
#         1. На вход получаю номер текста. Текст у меня должен быть предварительно токенизирован или токенизировать
#         его надо на лету? Незамаскированный текст известен заранее, да...
#         2. Беру токенизированный текст: id, att, token_type
#         3.1 Подтягиваю номера позиций, на которых встретились именованные сущности. Это 2D лист:
#         <число сущностей в предложении, число токенов в сущности>
#         3.2 Либо в качестве эмбеддинга сущности брать эмбеддинг только первого слова.
#         """
#         sentence = self.sentences[idx]
#         sent_entities = self.entities[idx]
#         # TODO: Надо задуматься, ничего ли я не потеряю при flatten?
#         # TODO: Записать куда-нибудь себе: Это всё можно попробовать как-то адаптировать под general domain
#         # TODO: А ещё это может получиться хорошая QA-модель. Помимо того, что может оказаться хорошей для пробинга
#         entity_node_ids = self.entity_node_ids[idx]
#         # TODO: Google List to set keep order
#         # TODO: Если у меня контекстуализированные node embeddings, то нельзя схлопывать дубликаты!!!
#         word_tokens = self.word_tokenizer(sentence)
#         words, word_concept_ids = self.label_entity_words(word_tokens, sent_entities)
#         sent_concept_info = self.bert_tokenize_and_preserve_labels(
#             sentence_words=words,
#             word_concept_ids=word_concept_ids,
#             mask_entities=mask_entities)
#         sent_concept_info["entity_node_ids"] = entity_node_ids
#         # TODO: Remove assert, too expensive
#         # TODO: Recheck assert
#         # TODO: ADD DYNAMIC RANDOM MASKING: GRAPH ONLY/MENTION ONLY/BOTH
#
#         assert (len(set(t.item() for t in sent_concept_info["tok_ent_tok_graph_data"].edge_index[1]))
#                 == len(entity_node_ids))
#         src_nodes_inp_ids, trg_nodes_inp_ids, src_neighbors_graph, trg_nodes_graph = self.sample_node_neighors_subgraph(
#             entity_node_ids,
#             use_rel=self.use_rel)
#         sent_concept_info["neighbors_graph"] = src_neighbors_graph
#         sent_concept_info["trg_nodes_graph"] = trg_nodes_graph
#         sent_concept_info["src_nodes_input_ids"] = src_nodes_inp_ids
#         sent_concept_info["trg_nodes_input_ids"] = trg_nodes_inp_ids
#         # TODO: SRC и TRG всегда должны быть списком, даже если в нём лежит один элемент!
#         # TODO: А потом не забыть прибавить к src_edge_index число entities
#
#         # TODO: Почему мне вообще нужен граф? Почему нельзя использовать один shared текстовый энкодер и делать
#         # TODO: Contrastive контекстуализированных mentions к словарю?
#         # TODO: А граф можно юзать для разного типа negative sampling.
#         # TODO: А поинт потенциальной статьи в том, что сапберт и кодер хорошо решают одну задачу и забывают остальные,
#         # TODO: а моя новая модель хорошо будет решать сразу несколько задач.
#         # TODO: Запиши себе это всё куда-нибудь. У тебя есть варианты: можно предобученный граф энкодер заморозить.
#         # TODO: Можно учить 2 энкодера с нуля. Можно учить один BERT и выбрать что-то вроде TransE.
#         # TODO: Можно маскировать центральную вершину (токены в тексте), а можно не
#         # I need unique node_ids here to serve as node_idx to sample node graphs
#         # TODO: IMPORTANT! To avoid leaks, I Must either [MASK] each central node or torch.zeros()
#         # TODO: Sample graph here? Or in collate_fn?
#
#         return sent_concept_info
#
#     def collate_fn(self, batch):
#         # TODO: Я НИЧЕГО НЕ ДЕЛАЮ С ВЛОЖЕННОСТЬЮ. НА САМОМ ДЕЛЕ ОДИН ТОКЕН МОЖЕТ БЫТЬ НЕСКОЛЬКИМИ СУЩНОСТЯМИ!!
#         # TODO: Pad sentences
#         # TODO: Pad entities
#         # TODO: Truncate entities!
#         # TODO: Предусмотреть маскирование нод
#
#         # TODO: Не надо ли сделать из entity_node_ids set?
#         sent_inp_ids, token_is_entity_mask = [], []
#         # entity_node_ids = []
#         # token_concept_ids = []
#         src_node_input_ids, trg_node_input_ids = [], []
#         tok_ent_tok_graph = []
#         neighbors_graph = []
#         trg_nodes_graph = []
#         # seen_node_ids = set()
#         batch_num_trg_nodes = 0
#         batch_sent_max_length = 0
#         batch_node_max_length = 0
#         batch_num_entities = 0
#         entity_node_ids = []
#         # tok_ent_tok_graph = (sample["tok_ent_tok_graph_data"] for sample in batch)
#         for sample in batch:
#             # entity_node_ids = sample["entity_node_ids"]
#             # for n_id in entity_node_ids:
#             #     entity_node_ids.append(n_id)
#             # if n_id in seen_node_ids:
#             #     continue
#             # else:
#             #     pass
#             # seen_node_ids.add(n_id)
#             entity_node_ids.extend(sample["entity_node_ids"])
#             batch_num_entities += len(sample["entity_node_ids"])
#             sent_inp_ids.append(sample["tokenized_sentence"]["input_ids"])
#             batch_sent_max_length = max(batch_sent_max_length, len(sample["tokenized_sentence"]["input_ids"]))
#             # sent_att_mask.append(sample["tokenized_sentence"]["attention_mask"])
#             # token_concept_ids.append(sample["token_concept_ids"])
#             token_is_entity_mask.append(sample["token_is_entity_mask"])
#             tok_ent_tok_graph.append(sample["tok_ent_tok_graph"])
#             neighbors_graph.append(sample["neighbors_graph"])
#             trg_nodes_graph.append(sample["trg_nodes_graph"])
#             batch_num_trg_nodes += sample["trg_nodes_graph"].x.size()
#             sample_src_nodes_input_ids = sample["src_nodes_input_ids"]
#             sample_trg_nodes_input_ids = sample["trg_nodes_input_ids"]
#             src_node_input_ids.extend(sample_src_nodes_input_ids)
#             trg_node_input_ids.extend(sample_trg_nodes_input_ids)
#
#             src_nodes_max_length = max((len(t) for t in sample_src_nodes_input_ids))
#             trg_nodes_max_length = max((len(t) for t in sample_trg_nodes_input_ids))
#             batch_node_max_length = max(batch_node_max_length, src_nodes_max_length, trg_nodes_max_length)
#
#         sent_inp_ids, sent_att_mask = self.pad_input_ids(input_ids=sent_inp_ids, max_length=batch_sent_max_length)
#
#         # TODO: Посмотреть в collate_fn торча и сделать так же
#         # TODO: При получении node_idx глобального по батчу мне надо, скорее всего, сохранить порядок вершин?
#         # node_ids = None
#         # (batch_size, n_id, adjs) = self.neighbor_sampler.sample(sent_node_ids_list)  # TODO
#         # TODO: Может, тут предусмотреть максирование целевых концептов (mentioned concepts)?
#
#         # torch_geometric batch
#         # TODO: Check in Colab
#         # TODO: Is tuple ok or list needed?
#         subtoken_entity_batch = Batch.from_data_list(tok_ent_tok_graph, self.follow_batch, self.exclude_keys)
#         # TODO: src, trg graphs.
#         src_nodes_graph = Batch.from_data_list(neighbors_graph, self.follow_batch, self.exclude_keys)
#         trg_nodes_graph = Batch.from_data_list(trg_nodes_graph, self.follow_batch, self.exclude_keys)
#         src_nodes_edge_index = src_nodes_graph.edge_src_index + batch_num_trg_nodes
#         trg_nodes_edge_index = trg_nodes_graph.edge_trg_index
#         assert batch_num_trg_nodes == trg_nodes_graph.x.size()
#         assert src_nodes_edge_index.dim() == trg_nodes_edge_index.dim() == 1
#         concept_graph_edge_index = torch.stack((src_nodes_edge_index, trg_nodes_edge_index), dim=0)
#
#         # TODO: Удалить старый код для инпута нод, завязанный на геометрик
#         # neighbors_input_ids, neighbors_att_mask = src_nodes_graph.input_ids, src_nodes_graph.att_masks
#         # trg_nodes_input_ids, trg_nodes_att_mask = trg_nodes_graph.input_ids, trg_nodes_graph.att_masks
#
#         # concept_graph_input_ids = torch.cat((trg_nodes_input_ids, neighbors_input_ids), dim=0)
#         # concept_graph_att_mask = torch.cat((trg_nodes_att_mask, neighbors_att_mask), dim=0)
#         trg_node_input_ids.extend(src_node_input_ids)
#         node_input_ids = trg_node_input_ids
#         node_input_ids, node_att_mask = self.pad_input_ids(node_input_ids, max_length=batch_node_max_length)
#
#         node_input = (node_input_ids, node_att_mask)
#         # TODO: Это вроде в новой редакции уже не нужно
#         # sent_inp_ids = torch.stack(sent_inp_ids)
#         # sent_att_mask = torch.stack(sent_att_mask)
#         sent_input = (sent_inp_ids, sent_att_mask)
#         token_is_entity_mask = torch.stack(token_is_entity_mask)
#         # token_concept_ids = torch.stack(token_concept_ids)
#         # entity_node_ids = torch.stack(entity_node_ids)
#         subtoken2entity_edge_index = subtoken_entity_batch.edge_index
#
#         d = {
#             "sentence_input": sent_input,
#             "token_is_entity_mask": token_is_entity_mask,
#             # "token_concept_ids": token_concept_ids,
#             "entity_node_ids": entity_node_ids,
#             "subtoken2entity_edge_index": subtoken2entity_edge_index,
#             "node_input": node_input,
#             "concept_graph_edge_index": concept_graph_edge_index,
#             "num_entities": batch_num_entities
#         }
#         return d
#
#         # graph_data_src_neighbors = torch_geometric.data.Data(x=torch.arange(cum_neigh_sample_size),
#         #                                                      input_ids=src_neigh_input_ids,
#         #                                                      att_masks=src_neigh_att_masks,
#         #                                                      edge_src_index=edge_src_index)
#         #
#         # graph_data_trg_nodes = torch_geometric.data.Data(x=torch.arange(num_target_concepts),
#         #                                                  input_ids=trg_neigh_input_ids,
#         #                                                  att_masks=trg_neigh_att_masks,
#         #                                                  edge_trg_index=edge_trg_index)
#         # # TODO: merge src, trg cconcepts
#         # TODO: add num_entities to src_index
#
#     #
#     # def process_texts_and_entities(self, sentences: List[str], entities: List[List[Entity]]):
#     #     assert len(sentences) == len(entities)
#     #     spacy_tokenizer = spacy.load("en_core_web_sm")
#     #
#     #     tokenized_sentences = []
#     #     token_labels = []
#     #     for sent, sent_entities in zip(sentences, entities):
#     #         # Word tokenization
#     #         sent_words = spacy_tokenizer(sent)
#     #         # Labeling words with node ids
#     #         words, word_labels = self.label_entity_words(sent_words, sent_entities)
#     #         # Applying BERT tokenization and labelling subword tokens
#     #         sent_tok, t_labels = self.bert_tokenize_and_preserve_labels(sentence_words=words,
#     #                                                                     word_concept_ids=word_labels)
#     #         tokenized_sentences.append(sent_tok)
#     #         token_labels.append(t_labels)
#     #     return tokenized_sentences, token_labels
