import logging
from typing import Dict, List, Tuple, Set, Union

import pandas as pd
from tqdm import tqdm


def get_concept_list_groupby_cui(mrconso_df: pd.DataFrame, cui2node_id: Dict[str, int]) \
        -> (Dict[int, Set[str]], Dict[int, str], Dict[str, int]):
    logging.info("Started creating CUI to terms mapping")
    node_id2terms_list: Dict[int, Set[str]] = {}

    unique_cuis_set = set(mrconso_df["CUI"].unique())
    logging.info(f"There are {len(unique_cuis_set)} unique CUIs in dataset")

    for _, row in tqdm(mrconso_df.iterrows(), miniters=mrconso_df.shape[0] // 50):
        cui = row["CUI"].strip()
        term_str = row["STR"].strip()  # TODO: Убрал .lower()
        if term_str == '':
            continue
        node_id = cui2node_id[cui]
        if node_id2terms_list.get(node_id) is None:
            node_id2terms_list[node_id] = set()
        node_id2terms_list[node_id].add(term_str.strip())
    logging.info("CUI to terms mapping is created")
    return node_id2terms_list


def extract_umls_edges(mrrel_df: pd.DataFrame, cui2node_id: Dict[str, int], ignore_not_mapped_edges=False) \
        -> List[Tuple[int, int]]:
    cui_str_set = set()
    logging.info("Started generating graph edges")
    edges: List[Tuple[int, int]] = []
    not_mapped_edges_counter = 0
    for idx, row in tqdm(mrrel_df.iterrows(), miniters=mrrel_df.shape[0] // 100, total=mrrel_df.shape[0]):
        cui_1 = row["CUI1"].strip()
        cui_2 = row["CUI2"].strip()
        if cui_1 > cui_2:
            cui_1, cui_2 = cui_2, cui_1
        if cui2node_id.get(cui_1) is not None and cui2node_id.get(cui_2) is not None:
            two_cuis_str = f"{cui_1}~~{cui_2}"
            if two_cuis_str not in cui_str_set:
                cui_1_node_id = cui2node_id[cui_1]
                cui_2_node_id = cui2node_id[cui_2]
                edges.append((cui_1_node_id, cui_2_node_id))
                edges.append((cui_2_node_id, cui_1_node_id))
            cui_str_set.add(two_cuis_str)
        else:
            if not ignore_not_mapped_edges:
                raise AssertionError(f"Either CUI {cui_1} or {cui_2} are not found in CUI2node_is mapping")
            else:
                not_mapped_edges_counter += 1
    if ignore_not_mapped_edges:
        logging.info(f"{not_mapped_edges_counter} edges are not mapped to any node")
    logging.info(f"Finished generating edges. There are {len(edges)} edges")

    return edges


def create_graph_adjacency_lists(mrrel_df: pd.DataFrame, cui2node_id: Dict[str, int],
                                 rel2rel_id: Dict[str, int], rela2rela_id: Dict[str, int],
                                 ignore_not_mapped_edges=False) -> Dict[int, List[Tuple[int, int, int]]]:
    cuis_relation_str_set = set()
    logging.info("Started generating graph edges")
    adjancency_lists: Dict[int, List[Tuple[int, int, int]]] = {}
    not_mapped_edges_counter = 0
    edges_counter = 0
    for idx, row in tqdm(mrrel_df.iterrows(), miniters=mrrel_df.shape[0] // 100, total=mrrel_df.shape[0]):
        cui_1 = row["CUI1"].strip()
        cui_2 = row["CUI2"].strip()
        rel = row["REL"]
        rela = row["RELA"]
        # Separator validation
        for att in (cui_1, cui_2, rel, rela):
            assert "|" not in str(att)
        if cui2node_id.get(cui_1) is not None and cui2node_id.get(cui_2) is not None:
            cuis_relation_str = f"{cui_1}|{cui_2}|{rel}|{rela}"
            if cuis_relation_str not in cuis_relation_str_set:
                cui_1_node_id = cui2node_id[cui_1]
                cui_2_node_id = cui2node_id[cui_2]
                rel_id = rel2rel_id[rel]
                rela_id = rela2rela_id[rela]
                if adjancency_lists.get(cui_1_node_id) is None:
                    adjancency_lists[cui_1_node_id] = []
                adjancency_lists[cui_1_node_id].append((cui_2_node_id, rel_id, rela_id))
                edges_counter += 1
            cuis_relation_str_set.add(cuis_relation_str)
        else:
            if not ignore_not_mapped_edges:
                raise AssertionError(f"Either CUI {cui_1} or {cui_2} are not found in CUI2node_id mapping")
            else:
                not_mapped_edges_counter += 1
    if ignore_not_mapped_edges:
        logging.info(f"{not_mapped_edges_counter} edges are not mapped to any node")
    logging.info(f"Finished generating edges. There are {edges_counter} edges")

    return adjancency_lists


def add_loops_to_edges_list(node_id2terms_list: Dict[int, List[str]], rel2rel_id: Dict[str, int],
                            rela2rela_id: Dict[str, int], edges: List[Tuple[int, int, int, int]]):
    """
    Takes node_id to terms list mapping and then for each node with more than 1 term(synonyms)
    adds a selp-loop edge to list of edges with "LOOP" relation
    """
    logging.info(f"Adding self-loops to the list of edge tuples. There are {len(edges)} edges")
    for node_id, terms_list in node_id2terms_list.items():
        if len(terms_list) > 1:
            loop = (node_id, node_id, rel2rel_id["LOOP"], rela2rela_id["LOOP"])
            edges.append(loop)
    logging.info(f"Finished adding self-loops to the list of edge tuples. There are {len(edges)} edges")


# def transitive_relations_filtering_recursive_call(all_ancestors_parents: Set[int], current_node_id: int,
#                                                   nodeid2parents: Dict[int, List[int]],
#                                                   nodeid2children: Dict[int, List[int]],
#                                                   deleted_edges_counter: int):
#     all_ancestors_parents_copy = all_ancestors_parents.copy()
#     current_node_parent_nodes = nodeid2parents.get(current_node_id)
#     if current_node_parent_nodes is not None:
#         all_ancestors_parents_copy.update(current_node_parent_nodes)
#
#     current_node_child_nodes = nodeid2children.get(current_node_id)
#     if current_node_child_nodes is not None:
#         for child_node in current_node_child_nodes:
#             child_node_parents = nodeid2parents.get(child_node)
#             if child_node_parents is not None:
#                 # Filtering parent nodes there are the parents of parents
#                 new_child_node_parents = [p for p in child_node_parents if p not in all_ancestors_parents]
#                 nodeid2parents[child_node] = new_child_node_parents
#                 deleted_edges_counter += len(new_child_node_parents) - len(child_node_parents)
#         for child_node in current_node_child_nodes:
#             deleted_edges_counter = transitive_relations_filtering_recursive_call(
#                 all_ancestors_parents=all_ancestors_parents_copy,
#                 current_node_id=child_node,
#                 nodeid2parents=nodeid2parents,
#                 nodeid2children=nodeid2children,
#                 deleted_edges_counter=deleted_edges_counter)
#     return deleted_edges_counter

def transitive_relations_filtering_recursive_call(all_ancestors_parents: Set[int], current_node_id: int,
                                                  nodeid2parents: Dict[int, Set[int]],
                                                  nodeid2children: Dict[int, Set[int]],
                                                  deleted_edges_counter: int):
    current_node_parent_nodes_set = nodeid2parents.get(current_node_id)
    current_node_parent_nodes_set = set() if current_node_parent_nodes_set \
                                             is None else set(current_node_parent_nodes_set)

    curr_node_parents_ancestor_parents_difference = current_node_parent_nodes_set.difference(all_ancestors_parents)
    curr_node_parents_ancestor_parents_intersection = current_node_parent_nodes_set.intersection(all_ancestors_parents)

    # Filtering nodeid2parents edges
    nodeid2parents[current_node_id] = curr_node_parents_ancestor_parents_difference

    # Filtering nodeid2children edges
    for n_id in curr_node_parents_ancestor_parents_intersection:
        nodeid2children[n_id].remove(current_node_id)
    deleted_edges_counter += len(curr_node_parents_ancestor_parents_intersection)

    curr_node_all_ancestors = current_node_parent_nodes_set.union(all_ancestors_parents)
    current_node_child_nodes = nodeid2children.get(current_node_id)
    if current_node_child_nodes is not None:
        for child_node in list(current_node_child_nodes):
            if child_node not in curr_node_all_ancestors:
                deleted_edges_counter = transitive_relations_filtering_recursive_call(
                    all_ancestors_parents=curr_node_all_ancestors,
                    current_node_id=child_node,
                    nodeid2parents=nodeid2parents,
                    nodeid2children=nodeid2children,
                    deleted_edges_counter=deleted_edges_counter)
    return deleted_edges_counter


def filter_transitive_hierarchical_relations(node_id2children: Dict[int, Set[int]],
                                             node_id2parents: Dict[int, Set[int]]):
    logging.info("Starting filtering transitive hierarchical relations. Finding root nodes.")
    root_node_ids = set(node_id2children.keys())
    for potential_root_node_id in node_id2children.keys():
        potential_root_node_id_parents = node_id2parents.get(potential_root_node_id)
        if potential_root_node_id_parents is not None and len(potential_root_node_id_parents) > 0:
            root_node_ids.remove(potential_root_node_id)
    logging.info(f"There are {len(root_node_ids)} root nodes in the hierarchy tree")
    deleted_edges_counter = 0
    for root_id in root_node_ids:
        all_ancestors_parents = set()
        deleted_edges_counter = transitive_relations_filtering_recursive_call(
            all_ancestors_parents=all_ancestors_parents,
            current_node_id=root_id, nodeid2parents=node_id2parents,
            nodeid2children=node_id2children,
            deleted_edges_counter=deleted_edges_counter)
    logging.info(f"Finished filtering transitive hierarchical relations. "
                 f"{deleted_edges_counter} edges have been deleted")


def filter_hierarchical_semantic_type_nodes(node_id2children: Dict[int, List[int]],
                                            node_id2parents: Dict[int, List[int]],
                                            node_id2terms: Dict,
                                            mrsty_df: pd.DataFrame):
    logging.info("Removing semantic type nodes")
    possible_sty_values = set(mrsty_df["STY"].unique())
    possible_sty_values = set(map(lambda s: s.lower(), map(lambda s: s.strip(), possible_sty_values)))
    logging.info(f"There are {len(possible_sty_values)} possible semantic types (STYs)")
    nodes_deleted = set()
    for node_id in list(node_id2children.keys()):
        node_terms = node_id2terms[node_id]
        assert isinstance(node_terms, list)

        if len(node_terms) == 1 and node_terms[0].strip().lower() in possible_sty_values:
            assert isinstance(node_terms[0], str)
            nodes_deleted.add(node_id)
            for child_n_id in node_id2children[node_id]:
                node_id2parents[child_n_id].remove(node_id)
            del node_id2children[node_id]
        # else:
        #     node_id2children[node_id] = [p for p in node_id2children[node_id] if not (p in possible_sty_values)]

    # for node_id in list(node_id2parents.keys()):
    #     node_terms = node_id2terms[node_id]
    #     assert isinstance(node_terms, list)
    #     if len(node_terms) == 1 and node_terms[0].strip().lower() in possible_sty_values:
    #         assert isinstance(node_terms[0], str)
    #         nodes_deleted.add(node_id)
    #         del node_id2parents[node_id]
    #     else:
    #         node_id2parents[node_id] = [p for p in node_id2parents[node_id] if not (p in possible_sty_values)]

    for node_id in list(node_id2terms.keys()):
        node_terms = node_id2terms[node_id]
        assert isinstance(node_terms, list)
        if len(node_terms) == 1 and node_terms[0].strip().lower() in possible_sty_values:
            nodes_deleted.add(node_id)
            # del node_id2token_ids_dict[node_id]
            # del node_id2terms[node_id]
    logging.info(f"Finished removing semantic type nodes. {len(nodes_deleted)} nodes have been deleted.")
    return nodes_deleted


def get_unique_sem_group_edge_rel_combinations(node_id2sem_group: Dict[int, int], edge_tuples):
    unique_edge_strings: Set[str] = set()
    combs: List[Tuple[str, str, int]] = []
    unique_src_sem_group_rel_combinations = set()
    for t in edge_tuples:
        src_node_id = t[0]
        trg_node_id = t[1]
        rel_id = t[2]
        assert isinstance(src_node_id, int) and isinstance(trg_node_id, int) and isinstance(rel_id, int)
        src_sem_group = node_id2sem_group[src_node_id]
        trg_sem_group = node_id2sem_group[trg_node_id]
        s = f"{src_sem_group}|{trg_sem_group}|{rel_id}"
        unique_src_sem_group_rel_combinations.add(f"{src_sem_group}|{rel_id}")
        unique_edge_strings.add(s)
    for src_sem_gr_rel_str in unique_src_sem_group_rel_combinations:
        src_sem_group, rel_id = src_sem_gr_rel_str.split('|')
        s = f"{src_sem_group}|SRC|{rel_id}"
        unique_edge_strings.add(s)

    for s in unique_edge_strings:
        attrs = s.split('|')
        assert len(attrs) == 3
        src_sem_group = attrs[0]
        trg_sem_group = attrs[1]
        rel_id = int(attrs[2])
        combs.append((src_sem_group, trg_sem_group, rel_id))

    return combs


def filter_edges(edge_tuples: List[Tuple[int, int, int, int]], rel2id: Dict[str, int]) \
        -> List[Tuple[int, int, int, int]]:
    logging.info(f"Starting filtering edges by relation type. There are {len(edge_tuples)} edges.")
    id2rel = {i: rel for rel, i in rel2id.items()}
    KEEP_RELATIONS = ('CHD', 'PAR', 'RN', 'RB')
    chd_rel_id = rel2id["CHD"]
    par_rel_id = rel2id["PAR"]
    new_edge_tuples = []
    for t in edge_tuples:
        src_node_id = t[0]
        trg_node_id = t[1]
        rel_id = t[2]
        rela_id = t[3]
        rel_str = id2rel[rel_id]
        if rel_str not in KEEP_RELATIONS:
            continue
        else:
            if rel_str == 'RN':
                rel_id = chd_rel_id
            if rel_str == 'RB':
                rel_id = par_rel_id
            new_edge_tuples.append((src_node_id, trg_node_id, rel_id, rela_id))
    logging.info(f"Finished filtering edges by relation type. There are {len(new_edge_tuples)} edges.")
    return new_edge_tuples


def filter_adjacency_lists(node_id2adjacency_list: Dict[int, Tuple[Union[Tuple[int, int, int], Tuple[int]]]],
                           global2local_concept_id: Dict[int, int], ensure_src_in_index):
    filtered_node_id2adjacency_list: Dict[int, Tuple[Union[Tuple[int, int, int], Tuple[int]]]] = {}
    num_dropped_edges = 0
    num_kept_edges = 0
    global_node_ids = set(global2local_concept_id.keys())
    for src_node_id, neighbors in node_id2adjacency_list.items():
        if ensure_src_in_index:
            assert src_node_id in global_node_ids
        new_neighbor_ids = tuple((t for t in neighbors if t[0] in global_node_ids))
        num_dropped_edges += len(neighbors) - len(new_neighbor_ids)
        num_kept_edges += len(new_neighbor_ids)
        if len(new_neighbor_ids) > 0:
            filtered_node_id2adjacency_list[src_node_id] = new_neighbor_ids
    logging.info(f"Finished filtering adjacency lists. Dropped {num_dropped_edges} edges, {num_kept_edges} remaining")
    return filtered_node_id2adjacency_list

