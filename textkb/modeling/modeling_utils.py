import torch


def bert_encode(bert_encoder, input_ids, att_mask, output_hidden_states=False):
    text_emb = bert_encoder(input_ids, attention_mask=att_mask,
                            return_dict=True,
                            output_hidden_states=output_hidden_states)
    if output_hidden_states:
        return text_emb
    else:
        return text_emb['last_hidden_state']


def bert_pool_specific_layer(bert_output_dict, reverse_layer_id):
    hidden_states_tuple = bert_output_dict["hidden_states"]
    bert_emb = hidden_states_tuple[reverse_layer_id]
    return bert_emb


def bert_pooling(emb, pooling_type, att_mask=None):
    assert pooling_type in ("cls", "mean", "mean1")
    if pooling_type == "mean":
        emb = mean_pooling(emb, att_mask)
    elif pooling_type == "mean1":
        emb = mean_pooling(emb[:, 1:, :], att_mask[:, 1:])
    elif pooling_type == "cls":
        emb = cls_pooling(emb)
    else:
        RuntimeError(f"Unsupported BERT pooling: {pooling_type}")
    return emb


def mean_pooling(embs, att_mask):
    """
    :param embs: <batch, seq, h>
    :param att_mask: <batch, seq>
    :return:
    """
    # <batch, seq, h>
    input_mask_expanded = att_mask.unsqueeze(-1).expand(embs.size()).float()
    # <batch, h>
    sum_embeddings = torch.sum(embs * input_mask_expanded, 1)
    # <batch, h>
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    return sum_embeddings / sum_mask


def cls_pooling(embs):
    return embs[:, 0]


def shuffle_neighboring_nodes(edge_index):
    edge_index_src = edge_index[0]
    edge_index_trg = edge_index[1]
    num_edges = len(edge_index_trg)
    perm_trg_nodes = torch.randperm(num_edges)

    corr_edge_index_trg = edge_index_trg[perm_trg_nodes]
    corr_edge_index = torch.stack((edge_index_src, corr_edge_index_trg)).to(edge_index.device)

    return corr_edge_index
