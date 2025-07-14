import logging
import os
import torch
import deepspeed
from textkb.utils.io import create_dir_if_not_exists


def save_model_checkpoint(model, epoch_id: int, num_steps: int, optimizer, scheduler, scaler, output_dir: str,
                          deepspeed_flag=False, rank_flag=False):
    chkpnt_dir = os.path.join(output_dir, f"checkpoint_e_{epoch_id}_steps_{num_steps}.pth")
    try:

        if rank_flag:
            create_dir_if_not_exists(chkpnt_dir)
        create_dir_if_not_exists(chkpnt_dir)
    except Exception as e:
        pass
    if deepspeed_flag:
        deepspeed.comm.barrier()
    model.save_model(output_dir=chkpnt_dir)
    try:
        bert_encoder_state = model.bert_encoder.state_dict()
    except Exception as e:
        bert_encoder_state = model.bert_encoder.module.state_dict()
    if deepspeed_flag:
        # DeepSpeedEngine.save_checkpoint
        model.save_checkpoint(save_dir=chkpnt_dir)
    else:
        checkpoint = {
            "epoch": epoch_id + 1,
            "num_steps": num_steps,
            # "sentence_bert_encoder": model.cpu().bert_encoder.state_dict(),
            # "concept_bert_encoder": model.cpu().concept_bert_encoder.state_dict(),
            "bert_encoder": bert_encoder_state,
            # "graph_encoder": model.cpu().graph_encoder.state_dict(),
            "optimizer": optimizer,
            "scheduler": scheduler,
            "scaler": scaler
        }
        chkpnt_path = os.path.join(chkpnt_dir, "train_state.pth")

        torch.save(checkpoint, chkpnt_path)


def load_checkpoint(checkpoint_dir: str, model, device):
    logging.info(f"Loading model checkpoint from: {checkpoint_dir}")
    model.load_from_checkpoint(checkpoint_dir)
    train_state_path = os.path.join(checkpoint_dir, "train_state.pth")
    checkpoint = torch.load(train_state_path, map_location=device)

    return checkpoint

