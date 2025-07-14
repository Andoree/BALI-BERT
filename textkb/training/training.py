import logging
import os

import deepspeed
import torch

from textkb.utils.io import update_log_file
from textkb.utils.model_utils import save_model_checkpoint, load_checkpoint


def train_model(model, optimizer, scaler, scheduler, train_epoch_fn, val_epoch_fn, chkpnt_path: str, train_loader,
                val_loader, num_epochs: int, output_dir: str, save_chkpnt_epoch_interval: int,
                save_chkpnt_step_interval: int, device: torch.device, eval_every_n_steps, **kwargs):
    if chkpnt_path is not None:

        # model.cpu()
        logging.info(f"Resuming training from checkpoint: {chkpnt_path}...")
        checkpoint = load_checkpoint(chkpnt_path, model, device)
        logging.info(f"Successfully loaded checkpoint from: {chkpnt_path}")
        start_epoch = 0

        del checkpoint
        torch.cuda.empty_cache()
    else:
        start_epoch = 0

    model = model.to(device)
    log_file_path = os.path.join(output_dir, "training_log.txt")
    init_epoch_id = kwargs.get("init_checkpoint_epoch_id", 0)
    deepspeed_flag = kwargs.get("deepspeed_flag", False)

    train_loss_history = []
    val_loss_history = []
    logging.info(f"Starting training process from epoch {init_epoch_id}....")
    global_num_steps = 0
    init_checkpoint_step_id = kwargs.get("init_checkpoint_step_id", 0)
    logging.info(f"Starting from pre-trained checkpoint. SKIPPING FIRST {init_checkpoint_step_id} STEPS.")
    if init_checkpoint_step_id > 0:
        train_loader.dataset.return_nothing = True
        train_loader.dataset.skip_first_n_steps = init_checkpoint_step_id
    local_rank = kwargs.get("local_rank", -1)
    rank_flag = local_rank in (-1, 0)

    for ep in range(start_epoch, start_epoch + num_epochs):
        if ep < init_epoch_id:
            logging.info(f"Skipping epoch {ep}")
            global_num_steps += len(train_loader)
            continue
        model.train()
        # model = model.to(device)
        step_loss_file_path = os.path.join(output_dir, f"losses_epoch_{ep}")
        tr_epoch_losses_dict, num_steps = train_epoch_fn(model=model, train_loader=train_loader, val_loader=val_loader,
                                                         optimizer=optimizer, initial_global_num_steps=global_num_steps,
                                                         scaler=scaler, scheduler=scheduler, device=device,
                                                         eval_every_n_steps=eval_every_n_steps,
                                                         save_chkpnt_step_interval=save_chkpnt_step_interval,
                                                         step_loss_file_path=step_loss_file_path, **kwargs)
        s = '\n\t'.join(f"{l_name}: {l_val:.5f}" for l_name, l_val in tr_epoch_losses_dict.items())
        logging.info(f"Epoch {ep}, train losses:\n\t{s}")

        global_num_steps += num_steps
        log_dict = {"epoch": ep}
        log_dict.update(tr_epoch_losses_dict)
        if val_epoch_fn is not None:
            if rank_flag:
                epoch_val_losses_dict = val_epoch_fn(model=model, val_loader=val_loader, device=device, **kwargs)
                for loss_name, loss_value in epoch_val_losses_dict.items():
                    log_dict[f"val_{loss_name}"] = loss_value

                val_loss_history.append(epoch_val_losses_dict["total_loss"])
                s = '\n\t'.join(f"{l_name}: {l_val:.5f}" for l_name, l_val in epoch_val_losses_dict.items())
                logging.info(f"Epoch {ep}, val losses:\n\t{s}")
            deepspeed.comm.barrier()
        # log_dict = {"epoch": i, "train loss": epoch_train_loss, "val loss": epoch_val_loss, }
        train_loss_history.append(tr_epoch_losses_dict["total_loss"])

        if save_chkpnt_epoch_interval is not None and (ep + 1) % save_chkpnt_epoch_interval == 0:
            save_model_checkpoint(model, epoch_id=ep+1, num_steps=global_num_steps, optimizer=optimizer, scaler=scaler,
                                  scheduler=scheduler, output_dir=output_dir, deepspeed_flag=deepspeed_flag,
                                  rank_flag=rank_flag)

        update_log_file(path=log_file_path, dict_to_log=log_dict)

    save_model_checkpoint(model, epoch_id=ep+1, num_steps=global_num_steps, optimizer=optimizer, scheduler=scheduler,
                          scaler=scaler, output_dir=output_dir, deepspeed_flag=deepspeed_flag, rank_flag=rank_flag)
    # checkpoint = {
    #     'epoch': i + 1,
    #     'model_state': model.cpu().bert_encoder.state_dict(),
    #     'optimizer': optimizer,
    # }
    # chkpnt_path = os.path.join(output_dir, f"checkpoint_e_{i + 1}_steps_{global_num_steps}.pth")
    # torch.save(checkpoint, chkpnt_path)
