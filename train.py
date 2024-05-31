import argparse
import logging
import os
from pathlib import Path
from tqdm import tqdm

import numpy as np
import torch
import torch.utils.checkpoint
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed, LoggerType

from model import ukws
from dataset import libriphrase, google, qualcomm
from dataset import KWSDataLoader
from criterion import total
from criterion.utils import eer, compute_eer

from torchmetrics.aggregation import MeanMetric
from torchmetrics.classification import BinaryAUROC

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False

seed = 42
torch.random.manual_seed(seed)
np.random.seed(seed)

logger = get_logger(__name__, log_level="INFO")


def parse_args():
    parser = argparse.ArgumentParser(description="Argument supportable on this training script.")
    parser.add_argument('--epoch', required=True, type=int)
    parser.add_argument('--lr', required=True, type=float)
    parser.add_argument('--batch_size', required=False, type=int, default=4096)
    parser.add_argument('--num_workers', required=False, type=int, default=4)
    parser.add_argument('--loss_weight', default=[1.0, 1.0], nargs=2, type=float)
    parser.add_argument('--text_input', required=False, type=str, default='g2p_embed')
    parser.add_argument('--audio_input', required=False, type=str, default='both')
    parser.add_argument('--stack_extractor', action='store_true')
    parser.add_argument('--audio_noise', action='store_true')

    parser.add_argument('--frame_length', required=False, type=int, default=400)
    parser.add_argument('--hop_length', required=False, type=int, default=160)
    # parser.add_argument('--num_mel', required=False, type=int, default=40)  
    # [Note] This argument is no longer used. If you want to change num_mel, the model/lin_to_mel_matrix.npy file should be re-extracted.  
    parser.add_argument('--sample_rate', required=False, type=int, default=16000)
    parser.add_argument('--log_mel', action='store_true')

    parser.add_argument('--train_pkl', required=False, type=str, default='/home/train_both.pkl')
    parser.add_argument('--google_pkl', required=False, type=str, default='/home/google.pkl')
    parser.add_argument('--qualcomm_pkl', required=False, type=str, default='/home/qualcomm.pkl')
    parser.add_argument('--libriphrase_pkl', required=False, type=str, default='/home/test_both.pkl')

    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="pmnet_torch",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )

    parser.add_argument('--comment', required=False, type=str)
    args = parser.parse_args()

    return args


def prepare_loader(args):
    if args.audio_input == "raw":
        gemb_dir = None
    else:
        gemb_dir = '/home/google_speech_embedding/DB/'
    train_dataset = libriphrase.LibriPhraseDataset(batch_size=args.batch_size, gemb_dir=gemb_dir, 
                                                      features=args.text_input, train=True, types='both', shuffle=True, pkl=args.train_pkl, 
                                                      frame_length=args.frame_length, hop_length=args.hop_length)
    train_loader = KWSDataLoader(train_dataset, args.batch_size, shuffle=True, pin_memory=True, drop_last=True, num_workers=args.num_workers)

    val_dataset = libriphrase.LibriPhraseDataset(batch_size=args.batch_size, gemb_dir=gemb_dir, 
                                                      features=args.text_input, train=False, types='both', shuffle=True, pkl=args.libriphrase_pkl, 
                                                      frame_length=args.frame_length, hop_length=args.hop_length)
    val_easy_dataset = libriphrase.LibriPhraseDataset(batch_size=args.batch_size, gemb_dir=gemb_dir, 
                                                      features=args.text_input, train=False, types='easy', shuffle=True, pkl=args.libriphrase_pkl, 
                                                      frame_length=args.frame_length, hop_length=args.hop_length)
    val_hard_dataset = libriphrase.LibriPhraseDataset(batch_size=args.batch_size, gemb_dir=gemb_dir, 
                                                      features=args.text_input, train=False, types='hard', shuffle=True, pkl=args.libriphrase_pkl, 
                                                      frame_length=args.frame_length, hop_length=args.hop_length)
    val_google_dataset = google.GoogleCommandsDataset(batch_size=args.batch_size, gemb_dir=gemb_dir, 
                                                      features=args.text_input, shuffle=True, pkl=args.google_pkl, 
                                                      frame_length=args.frame_length, hop_length=args.hop_length)
    val_qualcomm_dataset = qualcomm.QualcommKeywordSpeechDataset(batch_size=args.batch_size, gemb_dir=gemb_dir, 
                                                      features=args.text_input, shuffle=True, pkl=args.qualcomm_pkl, 
                                                      frame_length=args.frame_length, hop_length=args.hop_length)
    
    val_dataloader = KWSDataLoader(val_dataset, args.batch_size, shuffle=True, pin_memory=True, drop_last=True, num_workers=args.num_workers)
    val_easy_dataloader = KWSDataLoader(val_easy_dataset, args.batch_size, shuffle=True, pin_memory=True, drop_last=True, num_workers=args.num_workers)
    val_hard_dataloader = KWSDataLoader(val_hard_dataset, args.batch_size, shuffle=True, pin_memory=True, drop_last=True, num_workers=args.num_workers)
    val_google_dataloader = KWSDataLoader(val_google_dataset, args.batch_size, shuffle=True, pin_memory=True, drop_last=True, num_workers=args.num_workers)
    val_qualcomm_dataloader = KWSDataLoader(val_qualcomm_dataset, args.batch_size, shuffle=True, pin_memory=True, drop_last=True, num_workers=args.num_workers)
    eval_loader = [
        val_dataloader, 
        val_easy_dataloader,
        val_hard_dataloader,
        val_google_dataloader,
        val_qualcomm_dataloader,
        ]
    
    vocab = train_dataset.nPhoneme
    train_len = len(train_dataset)

    return train_loader, eval_loader, vocab, train_len


def main():
    args = parse_args()
    logging_dir = Path(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        log_with= LoggerType.TENSORBOARD,
        project_config=accelerator_project_config,
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    set_seed(seed)

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)


    train_dataloader, eval_dataloader, vocab, train_len = prepare_loader(args)

    kwargs = {
        'vocab' : vocab,
        'text_input' : args.text_input,
        'audio_input' : args.audio_input,
        'stack_extractor' : args.stack_extractor,

        'frame_length' : args.frame_length, 
        'hop_length' : args.hop_length, 
        'num_mel'  : 40,
        'sample_rate' : args.sample_rate,
        'log_mel' : args.log_mel,
    }

    model = ukws.BaseUKWS(**kwargs)
    model.to(accelerator.device)
    
    loss_object = total.TotalLoss(weight=args.loss_weight[0])
    loss_object_sce = total.TotalLoss_SCE(weight=args.loss_weight)
    train_loss = MeanMetric()
    train_loss_d = MeanMetric()
    train_loss_sce = MeanMetric()
    
    test_loss = MeanMetric()
    test_loss_d = MeanMetric()

    train_auc = BinaryAUROC()
    train_eer = eer()

    test_auc = BinaryAUROC()
    test_eer = eer()

    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, betas = (0.9, 0.999), eps = 1e-7)

    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )

    for i in range(len(eval_dataloader)):
        eval_dataloader[i] = accelerator.prepare(eval_dataloader[i])

    loss_object, loss_object_sce, train_loss, train_loss_d, train_loss_sce, test_loss, test_loss_d, train_auc, train_eer, test_auc, test_eer = accelerator.prepare(
        loss_object, loss_object_sce, train_loss, train_loss_d, train_loss_sce, test_loss, test_loss_d, train_auc, train_eer, test_auc, test_eer
    )

    if accelerator.is_main_process:
        accelerator.init_trackers("logs")

    # Train!
    logger.info("***** Running training *****")
    logger.info(f"  Epochs = {args.epoch}")

    # log config to the tracker
    if accelerator.is_main_process:
        for tracker in accelerator.trackers:
            if tracker.name == "tensorboard":                
                for arg in vars(args):
                    tracker.writer.add_text("config/" + str(arg), str(getattr(args, arg)))

    global_step = 0
    first_epoch = 0

    for epoch in range(first_epoch, args.epoch):
        model.train()
        progress_bar = tqdm(
            range(int(args.epoch * train_len/(args.batch_size * accelerator.num_processes))),
            initial=global_step,
            desc="Epoch {}".format(epoch),
            # Only show the progress bar once on each machine.
            disable=not accelerator.is_local_main_process,
        )
        for batch_idx, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            if args.audio_input == "raw":
                if args.audio_noise:
                    prob, affinity_matrix, LD, seq_logit, affinity_mask, seq_logit_mask = model(batch["x_noisy"], batch["y"], batch["x_len"], batch["y_len"])
                else:
                    prob, affinity_matrix, LD, seq_logit, affinity_mask, seq_logit_mask = model(batch["x"], batch["y"], batch["x_len"], batch["y_len"])
            elif args.audio_input == "google_embed":
                prob, affinity_matrix, LD, seq_logit, affinity_mask, seq_logit_mask = model(batch["gemb"], batch["y"], batch["gemb_len"], batch["y_len"])
            elif args.audio_input == "both":
                if args.audio_noise:
                    prob, affinity_matrix, LD, seq_logit, affinity_mask, seq_logit_mask = model((batch["x_noisy"], batch["gemb"]), batch["y"], (batch["x_len"], batch["gemb_len"]), batch["y_len"])
                else:
                    prob, affinity_matrix, LD, seq_logit, affinity_mask, seq_logit_mask = model((batch["x"], batch["gemb"]), batch["y"], (batch["x_len"], batch["gemb_len"]), batch["y_len"])
            else:
                raise NotImplementedError

            loss, LD, LC = loss_object_sce(batch['z'], LD, batch['l'], batch['t'], seq_logit, seq_logit_mask)
            loss /= args.batch_size
            LD /= args.batch_size
            LC /= args.batch_size
            train_auc.update(prob.detach(), batch['z'].detach())
            train_eer.update(batch['z'].detach(), prob.detach())
            
            if batch_idx == 0:
                if accelerator.is_main_process:
                    for tracker in accelerator.trackers:
                        if tracker.name == "tensorboard":                
                            match_idx = torch.where(batch['z'] == 0, 1, 0).nonzero().reshape(-1)
                            unmatch_idx = torch.where(batch['z'] != 0, 1, 0).nonzero().reshape(-1)
                            match_affn = affinity_matrix[match_idx[:5]].unsqueeze(-1)
                            unmatch_affn = affinity_matrix[unmatch_idx[:5]].unsqueeze(-1)
                            tracker.writer.add_images("affinity/match", match_affn.detach().cpu().numpy(), epoch, dataformats="NHWC")
                            tracker.writer.add_images("affinity/unmatch", unmatch_affn.detach().cpu().numpy(), epoch, dataformats="NHWC")
                        else:
                            raise NotImplementedError
                        
            loss.backward()
            optimizer.step()

            train_loss.update(loss.item())
            train_loss_d.update(LD.item())
            train_loss_sce.update(LC.item())

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({
                    "epoch": epoch,
                    "train/loss/total": train_loss.compute().detach().item(), 
                    "train/loss/d": train_loss_d.compute().detach().item(), 
                    "train/loss/sce": train_loss_sce.compute().detach().item(),
                    "train/auc": train_auc.compute().detach().item(), 
                    "train/eer": train_eer.compute().detach().item(), 
                    }, 
                                 step=global_step)

            logs = {"train_loss": train_loss.compute().detach().item(),
                    "train_loss_d": train_loss_d.compute().detach().item(), 
                    "train_loss_sce": train_loss_sce.compute().detach().item(),
                    }
            progress_bar.set_postfix(**logs)
        
        # --- End of one epoch ---
        logger.info("Train auc: {}".format(train_auc.compute().detach().item()))
        logger.info("Train eer: {}".format(train_eer.compute().detach().item()))

        train_loss.reset()
        train_loss_d.reset()
        train_loss_sce.reset()
        train_auc.reset()
        train_eer.reset()

        logger.info(
            f"Running validation..."
        )

        model.eval()
        
        for loader_idx, loader in enumerate(tqdm(eval_dataloader, disable=not accelerator.is_local_main_process,)):
            for batch_idx, batch in enumerate(tqdm(loader, desc="loader_idx={}".format(loader_idx), disable=not accelerator.is_local_main_process,)):
                with torch.no_grad():
                    if args.audio_input == "raw":
                        prob, affinity_matrix, LD, seq_logit, affinity_mask, seq_logit_mask = model(batch["x"], batch["y"], batch["x_len"], batch["y_len"])
                    elif args.audio_input == "google_embed":
                        prob, affinity_matrix, LD, seq_logit, affinity_mask, seq_logit_mask = model(batch["gemb"], batch["y"], batch["gemb_len"], batch["y_len"])
                    elif args.audio_input == "both":
                        prob, affinity_matrix, LD, seq_logit, affinity_mask, seq_logit_mask = model((batch["x"], batch["gemb"]), batch["y"], (batch["x_len"], batch["gemb_len"]), batch["y_len"])
                    else:
                        raise NotImplementedError
                    
                    t_loss, LD = loss_object(batch['z'], LD)
                    t_loss /= args.batch_size
                    LD /= args.batch_size
                    
                    test_loss.update(t_loss.item())
                    test_loss_d.update(LD.item())
                    test_auc.update(prob.detach(), batch['z'].detach())
                    test_eer.update(batch['z'].detach(), prob.detach())

            logger.info(
                f"Logging validation results..."
            )

            accelerator.log({
                "val/{}/total".format(loader_idx): test_loss.compute().detach().item(),
                "val/{}/d".format(loader_idx): test_loss_d.compute().detach().item(),
                "val/{}/auc".format(loader_idx): test_auc.compute().detach().item(),
                "val/{}/eer".format(loader_idx): test_eer.compute().detach().item(),
                }, 
                            step=global_step)
            
            test_loss.reset()
            test_loss_d.reset()
            test_auc.reset()
            test_eer.reset()

            logger.info(
                f"One validation finished..."
            )

        # Save checkpoing
        # ckpt_dir = f"checkpoint/epoch_{epoch}"
        # ckpt_dir = os.path.join(args.output_dir, ckpt_dir)
        # accelerator.save_state(ckpt_dir)

    accelerator.wait_for_everyone()
    return


if __name__=="__main__":
    main()