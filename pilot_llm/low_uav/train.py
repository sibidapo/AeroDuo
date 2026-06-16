"""
train.py — Stage 2 training for the AeroDuo low-UAV action head.

Trains FeatureMerger + DiT + state/action encoders jointly via a flow-matching
MSE loss.  Stage 1 (AeroDuoPolicy) and LowVLMEncoder are both fully frozen.

Architecture
------------
Frozen (called under torch.no_grad):
    AeroDuoPolicy.encode_graph  BEV + instruction + poses → z_graph [B, T, D_g]
    LowVLMEncoder               front-cam + instruction   → vl_embs [B, L, H]

Trainable:
    LowUAVActionHead
        FeatureMerger    cross-attention: vl_embs × z_graph
        DiT              flow-matching denoiser
        state_encoder    MLP  [B, 1, 5] → [B, 1, emb_dim]
        action_encoder   ActionEncoder
        action_decoder   MLP

Usage
-----
python train.py \\
    --dataset_root /path/to/hal-13k \\
    --stage1_ckpt  ../high_uav/checkpoint/main/final/trainable_state.pt \\
    --output_dir   ./checkpoint/stage2/main \\
    --gradient_accumulation_steps 4 \\
    --max_train_steps 50000

Resuming
--------
python train.py ... --resume ./checkpoint/stage2/main/checkpoint-10000/trainable_state.pt
"""

from __future__ import annotations

import logging
import math
import os
import shutil
import sys
import time
import argparse
from pathlib import Path
from typing import Dict, Optional

os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

import torch
import wandb
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed, DistributedDataParallelKwargs
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from transformers import SchedulerType, get_scheduler

# ── Path bootstrap ─────────────────────────────────────────────────────────────
_PKG_DIR    = Path(__file__).resolve().parent   # low_uav/
_PARENT_DIR = _PKG_DIR.parent                    # pilot_llm/

for _p in (_PKG_DIR, _PARENT_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from config.lowuavconfig import LowUAVConfig
from lowuav_policy import LowUAVPolicy
from high_uav.aeroduo_policy import AeroDuoPolicy
from high_uav.config import AeroduoConfig
from high_uav.dataset2 import AeroduoDataset, collate_fn
from high_uav.bev_segmentation import load_models

logger = get_logger(__name__)

_POLICY_KEYS = frozenset({
    "bev_images",
    "low_uav_front_image",
    "high_uav_poses",
    "low_uav_poses_window",
    "low_uav_pose_current",
    "low_uav_traj_target",
    "instruction",
})


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AeroDuo Stage 2 — low-UAV action head training"
    )

    parser.add_argument("--dataset_root", type=str, required=True,
                        help="Root of the Hal-13k dataset.")
    parser.add_argument("--stage1_ckpt", type=str, required=True,
                        help="Path to Stage 1 trainable_state.pt checkpoint.")
    parser.add_argument("--towns", nargs="*", default=None,
                        help="Restrict to these town directory names.  Default: all Carla_*.")
    parser.add_argument("--window_T",       type=int, default=5)
    parser.add_argument("--action_horizon", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device",     type=str, default="cuda", choices=["cuda", "cpu"])

    parser.add_argument("--output_dir", type=str, default="checkpoint/stage2/main")
    parser.add_argument("--resume",     type=str, default=None,
                        help="Path to a Stage 2 trainable_state.pt to resume from.")
    parser.add_argument("--checkpointing_steps",    type=int, default=500)
    parser.add_argument("--checkpoints_total_limit", type=int, default=3)
    parser.add_argument("--log_every_n_steps",      type=int, default=1)
    parser.add_argument("--wandb_project",   type=str, default=None)
    parser.add_argument("--wandb_run_name",  type=str, default=None)

    parser.add_argument("--num_train_epochs", type=int, default=10)
    parser.add_argument("--max_train_steps",  type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--mixed_precision", type=str, default="bf16",
                        choices=["no", "fp16", "bf16"])
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--adam_beta1",    type=float, default=0.9)
    parser.add_argument("--adam_beta2",    type=float, default=0.999)
    parser.add_argument("--adam_epsilon",  type=float, default=1e-8)
    parser.add_argument("--weight_decay",  type=float, default=1e-4)
    parser.add_argument(
        "--lr_scheduler_type", type=SchedulerType, default="cosine",
        choices=["linear", "cosine", "cosine_with_restarts",
                 "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument("--num_warmup_steps", type=int, default=200)

    return parser.parse_args()


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def _checkpoint_path(output_dir: str, step: int) -> str:
    return os.path.join(output_dir, f"checkpoint-{step}")


def _save_checkpoint(
    accelerator: Accelerator,
    policy: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler,
    completed_steps: int,
    epoch: int,
    output_dir: str,
    final: bool = False,
) -> None:
    if not accelerator.is_main_process:
        return

    save_dir = (
        os.path.join(output_dir, "final")
        if final
        else _checkpoint_path(output_dir, completed_steps)
    )
    os.makedirs(save_dir, exist_ok=True)

    unwrapped = accelerator.unwrap_model(policy)
    ckpt = {
        "model":           unwrapped.trainable_state_dict(),
        "optimizer":       optimizer.state_dict(),
        "lr_scheduler":    lr_scheduler.state_dict(),
        "completed_steps": completed_steps,
        "epoch":           epoch,
    }
    ckpt_path = os.path.join(save_dir, "trainable_state.pt")
    torch.save(ckpt, ckpt_path)
    logger.info("Saved checkpoint → %s", ckpt_path)


def _prune_checkpoints(output_dir: str, limit: int) -> None:
    if limit <= 0:
        return
    checkpoints = sorted(
        [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")],
        key=lambda x: int(x.split("-")[1]),
    )
    while len(checkpoints) > limit:
        stale = os.path.join(output_dir, checkpoints.pop(0))
        shutil.rmtree(stale)
        logger.info("Pruned old checkpoint: %s", stale)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        kwargs_handlers=[ddp_kwargs],
    )

    logging.basicConfig(
        format="%(asctime)s — %(levelname)s — %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if args.seed is not None:
        set_seed(args.seed, device_specific=True)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    tb_writer: Optional[SummaryWriter] = None
    if accelerator.is_main_process:
        log_dir = os.path.join(
            args.output_dir,
            "tb_" + time.strftime("%Y%m%d-%H%M%S", time.localtime()),
        )
        tb_writer = SummaryWriter(log_dir=log_dir)

    if accelerator.is_main_process and args.wandb_project:
        wandb.init(
            entity="sibidapo3-georgia-institute-of-technology",
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={
                "learning_rate":               args.learning_rate,
                "lr_scheduler":                args.lr_scheduler_type,
                "num_warmup_steps":            args.num_warmup_steps,
                "weight_decay":                args.weight_decay,
                "adam_beta1":                  args.adam_beta1,
                "adam_beta2":                  args.adam_beta2,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "batch_size":                  args.batch_size,
                "mixed_precision":             args.mixed_precision,
                "window_T":                    args.window_T,
                "action_horizon":              args.action_horizon,
                "max_train_steps":             args.max_train_steps,
                "seed":                        args.seed,
                "dataset_root":                args.dataset_root,
                "stage1_ckpt":                 args.stage1_ckpt,
            },
            dir=args.output_dir,
            resume="allow",
        )

    # ── Frozen Stage 1 policy ─────────────────────────────────────────────────
    device_str = str(accelerator.device)
    logger.info("Loading SAM2 + GroundingDINO on %s …", device_str)
    sam2_predictor, grounding_model, _ = load_models(device=device_str)

    stage1_cfg = AeroduoConfig(
        window_T=args.window_T,
        action_horizon=args.action_horizon,
        mixed_precision=args.mixed_precision,
    )
    logger.info("Building frozen Stage 1 AeroDuoPolicy …")
    stage1_policy = AeroDuoPolicy(stage1_cfg, sam2_predictor, grounding_model)

    raw_ckpt = torch.load(args.stage1_ckpt, map_location=device_str, weights_only=True)
    state_dict = raw_ckpt["model"] if "model" in raw_ckpt else raw_ckpt
    stage1_policy.load_trainable_state_dict(state_dict, strict=True)
    stage1_policy.eval()
    for p in stage1_policy.parameters():
        p.requires_grad_(False)
    stage1_policy = stage1_policy.to(device_str)
    logger.info("Stage 1 checkpoint loaded and frozen: %s", args.stage1_ckpt)

    # ── Stage 2 policy ────────────────────────────────────────────────────────
    _precision_to_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "no": torch.float32}
    low_cfg = LowUAVConfig(
        action_horizon=args.action_horizon,
        dtype=_precision_to_dtype[args.mixed_precision],
        device=args.device,
    )

    logger.info("Building LowUAVPolicy …")
    policy = LowUAVPolicy(low_cfg, stage1_policy)

    trainable_params = [p for p in policy.action_head.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_epsilon,
        weight_decay=args.weight_decay,
    )

    if accelerator.is_main_process:
        with open(os.path.join(args.output_dir, "params_trainable.txt"), "w") as f:
            for name, param in policy.action_head.named_parameters():
                if param.requires_grad:
                    f.write(f"action_head.{name}  {list(param.shape)}\n")
        with open(os.path.join(args.output_dir, "params_frozen.txt"), "w") as f:
            f.write("# Stage 1 AeroDuoPolicy — externally frozen, not a registered submodule\n")
            for name, param in policy.vlm_encoder.named_parameters():
                f.write(f"vlm_encoder.{name}  {list(param.shape)}\n")
        n_train = sum(p.numel() for p in trainable_params)
        logger.info("Trainable parameters: %s (action_head only)", f"{n_train:,}")

    # ── Dataset + DataLoader ──────────────────────────────────────────────────
    train_dataset = AeroduoDataset(
        dataset_root=args.dataset_root,
        window_T=args.window_T,
        action_horizon=args.action_horizon,
        towns=args.towns,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )

    # ── LR scheduler ─────────────────────────────────────────────────────────
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    overrode_max_train_steps = False
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * accelerator.num_processes,
        num_training_steps=(
            args.max_train_steps
            if overrode_max_train_steps
            else args.max_train_steps * accelerator.num_processes
        ),
    )

    # ── Accelerate prepare ────────────────────────────────────────────────────
    # stage1_policy is stored as a plain attribute in LowUAVPolicy, so Accelerate
    # only sees vlm_encoder and action_head.
    policy, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        policy, optimizer, train_dataloader, lr_scheduler
    )

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # ── Resume ────────────────────────────────────────────────────────────────
    completed_steps = 0
    starting_epoch = 0
    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location="cpu")
        accelerator.unwrap_model(policy).load_trainable_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
        completed_steps = ckpt.get("completed_steps", 0)
        starting_epoch = ckpt.get("epoch", 0)
        logger.info("Resumed from %s  (step %d)", args.resume, completed_steps)

    # ── Training summary ──────────────────────────────────────────────────────
    effective_batch = args.batch_size * args.gradient_accumulation_steps * accelerator.num_processes
    logger.info("***** AeroDuo Stage 2 Training *****")
    logger.info("  Dataset windows              = %d", len(train_dataset))
    logger.info("  Epochs                       = %d", args.num_train_epochs)
    logger.info("  Batch size per step          = %d", args.batch_size)
    logger.info("  Gradient accumulation steps  = %d", args.gradient_accumulation_steps)
    logger.info("  Effective batch size         = %d", effective_batch)
    logger.info("  Total optimiser steps        = %d", args.max_train_steps)
    logger.info("  Mixed precision              = %s", args.mixed_precision)
    logger.info("  Learning rate                = %g", args.learning_rate)

    progress_bar = tqdm(
        range(args.max_train_steps),
        desc="Stage 2",
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.update(completed_steps)

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(starting_epoch, args.num_train_epochs):
        policy.train()

        grad_norm = 0.0
        for _step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(policy):
                policy_inputs: Dict = {k: v for k, v in batch.items() if k in _POLICY_KEYS}
                out = policy(**policy_inputs, device=accelerator.device)
                loss = out["loss"]

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    grad_norm = sum(
                        p.grad.detach().norm().item() ** 2
                        for group in optimizer.param_groups
                        for p in group["params"]
                        if p.grad is not None
                    ) ** 0.5
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

                if completed_steps % args.log_every_n_steps == 0:
                    step_loss = loss.detach().item()
                    current_lr = lr_scheduler.get_last_lr()[0]

                    progress_bar.set_postfix(loss=f"{step_loss:.4f}", lr=f"{current_lr:.2e}")

                    if tb_writer is not None:
                        tb_writer.add_scalar("Loss/train", step_loss,  completed_steps)
                        tb_writer.add_scalar("LR",         current_lr, completed_steps)
                        tb_writer.add_scalar("grad_norm",  grad_norm,  completed_steps)

                    if wandb.run is not None:
                        wandb.log(
                            {
                                "loss":      step_loss,
                                "lr":        current_lr,
                                "grad_norm": grad_norm,
                                "epoch":     epoch,
                            },
                            step=completed_steps,
                        )

                if completed_steps % args.checkpointing_steps == 0:
                    _save_checkpoint(
                        accelerator, policy, optimizer, lr_scheduler,
                        completed_steps, epoch, args.output_dir,
                    )
                    if accelerator.is_main_process:
                        _prune_checkpoints(args.output_dir, args.checkpoints_total_limit)

                if completed_steps >= args.max_train_steps:
                    break

        if completed_steps >= args.max_train_steps:
            break

    # ── Final save ────────────────────────────────────────────────────────────
    _save_checkpoint(
        accelerator, policy, optimizer, lr_scheduler,
        completed_steps, args.num_train_epochs - 1, args.output_dir,
        final=True,
    )
    accelerator.wait_for_everyone()

    if tb_writer is not None:
        tb_writer.close()
    if wandb.run is not None:
        wandb.finish()

    logger.info("Training complete.  Final checkpoint in %s/final/", args.output_dir)


if __name__ == "__main__":
    main()
