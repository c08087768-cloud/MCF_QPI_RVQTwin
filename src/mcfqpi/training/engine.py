from __future__ import annotations

import contextlib
import csv
import json
import math
import time
from collections import defaultdict
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:  # 允许最小 smoke 环境不安装 tensorboard
    class SummaryWriter:  # type: ignore[no-redef]
        def __init__(self, *args: object, **kwargs: object) -> None:
            print("[提示] 未安装 tensorboard，训练仍会继续，但不写 TensorBoard 日志。")

        def add_scalar(self, *args: object, **kwargs: object) -> None:
            pass

        def flush(self) -> None:
            pass

        def close(self) -> None:
            pass

from ..config import save_json, save_yaml
from ..utils import (
    AverageMeter,
    atomic_torch_save,
    capture_rng_state,
    build_reproducibility_metadata,
    count_trainable_parameters,
    save_inference_checkpoint,
    restore_rng_state,
    worker_seed_init,
)


def load_resume_checkpoint(path: str | Path) -> dict[str, Any]:
    """Load training state on CPU so RNG and DataLoader states remain valid."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError("resume checkpoint 必须是包含训练状态的字典")
    return checkpoint

StepFunction = Callable[[nn.Module, dict[str, Any], bool], tuple[torch.Tensor, Mapping[str, torch.Tensor], dict[str, Any]]]


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            result[key] = value.to(device, non_blocking=True)
        else:
            result[key] = value
    return result


def build_loader(
    dataset: Dataset[Any],
    config: dict[str, Any],
    *,
    training: bool,
    generator_seed: int = 42,
) -> DataLoader[Any]:
    generator = torch.Generator()
    generator.manual_seed(generator_seed)
    workers = int(config.get("num_workers", 4))
    return DataLoader(
        dataset,
        batch_size=int(config.get("batch_size", 32)),
        shuffle=training,
        num_workers=workers,
        pin_memory=bool(config.get("pin_memory", True)) and torch.cuda.is_available(),
        persistent_workers=workers > 0 and bool(config.get("persistent_workers", True)),
        prefetch_factor=int(config.get("prefetch_factor", 2)) if workers > 0 else None,
        drop_last=training and bool(config.get("drop_last", True)),
        worker_init_fn=worker_seed_init,
        generator=generator,
    )


def build_optimizer(model: nn.Module, config: dict[str, Any]) -> Optimizer:
    name = str(config.get("name", "adamw")).lower()
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    kwargs = {
        "lr": float(config.get("lr", 1e-4)),
        "weight_decay": float(config.get("weight_decay", 1e-4)),
    }
    if name == "adamw":
        return torch.optim.AdamW(parameters, betas=tuple(config.get("betas", [0.9, 0.999])), **kwargs)
    if name == "adam":
        return torch.optim.Adam(parameters, betas=tuple(config.get("betas", [0.9, 0.999])), **kwargs)
    if name == "sgd":
        return torch.optim.SGD(parameters, momentum=float(config.get("momentum", 0.9)), **kwargs)
    raise ValueError(f"未知优化器：{name}")


def build_scheduler(optimizer: Optimizer, config: dict[str, Any], epochs: int) -> LRScheduler | ReduceLROnPlateau | None:
    name = str(config.get("name", "cosine")).lower()
    if name in {"none", "off", ""}:
        return None
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, int(config.get("t_max", epochs))),
            eta_min=float(config.get("min_lr", 1e-6)),
        )
    if name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(config.get("factor", 0.5)),
            patience=int(config.get("patience", 5)),
            min_lr=float(config.get("min_lr", 1e-7)),
        )
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(config.get("step_size", 20)),
            gamma=float(config.get("gamma", 0.5)),
        )
    raise ValueError(f"未知 scheduler：{name}")


def _autocast_context(device: torch.device, enabled: bool, dtype_name: str = "float16"):
    if not enabled or device.type != "cuda":
        return contextlib.nullcontext()
    dtype = torch.bfloat16 if dtype_name == "bfloat16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def nonfinite_loss_message(
    loss: torch.Tensor,
    terms: Mapping[str, torch.Tensor],
    *,
    batch_index: int,
) -> str:
    """生成可定位到 batch 和损失项的数值异常信息。"""
    values = {"loss": loss, **terms}
    nonfinite = {
        name: float(value.detach().float().cpu())
        for name, value in values.items()
        if not torch.isfinite(value).all()
    }
    detail = ", ".join(f"{name}={value}" for name, value in nonfinite.items())
    return f"检测到非有限 loss：batch={batch_index}; {detail}"


def _run_epoch(
    model: nn.Module,
    loader: DataLoader[Any],
    *,
    device: torch.device,
    step_function: StepFunction,
    optimizer: Optimizer | None,
    scaler: torch.amp.GradScaler | None,
    amp_enabled: bool,
    amp_dtype: str,
    gradient_clip: float,
    accumulation_steps: int,
    max_batches: int,
    batch_scheduler: LRScheduler | None = None,
) -> tuple[float, dict[str, float], dict[str, Any] | None]:
    training = optimizer is not None
    model.train(training)
    meters: dict[str, AverageMeter] = defaultdict(AverageMeter)
    last_outputs: dict[str, Any] | None = None
    if training:
        optimizer.zero_grad(set_to_none=True)

    total_batches = len(loader)
    num_batches = min(total_batches, max_batches) if max_batches > 0 else total_batches
    if num_batches <= 0:
        raise ValueError("DataLoader 为空或 max_batches 未允许处理任何 batch")

    for batch_index, raw_batch in enumerate(loader):
        if max_batches > 0 and batch_index >= max_batches:
            break
        batch = move_batch_to_device(raw_batch, device)
        with torch.set_grad_enabled(training), _autocast_context(device, amp_enabled, amp_dtype):
            loss, terms, outputs = step_function(model, batch, training)
            group_start = (batch_index // accumulation_steps) * accumulation_steps
            group_size = min(accumulation_steps, num_batches - group_start)
            scaled_loss = loss / group_size

        if not torch.isfinite(loss):
            raise FloatingPointError(nonfinite_loss_message(loss, terms, batch_index=batch_index))
        if training:
            assert optimizer is not None
            if scaler is not None:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
            processed_batches = batch_index + 1
            should_step = (
                processed_batches % accumulation_steps == 0
                or processed_batches == num_batches
            )
            if should_step:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                if gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                # 论文复现配置可指定 scheduler.interval=step，从而按优化器更新次数
                # 而不是按 epoch 调整学习率。梯度累积时只在真正 optimizer.step 后推进。
                if batch_scheduler is not None:
                    batch_scheduler.step()

        batch_size = int(batch["phase"].shape[0]) if "phase" in batch else int(next(iter(batch.values())).shape[0])
        meters["loss"].update(float(loss.detach().cpu()), batch_size)
        for name, value in terms.items():
            meters[name].update(float(value.detach().cpu()), batch_size)
        last_outputs = {
            key: value.detach().cpu() if isinstance(value, torch.Tensor) else value
            for key, value in outputs.items()
            if key in {"phase", "speckle", "log_scale", "reconstruction"}
        }
        if last_outputs is not None:
            for key in ("speckle", "phase", "valid_mask"):
                if key in batch:
                    last_outputs[f"batch_{key}"] = batch[key].detach().cpu()

    return meters["loss"].average, {name: meter.average for name, meter in meters.items()}, last_outputs


def fit_model(
    model: nn.Module,
    train_loader: DataLoader[Any],
    val_loader: DataLoader[Any],
    *,
    device: torch.device,
    step_function: StepFunction,
    training_config: dict[str, Any],
    full_config: dict[str, Any],
    output_dir: str | Path,
    resume_checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    """通用单 GPU 训练循环，保存 best/last checkpoint 和完整历史。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_yaml(full_config, output_dir / "resolved_config.yaml")
    reproducibility_metadata = build_reproducibility_metadata(full_config)
    save_json(reproducibility_metadata, output_dir / "reproducibility.json")
    model.to(device)
    optimizer = build_optimizer(model, training_config.get("optimizer", {}))
    epochs = int(training_config.get("epochs", 40))
    scheduler_config = training_config.get("scheduler", {})
    scheduler = build_scheduler(optimizer, scheduler_config, epochs)
    scheduler_interval = str(scheduler_config.get("interval", "epoch")).lower()
    if scheduler_interval not in {"epoch", "step"}:
        raise ValueError("scheduler.interval 必须是 epoch 或 step")
    if isinstance(scheduler, ReduceLROnPlateau) and scheduler_interval == "step":
        raise ValueError("ReduceLROnPlateau 只能按 epoch/验证损失更新")
    amp_enabled = bool(training_config.get("amp", True)) and device.type == "cuda"
    amp_dtype = str(training_config.get("amp_dtype", "float16"))
    scaler: torch.amp.GradScaler | None = None
    if amp_enabled and amp_dtype != "bfloat16":
        scaler = torch.amp.GradScaler("cuda", enabled=True)

    start_epoch = 1
    best_value = math.inf
    history: list[dict[str, float]] = []
    no_improvement = 0
    optimizer_updates = 0
    if resume_checkpoint:
        checkpoint = load_resume_checkpoint(resume_checkpoint)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if scheduler is not None and checkpoint.get("scheduler") is not None:
            scheduler.load_state_dict(checkpoint["scheduler"])
        if scaler is not None and checkpoint.get("scaler") is not None:
            scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_value = float(checkpoint.get("best_value", math.inf))
        history = list(checkpoint.get("history", []))
        no_improvement = int(checkpoint.get("no_improvement", 0))
        optimizer_updates = int(checkpoint.get("optimizer_updates", 0))
        if checkpoint.get("rng_state") is not None:
            restore_rng_state(checkpoint["rng_state"])
        if checkpoint.get("train_loader_generator_state") is not None:
            train_loader.generator.set_state(checkpoint["train_loader_generator_state"])
        if checkpoint.get("val_loader_generator_state") is not None:
            val_loader.generator.set_state(checkpoint["val_loader_generator_state"])

    writer = SummaryWriter(output_dir / "tensorboard")
    patience = int(training_config.get("early_stopping_patience", 15))
    gradient_clip = float(training_config.get("gradient_clip", 1.0))
    accumulation_steps = max(1, int(training_config.get("accumulation_steps", 1)))
    max_train_batches = int(training_config.get("max_train_batches", 0))
    max_val_batches = int(training_config.get("max_val_batches", 0))
    monitor = str(training_config.get("checkpoint_monitor", "val_loss"))

    print(f"可训练参数：{count_trainable_parameters(model):,}")
    print(f"设备：{device}；AMP：{amp_enabled} ({amp_dtype})")
    start_time = time.perf_counter()
    for epoch in range(start_epoch, epochs + 1):
        train_loss, train_terms, _ = _run_epoch(
            model,
            train_loader,
            device=device,
            step_function=step_function,
            optimizer=optimizer,
            scaler=scaler,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            gradient_clip=gradient_clip,
            accumulation_steps=accumulation_steps,
            max_batches=max_train_batches,
            batch_scheduler=(
                scheduler
                if scheduler is not None
                and scheduler_interval == "step"
                and not isinstance(scheduler, ReduceLROnPlateau)
                else None
            ),
        )
        with torch.no_grad():
            val_loss, val_terms, val_outputs = _run_epoch(
                model,
                val_loader,
                device=device,
                step_function=step_function,
                optimizer=None,
                scaler=None,
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
                gradient_clip=0,
                accumulation_steps=1,
                max_batches=max_val_batches,
                batch_scheduler=None,
            )
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(val_loss)
        elif scheduler is not None and scheduler_interval == "epoch":
            scheduler.step()
        learning_rate = float(optimizer.param_groups[0]["lr"])
        record: dict[str, float] = {
            "epoch": float(epoch),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": learning_rate,
        }
        record.update({f"train_{key}": value for key, value in train_terms.items()})
        record.update({f"val_{key}": value for key, value in val_terms.items()})
        history.append(record)
        if monitor not in record:
            raise KeyError(f"checkpoint_monitor={monitor} 不存在；可选项：{sorted(record)}")
        monitored_value = float(record[monitor])
        for key, value in record.items():
            if key != "epoch":
                writer.add_scalar(key, value, epoch)
        writer.flush()

        improved = monitored_value < best_value
        if improved:
            best_value = monitored_value
            no_improvement = 0
        else:
            no_improvement += 1
        optimizer_updates += math.ceil(
            (min(len(train_loader), max_train_batches) if max_train_batches > 0 else len(train_loader))
            / accumulation_steps
        )
        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "scaler": scaler.state_dict() if scaler is not None else None,
            "best_value": best_value,
            "checkpoint_monitor": monitor,
            "no_improvement": no_improvement,
            "optimizer_updates": optimizer_updates,
            "rng_state": capture_rng_state(),
            "train_loader_generator_state": train_loader.generator.get_state(),
            "val_loader_generator_state": val_loader.generator.get_state(),
            "history": history,
            "config": full_config,
            "reproducibility": reproducibility_metadata,
        }
        atomic_torch_save(checkpoint, output_dir / "last.pt")
        if improved:
            atomic_torch_save(checkpoint, output_dir / "best.pt")
            save_inference_checkpoint(
                model,
                output_dir / "best.inference.pt",
                model_config=full_config.get("model", {}),
                metadata={
                    "epoch": epoch,
                    "best_value": monitored_value,
                    "checkpoint_monitor": monitor,
                    **reproducibility_metadata,
                },
            )

        print(
            f"Epoch {epoch:03d}/{epochs} | train={train_loss:.6f} | "
            f"val={val_loss:.6f} | lr={learning_rate:.3e} | best={best_value:.6f}"
        )
        save_json(history, output_dir / "history.json")
        if patience > 0 and no_improvement >= patience:
            print(f"早停：连续 {no_improvement} 个 epoch 未改善。")
            break

    writer.close()
    elapsed = time.perf_counter() - start_time
    result = {
        "best_val_loss": best_value,
        "epochs_completed": int(history[-1]["epoch"]) if history else 0,
        "elapsed_seconds": elapsed,
        "trainable_parameters": count_trainable_parameters(model),
        "best_checkpoint": str((output_dir / "best.pt").resolve()),
        "best_inference_checkpoint": str((output_dir / "best.inference.pt").resolve()),
        "last_checkpoint": str((output_dir / "last.pt").resolve()),
    }
    save_json(result, output_dir / "training_summary.json")
    return result
