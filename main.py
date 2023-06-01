import argparse
import datetime
import json
import logging
import os
import pickle
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import wandb
import torchmetrics
from timm.utils import AverageMeter  # accuracy
from torchinfo import summary

from config import get_config
from IQA import IQA_build_loader
from logger import create_logger
from lr_scheduler import build_scheduler
from models import build_model
from optimizer import build_optimizer
from utils import (
    NativeScalerWithGradNormCount,
    auto_resume_helper,
    load_checkpoint,
    load_pretrained,
    reduce_tensor,
    save_checkpoint,
)


def parse_option():
    parser = argparse.ArgumentParser(
        "IQA method training and evaluation script", add_help=False
    )
    parser.add_argument(
        "--cfg",
        type=str,
        required=True,
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs="+",
    )

    # easy config modification
    parser.add_argument("--batch-size", type=int, help="batch size for single GPU")
    parser.add_argument("--data-path", type=str, help="path to dataset")
    parser.add_argument(
        "--zip",
        action="store_true",
        help="use zipped dataset instead of folder dataset",
    )
    parser.add_argument(
        "--cache-mode",
        type=str,
        default="part",
        choices=["no", "full", "part"],
        help="no: no cache, "
        "full: cache all data, "
        "part: sharding the dataset into nonoverlapping pieces and only cache one piece",
    )
    parser.add_argument(
        "--pretrained",
        help="pretrained weight from checkpoint, could be imagenet22k pretrained weight",
    )
    parser.add_argument("--resume", help="resume from checkpoint")
    parser.add_argument(
        "--accumulation-steps", type=int, help="gradient accumulation steps"
    )
    parser.add_argument(
        "--use-checkpoint",
        action="store_true",
        help="whether to use gradient checkpointing to save memory",
    )
    parser.add_argument(
        "--disable_amp", action="store_true", help="Disable pytorch amp"
    )
    parser.add_argument(
        "--amp-opt-level",
        type=str,
        choices=["O0", "O1", "O2"],
        help="mixed precision opt level, if O0, no amp is used (deprecated!)",
    )
    parser.add_argument(
        "--output",
        default="output",
        type=str,
        metavar="PATH",
        help="root of output folder, the full path is <output>/<model_name>/<tag> (default: output)",
    )
    parser.add_argument("--tag", help="tag of experiment")
    parser.add_argument("--eval", action="store_true", help="Perform evaluation only")
    parser.add_argument(
        "--throughput", action="store_true", help="Test throughput only"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Use torchinfo to show the flow of tensor in model",
    )
    parser.add_argument(
        "--repeat", action="store_true", help="Test model for publications"
    )
    parser.add_argument("--rnum", type=int, help="Repeat num")
    # distributed training
    local_rank = int(os.environ["LOCAL_RANK"])

    args, unparsed = parser.parse_known_args()

    config = get_config(args, local_rank)
    return args, config


def main(config):

    if dist.get_rank() == 0:
        group_name = config.TAG
        wandb_name = group_name + "_" + str(config.EXP_INDEX)
        os.makedirs(wandb_dir := (os.path.join(config.OUTPUT, "wandb")), exist_ok=True)
        wandb_runner = wandb.init(
            project="",
            entity="",
            group=group_name,
            name=wandb_name,
            config={
                "epochs": config.TRAIN.EPOCHS,
                "batch_size": config.DATA.BATCH_SIZE,
                "patch_num": config.DATA.PATCH_NUM,
            },
            dir=wandb_dir,
            reinit=True,
        )
        wandb_runner.log({"Validating SRCC": 0.0, "Validating PLCC": 0.0, "Epoch": 0})
    else:
        wandb_runner = None

    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config)
    logger.info(str(model))

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    if hasattr(model, "flops"):
        flops = model.flops()
        logger.info(f"number of GFLOPs: {flops / 1e9}")

    model.cuda()
    model_without_ddp = model

    if config.DEBUG_MODE:
        summary(
            model,
            (config.DATA.BATCH_SIZE, 3, config.DATA.IMG_SIZE, config.DATA.IMG_SIZE),
        )
        return
    (
        dataset_train,
        dataset_val,
        data_loader_train,
        data_loader_val,
    ) = IQA_build_loader(config)
    optimizer = build_optimizer(config, model)
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[config.LOCAL_RANK],
        broadcast_buffers=False,
        find_unused_parameters=False,
    )

    if dist.get_rank() == 0:
        wandb_runner.watch(model)

    loss_scaler = NativeScalerWithGradNormCount()

    if config.TRAIN.ACCUMULATION_STEPS > 1:
        lr_scheduler = build_scheduler(
            config, optimizer, len(data_loader_train) // config.TRAIN.ACCUMULATION_STEPS
        )
    else:
        lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))

    criterion = torch.nn.SmoothL1Loss()

    max_plcc = 0.0
    max_srcc = 0.0
    max_klcc = 0.0
    min_mse = 0.0

    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT)
        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(
                    f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}"
                )
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f"auto resuming from {resume_file}")
        else:
            logger.info(f"no checkpoint found in {config.OUTPUT}, ignoring auto resume")

    if config.MODEL.RESUME:
        max_plcc, epochs = load_checkpoint(
            config, model_without_ddp, optimizer, lr_scheduler, loss_scaler, logger
        )
        srcc, plcc, loss = validate(
            config, data_loader_val, model, epochs, len(dataset_val)
        )
        logger.info(
            f"SRCC and PLCC of the network on the {len(dataset_val)} test images: {srcc:.6f}, {plcc:.6f}"
        )
        if config.EVAL_MODE:
            return

    if config.MODEL.PRETRAINED and (not config.MODEL.RESUME):
        load_pretrained(config, model_without_ddp, logger)
        srcc, plcc, loss = validate(config, data_loader_val, model)
        logger.info(
            f"SRCC and PLCC of the network on the {len(dataset_val)} test images: {srcc:.6f}, {plcc:.6f}"
        )

    if config.THROUGHPUT_MODE:
        throughput(data_loader_val, model, logger)
        return

    logger.info("Start training")
    start_time = time.time()
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        data_loader_train.sampler.set_epoch(epoch)

        train_one_epoch(
            config,
            model,
            criterion,
            data_loader_train,
            optimizer,
            epoch,
            lr_scheduler,
            loss_scaler,
            wandb_runner,
        )
        if (
            dist.get_rank() == 0
            and (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1))
            and not config.DISABLE_SAVE
        ):
            save_checkpoint(
                config,
                epoch,
                model_without_ddp,
                max_plcc,
                optimizer,
                lr_scheduler,
                loss_scaler,
                logger,
            )
        srcc, plcc, klcc, meanse, loss = validate(
            config,
            data_loader_val,
            model,
            epoch,
            len(dataset_val),
            wandb_runner=wandb_runner,
        )
        if dist.get_rank() == 0:
            wandb_runner.log(
                {
                    "Validating SRCC": srcc,
                    "Validating PLCC": plcc,
                    "Validating KLCC": klcc,
                    "Validating MSE": meanse,
                    "Epoch": epoch + 1,
                }
            )
        logger.info(
            f"SRCC, PLCC, KLCC and MSE of the network on the {len(dataset_val)} test images: {srcc:.6f}, {plcc:.6f}, {klcc:.6f}, {meanse:.6f}"
        )
        if plcc >= max_plcc:
            max_plcc = max(max_plcc, plcc)
            max_srcc = srcc
            max_klcc = klcc
            min_mse = meanse
        elif plcc < 0:
            max_srcc = 0.0
            max_klcc = 0.0
        logger.info(
            f"Max PLCC: {max_plcc:.6f} Max SRCC: {max_srcc:.6f} Max KLCC: {max_klcc:.6f} Min MSE: {min_mse:.6f}"
        )
        if dist.get_rank() == 0:
            (
                wandb_runner.summary["Best PLCC"],
                wandb_runner.summary["Best SRCC"],
                wandb_runner.summary["Best KLCC"],
                wandb_runner.summary["Min MSE"],
            ) = (max_plcc, max_srcc, max_klcc, min_mse)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info("Training time {}".format(total_time_str))
    if dist.get_rank() == 0:
        wandb_runner.alert(
            title="Run Finished",
            text=f"Max PLCC: {max_plcc:.6f} Max SRCC: {max_srcc:.6f} Max KLCC: {max_klcc:.6f} Min MSE: {min_mse:.6f} Training time: {total_time_str}",
        )
        wandb_runner.finish()
        logging.shutdown()
    else:
        logging.shutdown()
    dist.barrier()
    return


def train_one_epoch(
    config,
    model,
    criterion,
    data_loader,
    optimizer,
    epoch,
    lr_scheduler,
    loss_scaler,
    wandb_runner=None,
):
    model.train()
    optimizer.zero_grad()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()
    scaler_meter = AverageMeter()
    pred_scores = []
    gt_scores = []
    start = time.time()
    end = time.time()
    for idx, (samples, targets) in enumerate(data_loader):
        samples = samples.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            outputs = model(samples)

        targets.unsqueeze_(dim=-1)
        loss = criterion(outputs, targets)
        loss = loss / config.TRAIN.ACCUMULATION_STEPS
        pred_scores.append(outputs.squeeze().detach())
        gt_scores.append(targets.squeeze().detach())
        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = (
            hasattr(optimizer, "is_second_order") and optimizer.is_second_order
        )
        # TODO: Too see if weights help the performace
        grad_norm = loss_scaler(
            loss,
            optimizer,
            clip_grad=config.TRAIN.CLIP_GRAD,
            parameters=model.parameters(),
            create_graph=is_second_order,
            update_grad=(idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0,
        )
        if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
            optimizer.zero_grad()
            lr_scheduler.step_update(
                (epoch * num_steps + idx) // config.TRAIN.ACCUMULATION_STEPS
            )
        loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        loss_meter.update(loss.item(), targets.size(0))
        if grad_norm is not None:  # loss_scaler return None if not update
            norm_meter.update(grad_norm)
        scaler_meter.update(loss_scale_value)
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]["lr"]
            wd = optimizer.param_groups[0]["weight_decay"]
            if wandb_runner:
                wandb_runner.log(
                    {
                        "Training Loss": loss_meter.val,
                        "Learning Rate": lr,
                        "Batch": epoch * len(data_loader) + idx,
                    },
                )
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f"Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t"
                f"eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t wd {wd:.4f}\t"
                f"time {batch_time.val:.4f} ({batch_time.avg:.4f})\t"
                f"loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t"
                f"grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t"
                f"loss_scale {scaler_meter.val:.4f} ({scaler_meter.avg:.4f})\t"
                f"mem {memory_used:.0f}MB"
            )
    epoch_time = time.time() - start
    try:
        pred_scores = torch.cat(pred_scores, dim=0)
        pred_scores = (
            (pred_scores.view(-1, config.DATA.PATCH_NUM)).mean(dim=-1)
        ).squeeze()
        gt_scores = torch.cat(gt_scores, dim=0)
        gt_scores = (
            (gt_scores.view(-1, config.DATA.PATCH_NUM)).mean(dim=-1)
        ).squeeze()
        train_srcc = torchmetrics.functional.spearman_corrcoef(
            pred_scores, gt_scores
        ).item()
    except:
        logger.warning("Array contains NaN or infs. Resetting cc relation to zero...")
        train_srcc = 0.0
    logger.info(
        f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}"
    )
    logger.info(f"EPOCH {epoch} training SRCC: {train_srcc}")


@torch.no_grad()
def validate(config, data_loader, model, epoch, val_len, wandb_runner=None):
    criterion = torch.nn.SmoothL1Loss()
    model.eval()
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    temp_pred_scores = []
    temp_gt_scores = []
    end = time.time()
    for idx, (images, target) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            output = model(images)
        target.unsqueeze_(dim=-1)
        temp_pred_scores.append(output.view(-1))
        temp_gt_scores.append(target.view(-1))
        # measure accuracy and record loss
        loss = criterion(output, target)
        loss = reduce_tensor(loss)

        loss_meter.update(loss.item(), target.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            if wandb_runner:
                wandb_runner.log(
                    {
                        "Validating Loss": loss_meter.val,
                        "Validate Batch": epoch * len(data_loader) + idx,
                    }
                )
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(
                f"Test: [{idx}/{len(data_loader)}]\t"
                f"Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                f"Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t"
                f"Mem {memory_used:.0f}MB"
            )
    pred_scores = torch.cat(temp_pred_scores)
    gt_scores = torch.cat(temp_gt_scores)
    # For distributed parallel, collect all data and then run metrics.
    if torch.distributed.is_initialized():
        preds_gather_list = [
            torch.zeros_like(pred_scores) for _ in range(dist.get_world_size())
        ]
        torch.distributed.all_gather(preds_gather_list, pred_scores)
        gather_preds = torch.cat(preds_gather_list, dim=0)[:val_len]
        gather_preds = (
            (gather_preds.view(-1, config.DATA.PATCH_NUM)).mean(dim=-1)
        ).squeeze()
        grotruth_gather_list = [
            torch.zeros_like(gt_scores) for _ in range(dist.get_world_size())
        ]
        torch.distributed.all_gather(grotruth_gather_list, gt_scores)
        gather_grotruth = torch.cat(grotruth_gather_list, dim=0)[:val_len]
        gather_grotruth = (
            (gather_grotruth.view(-1, config.DATA.PATCH_NUM)).mean(dim=-1)
        ).squeeze()
        final_preds = gather_preds.float().detach()
        final_grotruth = gather_grotruth.float().detach()
    try:
        test_srcc = torchmetrics.functional.spearman_corrcoef(
            final_preds, final_grotruth
        ).item()
        test_plcc = torchmetrics.functional.pearson_corrcoef(
            final_preds, final_grotruth
        ).item()
        test_klcc = torchmetrics.functional.kendall_rank_corrcoef(
            final_preds, final_grotruth
        ).item()
        meanse = torchmetrics.functional.mean_squared_error(
            final_grotruth, final_preds
        ).item()
    except:
        logger.warning("Array contains NaN or infs. Resetting cc relation to zero...")
        test_plcc = 0.0
        test_srcc = 0.0
        test_klcc = 0.0
        meanse = 0.0
    logger.info(
        f" * SRCC@ {test_srcc:.6f} PLCC@ {test_plcc:.6f} KLCC@ {test_klcc:.6f} MSE@ {meanse:.6f}"
    )
    return test_srcc, test_plcc, test_klcc, meanse, loss_meter.avg


@torch.no_grad()
def throughput(data_loader, model, logger):
    model.eval()

    for idx, (images, _) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        batch_size = images.shape[0]
        for i in range(50):
            model(images)
        torch.cuda.synchronize()
        logger.info(f"throughput averaged with 30 times")
        tic1 = time.time()
        for i in range(30):
            model(images)
        torch.cuda.synchronize()
        tic2 = time.time()
        logger.info(
            f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}"
        )
        return


if __name__ == "__main__":
    args, config = parse_option()
    if config.AMP_OPT_LEVEL:
        print("[warning] Apex amp has been deprecated, please use pytorch amp instead!")

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1
    torch.cuda.set_device(config.LOCAL_RANK)
    torch.distributed.init_process_group(
        backend="nccl", init_method="env://", world_size=world_size, rank=rank
    )
    dist.barrier()

    if args.repeat:
        assert args.rnum > 1
        num = args.rnum
    else:
        num = 1
    base_path = config.OUTPUT
    logger = logging.getLogger(name=f"{config.MODEL.NAME}")
    for i in range(num):

        if num > 1:
            config.defrost()
            config.OUTPUT = os.path.join(base_path, str(i))
            config.EXP_INDEX = i + 1
            config.SET.TRAIN_INDEX = None
            config.SET.TEST_INDEX = None
            config.freeze()
        random.seed(None)

        os.makedirs(config.OUTPUT, exist_ok=True)

        filename = "sel_num.data"
        if dist.get_rank() == 0:
            if not os.path.exists(sel_path := os.path.join(config.OUTPUT, filename)):
                sel_num = list(range(0, config.SET.COUNT))
                random.shuffle(sel_num)
                with open(os.path.join(config.OUTPUT, filename), "wb") as f:
                    pickle.dump(sel_num, f)
                del sel_num
        dist.barrier()

        with open(os.path.join(config.OUTPUT, filename), "rb") as f:
            sel_num = pickle.load(f)

        config.defrost()
        config.SET.TRAIN_INDEX = sel_num[0 : int(round(0.8 * len(sel_num)))]
        config.SET.TEST_INDEX = sel_num[int(round(0.8 * len(sel_num))) : len(sel_num)]
        config.freeze()

        seed = config.SEED + dist.get_rank()
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        cudnn.benchmark = True

        create_logger(
            logger,
            output_dir=config.OUTPUT,
            dist_rank=dist.get_rank(),
            name=f"{config.MODEL.NAME}",
        )

        if dist.get_rank() == 0:
            path = os.path.join(config.OUTPUT, "config.json")
            with open(path, "w") as f:
                f.write(config.dump())
            logger.info(f"Full config saved to {path}")

        # print config
        logger.info(config.dump())
        logger.info(json.dumps(vars(args)))

        main(config)
        logger.handlers.clear()
