"""
Train a diffusion model on images.
"""

import argparse
import torch
from patch_diffusion import dist_util, logger
from patch_diffusion.image_datasets import load_data
from patch_diffusion.resample import create_named_schedule_sampler
from patch_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from patch_diffusion.train_util import TrainLoop
import torch.multiprocessing as mp
import torch.distributed as dist

import datetime
import wandb
import numpy as np
import os 

import matplotlib.pyplot as plt

from tqdm import tqdm

def main():
    args = create_argparser().parse_args()

    args.distributed = True
    ngpus_per_node = torch.cuda.device_count()
    print(ngpus_per_node)
    if args.distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node,
                 args=(ngpus_per_node, args))
    else:
        main_worker(0, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):

    ##dist_util.setup_dist()
    ##logger.configure()

    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        
        args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend="nccl", init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank,
                                timeout=datetime.timedelta(minutes=1))

        if not args.misc and args.rank == 0:
            wandb.init(project="dropoutDiffusion", entity="dl-projects", config=args)
    else:
        wandb.init(project="dropoutDiffusion", entity="dl-projects", config=args)

    if not args.misc:
        wandb.Table.MAX_ROWS = args.num_samples *  ngpus_per_node


    args.dropout_args = {"conv_op_dropout": args.conv_op_dropout,
                         "conv_op_dropout_max": args.conv_op_dropout_max,
                         "conv_op_dropout_type": args.conv_op_dropout_type}

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    if gpu==0:
        print(model)



    if args.distributed:

        classifier_free = model.classifier_free
        num_classes = model.num_classes


        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            args.batch_size = int(args.batch_size / ngpus_per_node)
            # args.workers = int(
            #     (args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.gpu])
        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model)


        model.classifier_free = classifier_free
        model.num_classes = num_classes

    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        model = torch.nn.DataParallel(model).cuda()


    ##model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    


    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
        distributed = args.distributed,
        start = args.num_eval,
    )

    data_eval = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
        distributed = args.distributed,
        end = args.num_eval,
    )

    # for batch, cond in data_eval:
    #     batch = batch.cuda()
    #     print(batch.cpu().sum(1).sum(1).sum(1))
    #     dist.all_reduce(batch, op=dist.ReduceOp.SUM)
    #     print(batch.cpu().sum(1).sum(1).sum(1))

    #     print("batch of eval data: ", batch.shape)

    # for batch, cond in tqdm(data, desc = f"Iterating over training data"):
    #     pass

    logger.log("training...")
    trainer = TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        weight_schedule=args.weight_schedule,
        log = (gpu == -1)

    )

    evaluator = TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data_eval,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        weight_schedule=args.weight_schedule,
        log = False
    )

    
    steps = 0
    generate_every = args.generate_every
    cont = True
    Ts = [50,100,200,400,600,800]
    while cont:
        model.train()
        cont = trainer.run_loop_n(generate_every)
        model.eval()
        sample(model, diffusion, args, step = steps, gpu = gpu)
        results = evaluator.evaluate(Ts, int(args.num_eval / ngpus_per_node), ngpus_per_node)

        if gpu == 0:
            print(results)

        if gpu == 0 and  wandb.run is not None:
            wandb.log(results)
        steps += generate_every


def save_images(images, figure_path, figdims='4,4', scale='5', gpu = -1):
    

    figdims = [int(d) for d in figdims.split(',')]
    scale = float(scale)



    if figdims is None:
        m = len(images)//10 + 1
        n = 10
    else:
        m, n = figdims

    plt.figure(figsize=(scale*n, scale*m))

    imgs= []

    for i in range(len(images[:m*n])):
        plt.subplot(m, n, i+1)
        plt.imshow(images[i])
        plt.axis('off')
    
        if gpu == 0 and  wandb.run is not None:
            imgs.append(wandb.Image(images[i], caption=f"image_{i}"))

    if gpu == 0 and  wandb.run is not None:
        wandb.log({"Samples": imgs})

    plt.tight_layout()
    plt.savefig(figure_path)
    print(f"saved image samples at {figure_path}")

def sample(model,diffusion,args, step, gpu):

    run_name = ""
    rounded_steps = step - (step % 25000)
    if wandb.run is not None:
        run_name = wandb.run.name

    args.save_dir = f"samples__{run_name}_{rounded_steps}"

    if model.classifier_free and model.num_classes and args.guidance_scale != 1.0:
        model_fns = [diffusion.make_classifier_free_fn(model, args.guidance_scale)]

        def denoised_fn(x0):
            s = torch.quantile(torch.abs(x0).reshape([x0.shape[0], -1]), 0.995, dim=-1, interpolation='nearest')
            s = torch.maximum(s, torch.ones_like(s))
            s = s[:, None, None, None]
            x0 = x0.clamp(-s, s) / s
            return x0    
    else:
        model_fns = [model]
        denoised_fn = None

    logger.log("sampling...")
    all_images = []
    all_labels = []
    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = {}
        if args.class_cond:
            classes = torch.randint(
                low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
            )
            model_kwargs["y"] = classes
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        sample = sample_fn(
            model_fns,
            (args.batch_size, 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            denoised_fn=denoised_fn,
            device=dist_util.dev()
        )
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(torch.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        gathered_samples = [torch.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        if args.class_cond:
            gathered_labels = [
                torch.zeros_like(classes) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_labels, classes)
            all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        logger.log(f"created {len(all_images) * args.batch_size} samples")

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    if args.class_cond:
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: args.num_samples]
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])

        samples_index = len(os.listdir(args.save_dir))//2

        out_path = os.path.join(args.save_dir, f"samples_{shape_str}_{samples_index}.npz")
        if os.path.exists(out_path):
            print(f"Warning, there is already an npz file {out_path}, saving to a different file...")
            new_rands = np.random.randint(0, high=1e6)
            samples_index += new_rands
            out_path = os.path.join(args.save_dir, f"samples_{shape_str}_{samples_index}.npz")
        
        logger.log(f"saving to {out_path}")
        if args.class_cond:
            np.savez(out_path, arr, label_arr)
        else:
            np.savez(out_path, arr)

        out_path = os.path.join(args.save_dir, f"samples_{shape_str}_{samples_index}.png")
        if os.path.exists(out_path):
            print(f"Warning, there is already a png file {out_path}, overwriting this file...")

        save_images(arr, out_path, args.figdims, args.figscale, gpu)

    dist.barrier()
    logger.log("sampling complete")



def create_argparser():
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        weight_schedule="sqrt_snr",
        dist_url='tcp://224.66.41.62:23456',
        world_size = 1,
        misc = False,
        rank = 0,
        num_samples = 4,
        num_eval = 1000,
        clip_denoised=True,
        use_ddim=False,
        guidance_scale=1.5,
        save_dir="",
        figdims="4,4",
        figscale="5",
        generate_every = 100000,

        # model args
        conv_op_dropout=0.0,
        conv_op_dropout_max=1.0,
        conv_op_dropout_type=0,
        
    )

    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
