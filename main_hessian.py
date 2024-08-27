import math
from argparse import ArgumentParser
from itertools import permutations
import copy

import matplotlib.pyplot as plt
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pyhessian import hessian

from grokfast import *

from src.config import ExptSettings
from src.data_modp import mod_p_data
from src.model import Decoder


def main(args):
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # tokens for <op> and <=>. It's not clear why <=> is needed at all since it
    # has no effect on the output, but we'll leave it in to best follow the
    # paper.
    eq_token = args.p
    op_token = args.p + 1

    # "We trained a standard decoder-only transformer (Vaswani et al., 2017)
    # with causal attention masking, and calculated loss and accuracy only on
    # the answer part of the equation. For all experiments we used a
    # transformer with 2 layers, width 128, and 4 attention heads"
    model = Decoder(
        dim=128, num_layers=2, num_heads=4, num_tokens=args.p + 2, seq_len=5
    ).to(device)
    nparams = sum([p.numel() for p in model.parameters() if p.requires_grad])
    print(model)
    print(f'Total number of parameters: {nparams}')
    print(f"Device: {device}")

    data = mod_p_data(args.p, eq_token, op_token, task=args.task)

    train_idx, valid_idx = torch.randperm(data.shape[1]).split(data.shape[1] // 2)
    train_data, valid_data = data[:, train_idx], data[:, valid_idx]
    print(f"Train data: {train_data.shape}")
    print(f"Valid data: {valid_data.shape}")

    # For most experiments we used AdamW optimizer with learning rate 10−3,
    # weight decay 1, β1 = 0.9, β2 = 0.98
    optimizer = getattr(torch.optim, args.optimizer)(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2),
    )

    #  linear learning rate warmup over the first 10 updates
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda update: 1 if update > 10 else update / 10
    )

    steps_per_epoch = math.ceil(train_data.shape[1] / args.batch_size)

    its, train_acc, val_acc, train_loss, val_loss = [], [], [], [], []
    hessian_its, train_hessiantrace, val_hessiantrace = [], [], []
    grads = None
    i = 0
    
    # Compute/save hessian condition
    def save_hessian(e):
        return (e < 20 or (e+1) % args.hessian_save_every == 0)
    hessian_loss_func = lambda output, target: F.cross_entropy(output[-1], target)

    # For logging network weights.
    net_its, nets = [], []

    print(f"Epochs: {int(args.budget) // steps_per_epoch}")
    pbar = tqdm(range(int(args.budget) // steps_per_epoch))
    for e in pbar:
        
        if save_hessian(e):
            with torch.set_grad_enabled(True):
                # Compute train Hessian
                hessian_comp = hessian(model, hessian_loss_func, data=(train_data[:-1], train_data[-1]), cuda=(torch.cuda.is_available()))
                train_trace = np.mean(hessian_comp.trace())
                # print(f"Train Hessian trace: {trace}")
                train_hessiantrace.append(train_trace)
                hessian_its.append(i)
                
                # Compute valid Hessian
                hessian_comp = hessian(model, hessian_loss_func, data=(valid_data[:-1], valid_data[-1]), cuda=(torch.cuda.is_available()))
                val_trace = np.mean(hessian_comp.trace())
                # print(f"Valid Hessian trace: {trace}")
                val_hessiantrace.append(val_trace)
                
            pbar.set_description(f"Train Hessian: {train_trace}, valid Hessian: {val_trace}")
        # randomly shuffle train data
        train_data = train_data[:, torch.randperm(train_data.shape[1])]

        for data, is_train in [(train_data, True), (valid_data, False)]:

            model.train(is_train)
            total_loss = 0
            total_acc = 0

            # torch.split faster than dataloader with tensor
            dl = torch.split(data, args.batch_size, dim=1)
            for input in dl:
                input = input.to(device)
                # print(f"Input shape: {input.shape}")

                with torch.set_grad_enabled(is_train):
                    logits = model(input[:-1])
                    # calculate loss only on the answer part of the equation (last element
                    loss = F.cross_entropy(logits[-1], input[-1])
                    total_loss += loss.item() * input.shape[-1]

                if is_train:
                    model.zero_grad()
                    loss.backward()

                    #######

                    trigger = i < 500 if args.two_stage else False

                    if args.filter == "none":
                        pass
                    elif args.filter == "ma":
                        grads = gradfilter_ma(model, grads=grads, window_size=args.window_size, lamb=args.lamb, trigger=trigger)
                    elif args.filter == "ema":
                        grads = gradfilter_ema(model, grads=grads, alpha=args.alpha, lamb=args.lamb)
                    else:
                        raise ValueError(f"Invalid gradient filter type `{args.filter}`")

                    #######

                    optimizer.step()
                    scheduler.step()
                    i += 1

                acc = (logits[-1].argmax(-1) == input[-1]).float().mean()
                total_acc += acc.item() * input.shape[-1]

            if is_train:
                train_acc.append(total_acc / train_data.shape[-1])
                train_loss.append(total_loss / train_data.shape[-1])
                its.append(i)
            else:
                val_acc.append(total_acc / valid_data.shape[-1])
                val_loss.append(total_loss / valid_data.shape[-1])

        if args.save_weights:
            # do_save = e <= 500 or (e > 500 and (e + 1) % 100 == 0) or e == int(args.budget) // steps_per_epoch - 1
            do_save = ((e + 1) % 500 == 0) or (e == int(args.budget) // steps_per_epoch - 1)
        else:
            do_save = (e + 1) % 100 == 0
        if do_save:
            
            # Accuracy
            steps = torch.arange(len(train_acc)).numpy() * steps_per_epoch
            plt.plot(steps, train_acc, label="train")
            plt.plot(steps, val_acc, label="val")
            plt.legend()
            plt.title("Modular Multiplication (training on 50% of data)")
            plt.xlabel("Optimization Steps")
            plt.ylabel("Accuracy")
            plt.xscale("log", base=10)
            plt.grid()
            plt.savefig(f"results/acc_{args.label}.png", dpi=150)
            plt.close()
            
            # Loss
            plt.plot(steps, train_loss, label="train")
            plt.plot(steps, val_loss, label="val")
            plt.legend()
            plt.title("Modular Multiplication (training on 50% of data)")
            plt.xlabel("Optimization Steps")
            plt.ylabel("Loss")
            plt.xscale("log", base=10)
            plt.grid()
            plt.savefig(f"results/loss_{args.label}.png", dpi=150)
            plt.close()
            
            # Hessian
            plt.plot(hessian_its, [abs(trace) for trace in train_hessiantrace], label="train")
            plt.plot(hessian_its, [abs(trace) for trace in val_hessiantrace], label="val")
            plt.legend()
            plt.title("Modular Multiplication (training on 50% of data)")
            plt.xlabel("Optimization Steps")
            plt.ylabel("Hessian trace")
            plt.xscale("log", base=10)
            plt.yscale("log", base=10)
            plt.grid()
            plt.savefig(f"results/hessiantrace_{args.label}.png", dpi=150)
            plt.close()

            results = {
                'its': its,
                'train_acc': train_acc,
                'train_loss': train_loss,
                'val_acc': val_acc,
                'val_loss': val_loss,
                'hessian_its': hessian_its,
                'train_hessiantrace': train_hessiantrace,
                'val_hessiantrace': val_hessiantrace,
            }

            if args.save_weights:
                net_its.append(e)
                nets.append(copy.deepcopy(model.state_dict()))
                results['net_its'] = net_its
                results['net'] = nets

            torch.save(results, f"results/res_{args.label}.pt")
            
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--task", type=str, default="multiplication")
    parser.add_argument("--filter", type=str, default="none")
    parsed_args = parser.parse_args()
    
    # Instantiate and set values
    args = ExptSettings()
    args.label = args.task = parsed_args.task
    args.filter = parsed_args.filter
    args.save_weights = True
    args.hessian_save_every = 100
    
    # Run experiment
    filter_str = ('_' if args.label != '' else '') + args.filter
    window_size_str = f'_w{args.window_size}'
    alpha_str = f'_a{args.alpha:.3f}'.replace('.', '')
    lamb_str = f'_l{int(args.lamb)}'

    if args.filter == 'none':
        filter_suffix = ''
    elif args.filter == 'ma':
        filter_suffix = window_size_str + lamb_str
    elif args.filter == 'ema':
        filter_suffix = alpha_str + lamb_str
    else:
        raise ValueError(f"Unrecognized filter type {args.filter}")

    optim_suffix = ''
    if args.weight_decay != 0:
        optim_suffix = optim_suffix + f'_wd{args.weight_decay:.1e}'.replace('.', '')
    if args.lr != 1e-3:
        optim_suffix = optim_suffix + f'_lrx{int(args.lr / 1e-3)}'

    args.label = args.label + filter_str + filter_suffix + optim_suffix
    print(f'Experiment results saved under name: {args.label}')

    main(args)