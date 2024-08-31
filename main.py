import math
from argparse import ArgumentParser
from itertools import permutations
import copy

import matplotlib.pyplot as plt
from tqdm import tqdm
from time import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# Pytorch's efficient attention doesn't allow Hessian computation by default
from torch.nn.attention import SDPBackend, sdpa_kernel

from pyhessian import hessian

from grokfast import *

# Config
from src.config import ExptSettings
# Data
from src.data.modp import mod_p_data, split_data
# Model
from src.models.transformer import Decoder
# Optimizers: NSM, SAM
from src.optimizers.nsm import NSM
from src.optimizers.sam import SAM
# Margins
from src.util import compute_margin


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
    train_data, valid_data = split_data(data, split_ratio=args.split_ratio)

    # For most experiments we used AdamW optimizer with learning rate 10−3,
    # weight decay 1, β1 = 0.9, β2 = 0.98
    if args.nsm:
        # NSM optimizer
        print(f"Using NSM optimizer with sigma={args.nsm_sigma}, distribution={args.nsm_distribution}")
        base_optimizer = getattr(torch.optim, args.optimizer)
        optimizer = NSM(
            model.parameters(),
            base_optimizer,
            sigma=args.nsm_sigma,
            distribution=args.nsm_distribution,
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(args.beta1, args.beta2),
        )
        #  linear learning rate warmup over the first 10 updates
        # scheduler = torch.optim.lr_scheduler.LambdaLR(
        #     optimizer.base_optimizer, lambda update: 1 if update > 10 else update / 10
        # )
    elif args.sam:
        # SAM optimizer
        print(f"Using SAM optimizer")
        base_optimizer = getattr(torch.optim, args.optimizer)
        optimizer = SAM(
            model.parameters(),
            base_optimizer,
            rho=args.sam_rho,
            lr=args.lr,
            weight_decay=args.weight_decay,
            betas=(args.beta1, args.beta2),
        )
        #  linear learning rate warmup over the first 10 updates
        # scheduler = torch.optim.lr_scheduler.LambdaLR(
        #     optimizer.base_optimizer, lambda update: 1 if update > 10 else update / 10
        # )
    else:
        # AdamW optimizer
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
    
    # Loss
    criterion = F.cross_entropy
    # criterion = nn.CrossEntropyLoss()

    steps_per_epoch = math.ceil(train_data.shape[1] / args.batch_size)

    its, train_acc, val_acc, train_loss, val_loss = [], [], [], [], []
    hessian_its, train_hessiantrace, val_hessiantrace = [], [], []
    train_margin, val_margin = [], []
    grads = None
    i = 0
    
    # Compute/save hessian condition
    if args.hessian_save_every == 0:
        print("Not saving Hessian")
    def save_hessian(e):
        if args.hessian_save_every == 0:
            return False
        return (e < 20 or (e+1) % args.hessian_save_every == 0)
    hessian_loss_func = lambda output, target: F.cross_entropy(output[-1], target)

    # For logging network weights.
    net_its, nets = [], []

    print(f"Epochs: {int(args.budget) // steps_per_epoch}")
    pbar = tqdm(range(int(args.budget) // steps_per_epoch))
    for e in pbar:
        if save_hessian(e):
            with torch.set_grad_enabled(True), sdpa_kernel(SDPBackend.MATH):
            # with torch.set_grad_enabled(True):
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
            total_margin = 0
            total_num_correct_margin = 0

            # torch.split faster than dataloader with tensor
            dl = torch.split(data, args.batch_size, dim=1)
            for input in dl:
                input = input.to(device)
                # print(f"Input shape: {input.shape}")

                with torch.set_grad_enabled(is_train):
                    logits = model(input[:-1])
                    # calculate loss only on the answer part of the equation (last element)
                    loss = criterion(logits[-1], input[-1])

                    # Explicit Hessian regularization
                    if args.explicit_hessian_regularization > 0 and is_train:

                        with torch.backends.cuda.sdp_kernel(enable_math=True, enable_flash=False, enable_mem_efficient=False):
                            # Compute the first-order gradients of the loss with respect to the parameters
                            grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)

                            # Compute Hessian trace
                            hessian_trace = 0.0
                            for grad, param in zip(grads, model.parameters()):
                                grad_squared = grad ** 2
                                hessian_diag_elements = torch.autograd.grad(grad_squared.sum(), param, retain_graph=True)[0]
                                hessian_trace += hessian_diag_elements.sum()

                            # Add the regularization term to the loss
                            loss += 0.5 * args.explicit_hessian_regularization * hessian_trace

                    total_loss += loss.item() * input.shape[-1]

                if is_train:
                    if not(args.nsm or args.sam):
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

                    # NSM
                    elif args.nsm:
                        # TODO hardcoded params
                        nsm_lam = 0
                        nsm_num_perturbs = 1
                        nsm_use_neg = True

                        # 1st forward-backward step: compute the gradients on the original weight
                        logits = model(input[:-1])
                        loss = criterion(logits[-1], input[-1])
                        loss.backward()
                        optimizer.store_gradients(zero_grad=True, store_weights=True, update_weight=nsm_lam)
                        # 2nd forward-backward step: taking perturbations and computing gradients
                        update_weight = (1-nsm_lam) / (2*nsm_num_perturbs) if nsm_use_neg else (1-nsm_lam) / nsm_num_perturbs
                        for _ in range(nsm_num_perturbs):
                            optimizer.first_step(zero_grad=True, store_perturb=True)
                            # Model
                            logits = model(input[:-1])
                            criterion(logits[-1], input[-1]).backward()
                            optimizer.store_gradients(zero_grad=True, store_weights=False, update_weight=update_weight)
                            if nsm_use_neg:
                                optimizer.first_step(zero_grad=True, store_perturb=False)
                                # Model
                                logits = model(input[:-1])
                                criterion(logits[-1], input[-1]).backward()
                                optimizer.store_gradients(zero_grad=True, store_weights=False, update_weight=update_weight)
                        optimizer.second_step(zero_grad=True)

                    elif args.sam:
                        # SAM
                        logits = model(input[:-1])
                        loss = criterion(logits[-1], input[-1])
                        loss.backward()
                        optimizer.first_step(zero_grad=True)

                        logits = model(input[:-1])
                        criterion(logits[-1], input[-1]).backward()
                        optimizer.second_step(zero_grad=True)

                        # Closure implementation of SAM
                        # def closure():
                        #     logits = model(input[:-1])
                        #     loss = criterion(logits[-1], input[-1])
                        #     loss.backward()
                        #     return loss
                        # loss = criterion(logits[-1], input[-1])
                        # loss.backward()
                        # optimizer.step(closure)
                        # optimizer.zero_grad()

                    i += 1

                # Compute accuracy
                acc = (logits[-1].argmax(-1) == input[-1]).float().mean()
                total_acc += acc.item() * input.shape[-1]

                # Compute margin
                # logits: (seq_len, batch_size, num_tokens)
                # input[-1]: (batch_size)
                # output: (batch_size)
                margin, num_correct_margin = compute_margin(logits[-1], input[-1])
                total_margin += margin * num_correct_margin
                total_num_correct_margin += num_correct_margin

            if is_train:
                train_acc.append(total_acc / train_data.shape[-1])
                train_loss.append(total_loss / train_data.shape[-1])
                train_margin.append(total_margin / total_num_correct_margin)
                its.append(i)
            else:
                val_acc.append(total_acc / valid_data.shape[-1])
                val_loss.append(total_loss / valid_data.shape[-1])
                val_margin.append(total_margin / total_num_correct_margin)

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
            plt.title(f"{args.task} (train split = {args.split_ratio})")
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
            plt.title(f"{args.task} (train split = {args.split_ratio})")
            plt.xlabel("Optimization Steps")
            plt.ylabel("Loss")
            plt.xscale("log", base=10)
            plt.grid()
            plt.savefig(f"results/loss_{args.label}.png", dpi=150)
            plt.close()

            # Log loss
            plt.plot(steps, train_loss, label="train")
            plt.plot(steps, val_loss, label="val")
            plt.legend()
            plt.title(f"{args.task} (train split = {args.split_ratio})")
            plt.xlabel("Optimization Steps")
            plt.ylabel("Log Loss")
            plt.xscale("log", base=10)
            plt.yscale("log", base=10)
            plt.grid()
            plt.savefig(f"results/logloss_{args.label}.png", dpi=150)
            plt.close()

            if args.hessian_save_every > 0:
                # Hessian
                plt.plot(hessian_its, [abs(trace) for trace in train_hessiantrace], label="train")
                plt.plot(hessian_its, [abs(trace) for trace in val_hessiantrace], label="val")
                plt.legend()
                plt.title(f"{args.task} (train split = {args.split_ratio})")
                plt.xlabel("Optimization Steps")
                plt.ylabel("Hessian trace")
                plt.xscale("log", base=10)
                plt.yscale("log", base=10)
                plt.ylim(1e-7, 1e7)
                plt.grid()
                plt.savefig(f"results/hessiantrace_{args.label}.png", dpi=150)
                plt.close()

            # Margin
            plt.plot(its, train_margin, label="train")
            plt.plot(its, val_margin, label="val")
            plt.legend()
            plt.title(f"{args.task} (train split = {args.split_ratio})")
            plt.xlabel("Optimization Steps")
            plt.ylabel("Margin")
            plt.xscale("log", base=10)
            plt.grid()
            plt.savefig(f"results/margin_{args.label}.png", dpi=150)
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
    parser.add_argument("--seed", type=int, default=0)
    # Data
    parser.add_argument("--task", type=str, default="multiplication")
    parser.add_argument("--p", type=int, default=97)
    parser.add_argument("--split_ratio", type=float, default=0.5)
    # Optimizer
    parser.add_argument("--budget", type=int, default=300000)
    parser.add_argument("--filter", type=str, default="none")
    parser.add_argument("--wd", type=float, default=0)
    parser.add_argument("--optimizer", type=str, default="AdamW")
    # Hessian regularization
    parser.add_argument("--nsm", action="store_true")
    parser.add_argument("--nsm_sigma", type=float, default=0.01)
    parser.add_argument("--sam", action="store_true")
    parser.add_argument("--no_hessian", action="store_true")
    parser.add_argument("--explicit_hessian_regularization", type=float, default=0)
    parsed_args = parser.parse_args()
    
    # Instantiate and set values
    args = ExptSettings()
    # Seed
    args.seed = parsed_args.seed
    # Data
    args.label = args.task = parsed_args.task
    args.p = parsed_args.p
    args.split_ratio = parsed_args.split_ratio
    # Optimizer
    args.budget = parsed_args.budget
    args.filter = parsed_args.filter
    args.save_weights = True
    args.weight_decay = parsed_args.wd
    args.optimizer = parsed_args.optimizer
    # Hessian
    args.hessian_save_every = 0 if parsed_args.no_hessian else 100
    args.explicit_hessian_regularization = parsed_args.explicit_hessian_regularization
    # NSM
    args.nsm = parsed_args.nsm
    args.nsm_sigma = parsed_args.nsm_sigma
    # SAM
    args.sam = parsed_args.sam
    args.sam_rho = 0.05
    
    # Run experiment
    filter_str = ('_' if args.label != '' else '') + args.filter
    window_size_str = f'_w{args.window_size}'
    alpha_str = f'_a{args.alpha:.3f}'.replace('.', '')
    lamb_str = f'_l{int(args.lamb)}'

    # Data split
    split_str = f'_split{int(args.split_ratio * 100)}'

    if args.filter == 'none':
        filter_suffix = ''
    elif args.filter == 'ma':
        filter_suffix = window_size_str + lamb_str
    elif args.filter == 'ema':
        filter_suffix = alpha_str + lamb_str
    else:
        raise ValueError(f"Unrecognized filter type {args.filter}")

    optim_suffix = ''
    if args.nsm:
        optim_suffix = optim_suffix + f'_nsm{args.nsm_sigma:.1e}'.replace('.', '')
    elif args.sam:
        optim_suffix = optim_suffix + '_sam'
    if args.weight_decay != 0:
        optim_suffix = optim_suffix + f'_wd{args.weight_decay:.1e}'.replace('.', '')
    if args.lr != 1e-3:
        optim_suffix = optim_suffix + f'_lrx{int(args.lr / 1e-3)}'
    if args.explicit_hessian_regularization != 0:
        optim_suffix = optim_suffix + f'_her{args.explicit_hessian_regularization:.1e}'.replace('.', '')

    seed_suffix = f'_seed{args.seed}'

    args.label = args.label + split_str + filter_str + filter_suffix + optim_suffix + seed_suffix
    print(f'Experiment results saved under name: {args.label}')

    main(args)
