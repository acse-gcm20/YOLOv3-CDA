#! /usr/bin/env python3

from __future__ import division

import os
import tqdm
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader
import torch.optim as optim

from pytorchyolo.models import load_model
from pytorchyolo.utils.logger import Logger
from pytorchyolo.utils.utils import to_cpu, load_classes, print_environment_info, provide_determinism, worker_seed_set
from pytorchyolo.utils.datasets import ListDataset
from pytorchyolo.utils.augmentations import AUGMENTATION_TRANSFORMS
# from pytorchyolo.utils.transforms import DEFAULT_TRANSFORMS
from pytorchyolo.utils.parse_config import parse_data_config
from pytorchyolo.utils.loss import compute_loss
from pytorchyolo.test import _evaluate, _create_validation_data_loader

from terminaltables import AsciiTable

from torchsummary import summary


class Args:
    def __init__(self, model, epochs, seed, pretrained_weights, config):
        self.model = model
        self.data = config
        self.epochs = epochs
        self.verbose = False
        self.n_cpu = 2
        self.pretrained_weights = pretrained_weights
        self.checkpoint_interval = 1
        self.evaluation_interval = 1
        self.multiscale_training = True
        self.iou_thres = 0.5
        self.conf_thres = 0.1
        self.nms_thres = 0.5
        self.logdir = 'logs'
        self.seed = seed

def _create_data_loader(img_path, batch_size, img_size, n_cpu, 
                        multiscale_training=False, shuffle_order=True):
    """Creates a DataLoader for training.

    :param img_path: Path to file containing all paths to training images.
    :type img_path: str
    :param batch_size: Size of each image batch
    :type batch_size: int
    :param img_size: Size of each image dimension for yolo
    :type img_size: int
    :param n_cpu: Number of cpu threads to use during batch generation
    :type n_cpu: int
    :param multiscale_training: Scale images to different sizes randomly
    :type multiscale_training: bool
    :return: Returns DataLoader
    :rtype: DataLoader
    """

    dataset = ListDataset(
        img_path,
        img_size=img_size,
        multiscale=multiscale_training,
        transform=AUGMENTATION_TRANSFORMS)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle_order,
        num_workers=n_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
        worker_init_fn=worker_seed_set)
    return dataloader

def save_losses(model_path, weights, paths):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(model_path, weights)

    dataloader = _create_data_loader(paths, 1, model.hyperparams['height'], 2, shuffle_order=False)

    loss_df = pd.DataFrame(columns=['img', 'box', 'obj', 'cls', 'loss'])

    for i, (fname, imgs, targets) in enumerate(tqdm.tqdm(dataloader)):
        img_name = os.path.basename(fname[0])
        imgs = imgs.to(device, non_blocking=True)
        targets = targets.to(device)
        
        outputs = model(imgs)

        loss, losses = compute_loss(outputs, targets, model)
        iou = float(losses[0])
        obj = float(losses[1])
        cls_loss = float(losses[2])
        total = float(losses[3])
        loss_df = loss_df.append({'img': img_name,
                        'box': iou,
                        'obj': obj,
                        'cls': cls_loss,
                        'loss': total}, ignore_index=True)

    loss_df.to_csv('loss_table.csv')
    print('Loss table saved to loss_table.csv')

def run(model, epochs, config, seed=42, pretrained_weights=None, append_file=None, show_loss=False):
    print("Training\n")
    args = Args(model, epochs, seed, pretrained_weights, config)
    print(f"Parameters: \nEpochs: {args.epochs}, Seed: {args.seed}")

    if args.seed != -1:
        provide_determinism(args.seed)

    # Create output directories if missing
    os.makedirs("output", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # Get data configuration
    data_config = parse_data_config(args.data)
    train_path = data_config["train"]
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ############
    # Create model
    # ############

    model = load_model(args.model, args.pretrained_weights)

    # Print model
    if args.verbose:
        summary(model, input_size=(3, model.hyperparams['height'], model.hyperparams['height']))

    mini_batch_size = model.hyperparams['batch'] // model.hyperparams['subdivisions']

    # #################
    # Create Dataloader
    # #################

    # Load training dataloader
    dataloader = _create_data_loader(
        train_path,
        mini_batch_size,
        model.hyperparams['height'],
        args.n_cpu,
        args.multiscale_training)

    # Load validation dataloader
    validation_dataloader = _create_validation_data_loader(
        valid_path,
        mini_batch_size,
        model.hyperparams['height'],
        args.n_cpu)

    # ################
    # Create optimizer
    # ################

    params = [p for p in model.parameters() if p.requires_grad]

    if (model.hyperparams['optimizer'] in [None, "adam"]):
        optimizer = optim.Adam(
            params,
            lr=model.hyperparams['learning_rate'],
            weight_decay=model.hyperparams['decay'],
        )
    elif (model.hyperparams['optimizer'] == "sgd"):
        optimizer = optim.SGD(
            params,
            lr=model.hyperparams['learning_rate'],
            weight_decay=model.hyperparams['decay'],
            momentum=model.hyperparams['momentum'])
    else:
        print("Unknown optimizer. Please choose between (adam, sgd).")

    trainingLosses = []
    validationLosses = []
    precisionVals = []
    recallVals = []
    mAPs = []

    for epoch in range(args.epochs):

        print("\n## Epoch {} of {} ##\n".format(epoch+1, args.epochs))

        for threshold, value in model.hyperparams['lr_steps']:
            if epoch == threshold:
                print(f'New learning rate: {model.hyperparams["learning_rate"] * value}\n')
        
        model.train()  # Set model to training mode

        for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc=f"Training")):

            batches_done = len(dataloader) * epoch + batch_i

            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device)
            
            outputs = model(imgs)

            loss, loss_components = compute_loss(outputs, targets, model)

            loss.backward()

            ###############
            # Run optimizer
            ###############

            if batches_done % model.hyperparams['subdivisions'] == 0:
                # Adapt learning rate
                # Get learning rate defined in cfg
                lr = model.hyperparams['learning_rate']
                if batches_done < model.hyperparams['burn_in']:
                    # Burn in
                    lr *= (batches_done / model.hyperparams['burn_in'])
                else:
                    # Set and parse the learning rate to the steps defined in the cfg
                    for threshold, value in model.hyperparams['lr_steps']:
                        if epoch >= threshold:
                            lr *= value

                # Set learning rate
                for g in optimizer.param_groups:
                    g['lr'] = lr

                # Run optimizer
                optimizer.step()
                # Reset gradients
                optimizer.zero_grad()

            # ############
            # Log progress
            # ############
            if show_loss and batch_i % (len(dataloader)//4) == 0 and batch_i > 1:
                print('\n'+AsciiTable(
                    [
                        ["Type", "Value"],
                        ["IoU loss", float(loss_components[0])],
                        ["Object loss", float(loss_components[1])],
                        ["Class loss", float(loss_components[2])],
                        ["Loss", float(loss_components[3])],
                        ["Batch loss", to_cpu(loss).item()],
                    ]).table)

                model.seen += imgs.size(0)

        # #############
        # Validation
        # #############

        with torch.no_grad():
            for batch_i, (_, imgs, targets) in enumerate(validation_dataloader):
                batches_done = len(validation_dataloader) * epoch + batch_i

                imgs = imgs.to(device, non_blocking=True)
                targets = targets.to(device)

                outputs = model(imgs)

                val_loss, val_loss_components = compute_loss(outputs, targets, model)

        metrics_output = _evaluate(
            model,
            validation_dataloader,
            class_names,
            img_size=model.hyperparams['height'],
            iou_thres=args.iou_thres,
            conf_thres=args.conf_thres,
            nms_thres=args.nms_thres,
            verbose=args.verbose)

        precision, recall, AP, f1, ap_class = metrics_output
        precisionVals.append(precision.mean())
        recallVals.append(recall.mean())
        mAPs.append(AP.mean())

        print("\nTraining Loss", float(loss_components[3]))
        trainingLosses.append(float(loss_components[3]))

        print("Validation loss:", float(val_loss_components[3]))
        validationLosses.append(float(val_loss_components[3]))

        # Save model to checkpoint file
        if epoch % args.checkpoint_interval == 0:
            checkpoint_path = f"checkpoints/yolov3_ckpt_{epoch+1}.pth"
            print(f"\n---- Saving checkpoint to: '{checkpoint_path}' ----")
            torch.save(model.state_dict(), checkpoint_path)

    if append_file is not None:
        stats_file = append_file
    else:
        stats_file = 'stats.txt'
        
    with open(stats_file, "a") as stats:
        for i in range(epochs):
            stats.write("{} {} {} {} {}\n".format(trainingLosses[i],
                                               validationLosses[i],
                                               precisionVals[i],
                                               recallVals[i],
                                               mAPs[i]))

    print("\nTraining finished. Statisics are saved in:", stats_file)
