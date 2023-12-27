import os
import time
import torch
import numpy as np
from options import *
from model.fwm import FWM

import utils.dataloader as dataloader
import utils.utils as utils
import logging
from collections import defaultdict
from average_meter import AverageMeter
import torchvision

def train(model:FWM,
          device: torch.device,
          fwm_config: FWM_Options,
          train_options: TrainingOptions,
          this_run_folder: str,
          writer):
    """
    Trains the FWM model
    :param model: The model
    :param device: torch.device object, usually this is GPU (if avaliable), otherwise CPU.
    :param fwm_config: The network configuration
    :param train_options: The training settings
    :param this_run_folder: The parent folder for the current training run to store training artifacts/results/logs.
    :param tb_logger: TensorBoardLogger object which is a thin wrapper for TensorboardX logger.
                Pass None to disable TensorboardX logging
    :return:
    """
    bg_loader = dataloader.dataloader_background(train_options.background_folder,1)# default batchsize of  background 1
    train_loader = dataloader.dataloader_img_mask(train_options.train_folder,train_options.batch_size)
    l = len(train_loader)
    rate = 0.05
    val_l = int(l*rate)
    train_l = l - val_l
    # val_laoder = dataloader.dataloader_img_mask(train_options.validation_folder,train_options.batch_size)
    file_count = len(train_loader.dataset)
    if file_count % train_options.batch_size == 0:
        steps_in_epoch = file_count // train_options.batch_size
    else:
        steps_in_epoch = file_count // train_options.batch_size + 1


    print_each = 10

    for epoch in range(train_options.start_epoch,train_options.number_of_epochs+1):
        logging.info('\nStarting epoch {}/{}'.format(epoch, train_options.number_of_epochs))
        logging.info('Batch size = {}\nSteps in epoch = {}'.format(train_options.batch_size, steps_in_epoch))
        training_losses = defaultdict(AverageMeter)
        epoch_start = time.time()
        step = 1
        for i in range(train_l):
            images,masks = train_loader.__iter__().__next__()
            images = images.to(device)
            masks = masks.to(device)
            masks[masks>0]=1
            masks[masks<1]=0
            messages = torch.Tensor(np.random.choice([0, 1], (images.shape[0], fwm_config.message_length))).to(device)
            background = bg_loader.__iter__().__next__()
            background = background.to(device)
            losses , (merge_img , pred_mask ,final_mask) = model.train_on_batch([background,images,masks,messages],epoch)
            # if not torch.is_tensor(merge_img):
            #     continue
            for name , loss in losses.items():
                training_losses[name].update(loss)
            if (step % print_each == 0 or step == steps_in_epoch) and writer is not None:
                logging.info(
                    'Epoch: {}/{} Step: {}/{}'.format(epoch, train_options.number_of_epochs, step, steps_in_epoch))
                utils.log_progress(training_losses)
                logging.info('-' * 40)
                writer.add_scalar("train loss",losses['loss           '],epoch*(len(train_loader))+step)
            
                
            
            step += 1
        model.lr_scheduler.step()
        # model.lr_scheduler_enc.step()
        train_duration = time.time() - epoch_start
        logging.info('Epoch {} training duration {:.2f} sec'.format(epoch, train_duration))
        logging.info('-' * 40)
        utils.write_losses(os.path.join(this_run_folder, 'train.csv'), training_losses, epoch, train_duration)

        validation_losses = defaultdict(AverageMeter)
        logging.info('Running validation for epoch {}/{}'.format(epoch, train_options.number_of_epochs))
        step = 1
        for i in range(val_l):
            images,masks = train_loader.__iter__().__next__()
            images = images.to(device)
            masks = masks.to(device)
            masks[masks>0]=1
            masks[masks<1]=0
            if writer is not None:
                grid = torchvision.utils.make_grid(images)
                writer.add_image("batch of image",grid)
                grid = torchvision.utils.make_grid(masks)
                writer.add_image("masks",grid)
            messages = torch.Tensor(np.random.choice([0, 1], (images.shape[0], fwm_config.message_length))).to(device)
            background = bg_loader.__iter__().__next__()
            background = background.to(device)

            losses , (merge_img , pred_mask ,final_mask) = model.validation_on_batch([background,images,masks,messages])
            # if not torch.is_tensor(merge_img):
            #     continue
            for name , loss in losses.items():
                validation_losses[name].update(loss)
            if (step % print_each == 0 or step == steps_in_epoch) and writer is not None:
                writer.add_scalar("validation loss",losses['dec_mse        '],epoch*(len(train_loader))+step)
            if step%100==0 and writer is not None:
                grid = torchvision.utils.make_grid(merge_img)
                writer.add_image("merge_img",grid)
                tobe_save = torch.cat([pred_mask,final_mask],dim=0)
                grid = torchvision.utils.make_grid(tobe_save)
                writer.add_image("pred_mask",grid)
            step += 1
        utils.log_progress(validation_losses)
        logging.info('-' * 40)
        utils.save_checkpoint(model, train_options.experiment_name, epoch, os.path.join(this_run_folder, 'checkpoints'))
        utils.write_losses(os.path.join(this_run_folder, 'validation.csv'), validation_losses, epoch,
                           time.time() - epoch_start)


def train_loc(model:FWM,
          device: torch.device,
          fwm_config: FWM_Options,
          train_options: TrainingOptions,
          this_run_folder: str,
          writer):
    """
    Trains the FWM model
    :param model: The model
    :param device: torch.device object, usually this is GPU (if avaliable), otherwise CPU.
    :param fwm_config: The network configuration
    :param train_options: The training settings
    :param this_run_folder: The parent folder for the current training run to store training artifacts/results/logs.
    :param tb_logger: TensorBoardLogger object which is a thin wrapper for TensorboardX logger.
                Pass None to disable TensorboardX logging
    :return:
    """
    bg_loader = dataloader.dataloader_background(train_options.background_folder,1)# default batchsize of  background 1
    train_loader = dataloader.dataloader_img_mask(train_options.train_folder,train_options.batch_size)
    l = len(train_loader)
    rate = 0.1
    val_l = int(l*rate)
    train_l = l - val_l
    # val_laoder = dataloader.dataloader_img_mask(train_options.validation_folder,train_options.batch_size)
    file_count = len(train_loader.dataset)
    if file_count % train_options.batch_size == 0:
        steps_in_epoch = file_count // train_options.batch_size
    else:
        steps_in_epoch = file_count // train_options.batch_size + 1


    print_each = 10

    for epoch in range(train_options.start_epoch,train_options.number_of_epochs+1):
        logging.info('\nStarting epoch {}/{}'.format(epoch, train_options.number_of_epochs))
        logging.info('Batch size = {}\nSteps in epoch = {}'.format(train_options.batch_size, steps_in_epoch))
        training_losses = defaultdict(AverageMeter)
        epoch_start = time.time()
        step = 1
        for i in range(train_l):
            images,masks = train_loader.__iter__().__next__()
            images = images.to(device)
            masks = masks.to(device)
            
            messages = torch.Tensor(np.random.choice([0, 1], (images.shape[0], fwm_config.message_length))).to(device)
            background = bg_loader.__iter__().__next__()
            background = background.to(device)
            losses , (merge_img) = model.train_on_batch_loc([background,images,masks,messages],step)

            for name , loss in losses.items():
                training_losses[name].update(loss)
            if (step % print_each == 0 or step == steps_in_epoch) and writer is not None:
                logging.info(
                    'Epoch: {}/{} Step: {}/{}'.format(epoch, train_options.number_of_epochs, step, steps_in_epoch))
                utils.log_progress(training_losses)
                logging.info('-' * 40)
                writer.add_scalar("train loss",losses['loss           '],epoch*(len(train_loader))+step)
            
                
            step += 1
        
        train_duration = time.time() - epoch_start
        logging.info('Epoch {} training duration {:.2f} sec'.format(epoch, train_duration))
        logging.info('-' * 40)
        utils.write_losses(os.path.join(this_run_folder, 'train.csv'), training_losses, epoch, train_duration)

        validation_losses = defaultdict(AverageMeter)
        logging.info('Running validation for epoch {}/{}'.format(epoch, train_options.number_of_epochs))
        step = 1
        for j in range(val_l):
            images,masks = train_loader.__iter__().__next__()
            images = images.to(device)
            masks = masks.to(device)
            # grid = torchvision.utils.make_grid(images)
            # writer.add_image("batch of image",grid)
            # grid = torchvision.utils.make_grid(masks)
            # writer.add_image("masks",grid)
            messages = torch.Tensor(np.random.choice([0, 1], (images.shape[0], fwm_config.message_length))).to(device)
            background = bg_loader.__iter__().__next__()
            background = background.to(device)

            losses , (merge_img) = model.validation_on_batch_loc([background,images,masks,messages])
            for name , loss in losses.items():
                validation_losses[name].update(loss)
            if (step % print_each == 0 or step == steps_in_epoch) and writer is not None:
                writer.add_scalar("validation loss",losses['loss           '],epoch*(len(train_loader))+step)
            # if step%100==0:
            #     grid = torchvision.utils.make_grid(merge_img)
            #     writer.add_image("merge_img",grid)
            #     tobe_save = torch.cat([pred_mask,final_mask],dim=0)
            #     grid = torchvision.utils.make_grid(tobe_save)
            #     writer.add_image("pred_mask",grid)
            step += 1
        utils.log_progress(validation_losses)
        logging.info('-' * 40)
        utils.save_checkpoint(model, train_options.experiment_name, epoch, os.path.join(this_run_folder, 'checkpoints'))
        utils.write_losses(os.path.join(this_run_folder, 'validation.csv'), validation_losses, epoch,
                           time.time() - epoch_start)

