import os
import csv
import time
import torch
import logging
import torchvision

from model.fwm import FWM


def save_img(origin_img,filename):
    img = origin_img[:,:,:,:].cpu()
    img = (img + 1) / 2
    torchvision.utils.save_image(img,filename,normalize=False)

def create_folder_for_run(runs_folder, experiment_name):
    if not os.path.exists(runs_folder):
        os.makedirs(runs_folder)

    this_run_folder = os.path.join(runs_folder, f'{experiment_name} {time.strftime("%Y.%m.%d--%H-%M-%S")}')

    os.makedirs(this_run_folder)
    os.makedirs(os.path.join(this_run_folder, 'checkpoints'))
    os.makedirs(os.path.join(this_run_folder, 'images'))
    os.makedirs(os.path.join(this_run_folder, 'masks'))

    return this_run_folder

def save_checkpoint(model: FWM, experiment_name: str, epoch: int, checkpoint_folder: str):
    """ Saves a checkpoint at the end of an epoch. """
    if not os.path.exists(checkpoint_folder):
        os.makedirs(checkpoint_folder)

    checkpoint_filename = f'{experiment_name}--epoch-{epoch}.pyt'
    checkpoint_filename = os.path.join(checkpoint_folder, checkpoint_filename)
    logging.info('Saving checkpoint to {}'.format(checkpoint_filename))
    checkpoint = {
        'enc-model': model.enc_dec.encoder.state_dict(),
        'dec-model': model.enc_dec.decoder.state_dict(),
        'enc-dec-optim': model.optimizer_enc_dec.state_dict(),
        'loc-model': model.localizer.state_dict(),
        'loc-optim': model.optimizer_loc.state_dict(),
        'discrim-model': model.discriminator.state_dict(),
        'discrim-optim': model.optimizer_dis.state_dict(),
        'epoch': epoch
    }
    torch.save(checkpoint, checkpoint_filename)
    logging.info('Saving checkpoint done.')

def load_from_checkpoint(net:FWM,checkpoint):
    net.enc_dec.encoder.load_state_dict(checkpoint['enc-model'],strict=False)
    net.enc_dec.decoder.load_state_dict(checkpoint['dec-model'],strict=False)
    net.optimizer_enc_dec.load_state_dict(checkpoint['enc-dec-optim'],strict=False)
    net.discriminator.load_state_dict(checkpoint['discrim-model'],strict=False)
    net.optimizer_dis.load_state_dict(checkpoint['discrim-optim'])
    net.localizer.load_state_dict(checkpoint['loc-model'],strict=False)
    net.optimizer_loc.load_state_dict(checkpoint['loc-optim'])

def load_loc_from_checkpoint(net,checkpoint):
    net.localizer.load_state_dict(checkpoint['loc-model'],strict=False)
    net.optimizer_loc.load_state_dict(checkpoint['loc-optim'])

def log_print_helper(losses_accu, log_or_print_func):
    max_len = max([len(loss_name) for loss_name in losses_accu])
    for loss_name, loss_value in losses_accu.items():
        log_or_print_func(loss_name.ljust(max_len + 4) + '{:.4f}'.format(loss_value.avg))

def log_progress(losses_accu):
    log_print_helper(losses_accu, logging.info)

def write_losses(file_name, losses_accu, epoch, duration):
    with open(file_name, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if epoch == 1:
            row_to_write = ['epoch'] + [loss_name.strip() for loss_name in losses_accu.keys()] + ['duration']
            writer.writerow(row_to_write)
        row_to_write = [epoch] + ['{:.4f}'.format(loss_avg.avg) for loss_avg in losses_accu.values()] + [
            '{:.0f}'.format(duration)]
        writer.writerow(row_to_write)