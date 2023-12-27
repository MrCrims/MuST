import os
import argparse
import torch
import pickle
import sys
import logging
import pprint
import numpy as np
from train import train,train_loc
from options import TrainingOptions , FWM_Options
from model.fwm import FWM
import utils.utils as utils
import random

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)

def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    parent_parser = argparse.ArgumentParser(description='Training of FWM nets')
    subparsers = parent_parser.add_subparsers(dest='command', help='Sub-parser for commands')
    new_run_parser = subparsers.add_parser('new', help='starts a new run')
    new_run_parser.add_argument('--data-dir', '-d', required=True, type=str,
                                help='The directory where the data is stored.')
    new_run_parser.add_argument('--batch-size-background', '-bg', default=1, type=int, help='The batch size.')
    new_run_parser.add_argument('--batch-size-image', '-im', required=True, type=int, help='The batch size.')
    new_run_parser.add_argument('--resize-bound', '-r', required=True, type=str, help='The bound of image resize.')
    new_run_parser.add_argument('--epochs', '-e', default=500, type=int, help='Number of epochs to run the simulation.')
    new_run_parser.add_argument('--name', required=True, type=str, help='The name of the experiment.')
    new_run_parser.add_argument('--tensorboard', action='store_true',
                                help='Use to switch on Tensorboard logging.')
    new_run_parser.add_argument('--train_loc', action='store_true',
                                help='Use to switch on localization training.')
    new_run_parser.add_argument('--train_enc', action='store_true',
                                help='Use to switch on encoder training.')
    new_run_parser.add_argument('--train_dec', action='store_true',
                                help='Use to switch on decoder training.')
    new_run_parser.add_argument('--message', '-m', default=30, type=int, help='The length in bits of the watermark.')
    new_run_parser.add_argument('--enable-fp16', dest='enable_fp16', action='store_true',
                                help='Enable mixed-precision training.')
    new_run_parser.add_argument('--cotinue', '-c',  action='store_true', help='Continue Learning')

    args = parent_parser.parse_args()
    checkpoint = None
    loaded_checkpoint_file_name = None

    assert args.command == 'new'
    start_epoch = 1
    train_options = TrainingOptions(
            batch_size=args.batch_size_image,
            number_of_epochs=args.epochs,
            background_folder=args.data_dir,
            train_folder=os.path.join(args.data_dir, 'train'),
            validation_folder=os.path.join(args.data_dir, 'val'),
            runs_folder=os.path.join('.', 'runs'),
            start_epoch=start_epoch,
            experiment_name=args.name)
    resize_bound = args.resize_bound.split(" ")
    resize_bound = [float(resize_bound[0]),float(resize_bound[-1])]
   

    FWM_config = FWM_Options(H=640,W=640,
                            message_length=args.message,
                            encoder_blocks=8, encoder_channels=64,
                            decoder_blocks=9, decoder_channels=64,
                            use_discriminator=True,
                            discriminator_blocks=4, discriminator_channels=64,
                            decoder_loss=2,
                            encoder_loss=0.7,
                            adversarial_loss=1e-3,
                            localization_loss=1e-2,
                            enable_fp16=args.enable_fp16,
                            resize_bound=resize_bound,
                            batchsize= args.batch_size_image   )
    
    this_run_folder = utils.create_folder_for_run(train_options.runs_folder, args.name)
    writer = None
    if args.tensorboard:
        from  tensorboardX import SummaryWriter
        logging.info('Tensorboard enabled')
        writer = SummaryWriter()
    with open(os.path.join(this_run_folder, 'options-and-config.pickle'), 'wb+') as f:
            pickle.dump(train_options, f)
            pickle.dump(FWM_config, f)
    logging.basicConfig(level=logging.INFO,
                        format='%(message)s',
                        handlers=[
                            logging.FileHandler(os.path.join(this_run_folder, '{}.log'.format(train_options.experiment_name))),
                            logging.StreamHandler(sys.stdout)
                        ])
    if args.cotinue:
        with open("","rb") as f:
            train_options = pickle.load(f)
            fwm_options = pickle.load(f)
        model = FWM(fwm_options , device ,this_run_folder,writer)
    else:
        model = FWM(FWM_config , device ,this_run_folder,writer)

    logging.info('FWM model: {}\n'.format(model.to_stirng()))
    logging.info('Model Configuration:\n')
    logging.info(pprint.pformat(vars(FWM_config)))
    logging.info('\nTraining train_options:\n')
    logging.info(pprint.pformat(vars(train_options)))

    if args.train_loc:
        train_loc(model,device,FWM_config,train_options,this_run_folder,writer)
    else:
        if args.cotinue:
            checkpoint = torch.load("")
            utils.load_loc_from_checkpoint(model,checkpoint)
            train(model , device , fwm_options , train_options , this_run_folder , writer)

        else:
            checkpoint = torch.load("")
            utils.load_loc_from_checkpoint(model,checkpoint)
            train(model , device , FWM_config , train_options , this_run_folder , writer)


if __name__ == '__main__':
    setup_seed(42)
    main()
