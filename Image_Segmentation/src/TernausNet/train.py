#
from tqdm import tqdm
import logging
import os
import sys
#
from torch import optim
import torch.nn as nn 
import torch
from torch.utils.tensorboard import SummaryWriter
# 
from models.unet import UNet
import configs_param
from datasets import dataloader



if __name__ == '__main__':
    args = configs_param.get_args()
    # device = torch.device('cuda' if torch.cuda.is_available() )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using deveice {device}')

    """
    Change here to adapt to your data
    n_channels=3 for RGB images
    n_classes is the number of probabilities you want to get per pixel
      - For 1 class and background, use n_classes=1
      - For 2 classes, use n_classes=1
      - For N > 2 classes, use n_classes=N
    """
    net = UNet(n_channels=3, n_classes=1, bilinear=True)
    print("11111")
    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)
    # faster convolutions, but more memory
    # cudnn.benchmark = True

    try:
        train(
            net=net,
            epochs=args.epochs,
            batch_size=args.batchsize,
            lr=args.lr,
            save_cp= True,
            val_percent=args.val / 100,
            img_scale=args.scale)

    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)