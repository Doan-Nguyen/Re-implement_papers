#
from tqdm import tqdm
import logging
import sys
#
from torch import optim
import torch.nn as nn 
import torch
# 
from networks import unet_vgg
import configs, utils_ai
import datasets


if __name__ == '__main__':
    args = configs.get_args()
    # device = torch.device('cuda' if torch.cuda.is_available() )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using deveice {device}')
    ###
    net = unet_vgg.UNet_VGG11()
    device_ids = list(map(int, args.device_ids.split(',')))
    model = nn.DataParallel(model, device_ids=device_ids).cuda()
    loss = Loss()

    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)
    # faster convolutions, but more memory
    # cudnn.benchmark = True

    try:
        utils_ai.train(
            init_optimizer=lambda lr: Adam(model.parameters(), lr=lr),
            args=args,
            model=net,
            criterion=loss,
            train_loader=train_loader,
            valid_loader=valid_loader,
            validation=validation,
            fold=args.fold
        )   
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)