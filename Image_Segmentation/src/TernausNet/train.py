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


def train(
        net, 
        epochs=1,
        batch_size=1,
        lr=1, 
        val_percent=0.1,
        save_cp=True, 
        img_scale=1.0):
    """
    This function to train UNet 
    Parameters:
        - net: 
        - epochs
    """

    ### loader datasets
    train_loader, val_loader, n_train, n_val = dataloader(configs_param.img_path, configs_param.mask_path)
    ###
    writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}_SCALE_{img_scale}')
    global_step = 0

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images scaling:  {img_scale}
    ''')
    ### 
    optimizer = optim.RMSprop(
            net.parameters(), 
            lr=lr, 
            weight_decay=1e-8, 
            momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            'min' if net.n_classes > 1 else 'max', 
            patience=2)

    if net.n_classes > 1:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        net.train()

        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs = batch['image']
                true_masks = batch['mask']
                assert imgs.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'
                
                imgs = imgs.to(device=device, dtype=torch.float32)
                mask_type = torch.float32 if net.n_classes == 1 else torch.long
                true_masks = true_masks.to(device=device, dtype=mask_type)

                masks_pred = net(imgs)
                loss = criterion(masks_pred, true_masks)
                epoch_loss += loss.item()
                writer.add_scalar('Loss/train', loss.item(), global_step)

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                pbar.update(imgs.shape[0])
                global_step += 1
                if global_step % (n_train // (10 * batch_size)) == 0:
                    for tag, value in net.named_parameters():
                        tag = tag.replace('.', '/')
                        writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                        writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)
                    # val_score = eval_net(net, val_loader, device)
                    # scheduler.step(val_score)
                    # writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)

                    # if net.n_classes > 1:
                    #     logging.info('Validation cross entropy: {}'.format(val_score))
                    #     writer.add_scalar('Loss/test', val_score, global_step)
                    # else:
                    #     logging.info('Validation Dice Coeff: {}'.format(val_score))
                    #     writer.add_scalar('Dice/test', val_score, global_step)

                    writer.add_images('images', imgs, global_step)
                    if net.n_classes == 1:
                        writer.add_images('masks/true', true_masks, global_step)
                        writer.add_images('masks/pred', torch.sigmoid(masks_pred) > 0.5, global_step)

        if save_cp:
            try:
                os.mkdir(configs_param.dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(net.state_dict(),
                       configs_param.dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
            logging.info(f'Checkpoint {epoch + 1} saved !')

    writer.close()


if __name__ == '__main__':
    args = configs_param.get_args()
    # device = torch.device('cuda' if torch.cuda.is_available() )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using deveice {device}')
    
    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    #   - For 1 class and background, use n_classes=1
    #   - For 2 classes, use n_classes=1
    #   - For N > 2 classes, use n_classes=N
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