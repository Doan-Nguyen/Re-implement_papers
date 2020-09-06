#
from tqdm import tqdm
import logging
import sys
import random
#
from torch import optim
import torch.nn as nn 
import torch
# 
from networks import unet_vgg
import configs
import datasets
import utils

###         Logging 
logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
logging.info('Start training process')
handler = logging.FileHandler('train_log.log')
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


def train(
    args, 
    model: nn.Module, 
    criterion, 
    *, 
    train_loader, 
    valid_loader,
    validation, 
    init_optimizer,
    fold=None,
    save_predictions=None,
    n_epochs=configs.EPOCHS
    ):
    ##          Log 

    ##          Checkpoint path
    checkpoint_path = configs.CHECKPOINT_PATH
    model_checkpoint = checkpoint_path / 'ternaus_{fold}.pt'.format(fold=fold)
    best_model_checkpoint = checkpoint_path /  'ternaus_best_{fold}.pt'.format(fold=fold)
    
    ##          Start training
    if model_checkpoint.exists():
        state = torch.load(str(model_checkpoint))
        epoch = state['epoch']
        step = state['step']
        best_valid_loss = state['best_valid_loss']
        model.load_state_dict(state['model'])
        print("Restored model, epoch {}, step {:,}".format(epoch, step))
    else:
        epoch = 1
        step = 0
        best_valid_loss = float('inf')
    
    ##          Save checkpoint
    save = lambda ep: torch.save({
        'model': model.state_dict(),
        'epoch': ep, 
        'step': step, 
        'best_valid_loss': best_valid_loss
    }, str(model_checkpoint))

    ##      Initializer epoch
    report_each = 10
    save_prediction_each = report_each*20
    valid_losses = []
    
    ###         Training
    for epoch in range(epoch, n_epochs + 1):
        lr = utils.cyclic_lr(epoch)
        optimizer = init_optimizer(lr)

        model.train()
        random.seed()
        tq = tqdm(total=(len(train_loader)*configs.BATCH_SIZE))
        tq.set_description("Epoch {}, lr: {}".format(epoch, lr))
        
        losses = []
        # tl = train_loader
        # if args.epoch_size:
        #     tl = islice(tl, args.epoch_size // args.batch_size)
        
        mean_loss = 0
        for i, (inputs, targets) in enumerate(train_loader):
            ##
            inputs, targets = utils.variable(inputs), utils.variable(targets)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            ##
            optimizer.zero_grad()
            batch_size = inputs.size(0)
            step += 1
            tq.update(batch_size)
            losses.append(loss.data[0])
            mean_loss = np.mean(losses[-report_each:])
            tq.set_postfix(loss='{:.5f}'.format(mean_loss))

            (batch_size*loss).backward()
            optimizer.step()

            if i and i % report_each == 0:
                utils.write_event(log, step, loss=mean_loss)
                if save_predictions and i % save_prediction_each == 0:
                    p_i = (i//save_prediction_each) % 5
                    save_predictions(root, p_i, inputs, targets, outputs)
        utils.write_event(log, step, loss=mean_loss)
        tq.close()
        save(epoch+1)
        valid_metrics = validation(model, criterion, valid_loader)
        write_event(log, step, **valid_metrics)
        valid_loss = valid_metrics['valid_loss']
        valid_losses.append(valid_loss)
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            shutil.copy(str(model_path), str(best_model_path))
    # except KeyboardInterrupt:
    #     tq.close()
    #     print('Ctrl+C, saving snapshot')
    #     save(epoch)
    #     print('done.')
    #     return

