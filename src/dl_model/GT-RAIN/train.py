import os
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torchvision

from tqdm.notebook import tqdm
from pathlib import Path
import time
from tensorboardX import SummaryWriter
from utils import *
from model import GTRainModel
from data import GTRainDataset, CustomBatchSampler
# Parameters
params = {
    'batch_size': 4,  # batch size
    'num_epochs': 3,  # number of epochs to train
    'warmup_epochs': 4,  # number of epochs for warmup
    'initial_lr': 2e-4,  # initial learning rate used by scheduler
    'min_lr': 1e-6,  # minimum learning rate used by scheduler
    'val_epoch': 1,  # validation done every k epochs
    'init_type': 'normal',  # 'xavier', # Initialization type
    'ngf': 64,  # the number of channels for the model capacity
    'n_blocks': 9,  # the number of blocks in ResNet
    'norm_layer_type': 'batch',  # Normalization type
    'activation_func': torch.nn.LeakyReLU(negative_slope=0.10, inplace=True),  # Activation function
    'upsample_mode': 'bilinear',  # Mode for upsampling
    'save_dir': './checkpoints/weights_dir',  # Dir to save the model weights
    'save_every': 1,  # save every k epoch
    'train_dir_list': ['../../data/rain/GT-RAIN_train'],  # Dir for the training data
    'val_dir_list': ['../../data/rain/GT-RAIN_val'],  # Dir for the val data
    'rain_mask_dir': '../../data/rain/rain_mask_dir',  # Dir for the rain masks
    'img_size': 256,  # the size of image input
    'zoom_min': .06,  # the minimum zoom for RainMix
    'zoom_max': 1.8,  # the maximum zoom for RainMix
    'l1_loss_weight': 0.1,  # weight for l1 loss
    'ssim_loss_weight': 1.0,  # weight for the ssim loss
    'robust_loss_weight': 0.1,  # weight for rain robust loss
    'temperature': 0.25,  # Temperature for the rain robust loss
    'resume_train': False,  # begin training using loaded checkpoint
    'model_path': None,  # Dir to load model weights
    'tensorboard_log_step_train': 100,  # Number of steps to log into tensorboard when training
    'tensorboard_log_step_val': 1,  # This number will be updated automatically based after creating the dataloaders
    'use_mode': 'train',
}

# Create dir to save the weights
Path(params['save_dir']).mkdir(parents=True, exist_ok=True)

# Set up tensorboard SummaryWriter and directories
writer = SummaryWriter(os.path.join(params['save_dir'], 'tensorboard'))

# Create the DataLoaders for training and validation
train_dataset = GTRainDataset(
    train_dir_list=params['train_dir_list'],
    val_dir_list=params['val_dir_list'],
    rain_mask_dir=params['rain_mask_dir'],
    img_size=params['img_size'],
    is_train=True,
    zoom_min=params['zoom_min'],
    zoom_max=params['zoom_max'])

train_loader = DataLoader(
    dataset=train_dataset,
    batch_sampler=CustomBatchSampler(
        train_dataset.get_scene_indices(),
        batch_size=params['batch_size']),
    pin_memory=True
)

val_dataset = GTRainDataset(
    train_dir_list=params['train_dir_list'],
    val_dir_list=params['val_dir_list'],
    rain_mask_dir=params['rain_mask_dir'],
    img_size=params['img_size'],
    is_train=False)

val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=params['batch_size'],
    shuffle=True,
    drop_last=True,
    pin_memory=True)

print('Train set length:', len(train_dataset))
print('Val set lenth:', len(val_dataset))

# Adjust the log freq based on the number of training and val samples
params['tensorboard_log_step_val'] = int(params['tensorboard_log_step_train'] * len(val_dataset) / len(train_dataset))

"""
Script for training
"""

# Make the model
model = GTRainModel(
    ngf=params['ngf'],
    n_blocks=params['n_blocks'],
    norm_layer_type=params['norm_layer_type'],
    activation_func=params['activation_func'],
    upsample_mode=params['upsample_mode'],
    init_type=params['init_type'],
    use_mode=params['use_mode'])

print(model)
model.cuda()

# Setting up the optimizer and LR scheduler
# Different learning rate for the deformable groups
key_name_list = ['offset', 'modulator']
deform_params = []
normal_params = []
for cur_name, parameters in model.named_parameters():
    if any(key_name in cur_name for key_name in key_name_list):
        deform_params.append(parameters)
    else:
        normal_params.append(parameters)
print('deform:', len(deform_params), 'normal:', len(normal_params))

optimizer = optim.Adam(
    [{"params": normal_params},
     {"params": deform_params, "lr": params['initial_lr'] / 10}],
    lr=params['initial_lr'],
    betas=(0.9, 0.999),
    eps=1e-8)

scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    params['num_epochs'] - params['warmup_epochs'],
    eta_min=params['min_lr'])

scheduler = GradualWarmupScheduler(
    optimizer,
    multiplier=1.0,
    total_epoch=params['warmup_epochs'],
    after_scheduler=scheduler_cosine)

optimizer.zero_grad()
optimizer.step()
scheduler.step()  # To start warmup

# Setting up Loss Function
criterion_l1 = nn.L1Loss().cuda()
criterion_neg_ssim = ShiftMSSSIM().cuda()
criterion_robust = RainRobustLoss(
    batch_size=params['batch_size'],
    n_views=2,
    device=torch.device("cuda"),
    temperature=params['temperature']).cuda()

start_epoch = 0

if params['resume_train']:
    print(f"Loading checkpoint {params['model_path']}")
    checkpoint = torch.load(params['model_path'])

    # Load Model
    model.load_state_dict(checkpoint['state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"Resuming epoch: {start_epoch}")

    for i in range(start_epoch):
        scheduler.step()

    print(f"Resuming lr: {optimizer.param_groups[0]['lr']}")
else:
    print(f"Initial lr: {optimizer.param_groups[0]['lr']}")

# TRAINING AND VALIDATION
best_epoch = 0
best_psnr = 0
for epoch in range(start_epoch, params['num_epochs']):
    epoch_start_time = time.time()
    epoch_loss = 0

    # TRAINING
    model.train()
    train_loop = tqdm(train_loader, leave=False, position=0)
    train_loop.set_description(f"Epoch {epoch}/{params['num_epochs']}")
    for batch_idx, batch_data in enumerate(train_loop):

        # Load the data
        input_img = batch_data['input_img'].cuda()
        target_img = batch_data['target_img'].cuda()

        # Check the current step
        current_step = epoch * len(train_loader) + batch_idx

        #########
        # Train #
        #########

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass through the entire model (both parts)
        output_img, output_features = model(input_img, target_img)

        # Calculate losses between gt and output
        loss = 0

        if params['l1_loss_weight']:
            loss_l1 = criterion_l1(output_img, target_img)
            loss += params['l1_loss_weight'] * loss_l1
            loss_l1_log = loss_l1.item()

        if params['ssim_loss_weight']:
            loss_ssim = criterion_neg_ssim(output_img, target_img)
            loss += params['ssim_loss_weight'] * loss_ssim
            loss_ssim_log = loss_ssim.item()

        if params['robust_loss_weight']:
            loss_robust = criterion_robust(output_features)
            loss += params['robust_loss_weight'] * loss_robust
            loss_robust_log = loss_robust.item()

        # Backwards pass and step
        loss.backward()
        optimizer.step()

        # Log
        epoch_loss += loss.item()
        loss_log = loss.item()

        # Tensorboard
        if (current_step % params['tensorboard_log_step_train']) == 0:

            # Log loss
            writer.add_scalar('loss/train', loss_log, current_step)

            # Seperate loss
            if params['l1_loss_weight']:
                writer.add_scalar('l1_loss/train', loss_l1_log, current_step)
            if params['ssim_loss_weight']:
                writer.add_scalar('ssim_loss/train', loss_ssim_log, current_step)
            if params['robust_loss_weight']:
                writer.add_scalar('robust_loss/train', loss_robust_log, current_step)

            # Log images
            log_img = torchvision.utils.make_grid(
                torch.cat([
                    input_img[0:2, ...].cpu() * 0.5 + 0.5,
                    output_img[0:2, ...].detach().cpu() * 0.5 + 0.5,
                    target_img[0:2, ...].cpu() * 0.5 + 0.5], dim=-1), nrow=1)

            log_img_difference = torchvision.utils.make_grid(
                torch.cat([
                    torch.abs(input_img[0:2, ...].cpu() - target_img[0:2, ...].cpu()) / 2,
                    torch.abs(input_img[0:2, ...].cpu() - output_img[0:2, ...].detach().cpu()) / 2,
                    torch.abs(output_img[0:2, ...].detach().cpu() - target_img[0:2, ...].cpu()) / 2,
                ], dim=-1), nrow=1)
            writer.add_image(
                'images/train (input, out, gt)',
                log_img, current_step)
            writer.add_image(
                'images_diff/train (input-gt, input-out, out-gt)',
                log_img_difference, current_step)

    # Print info
    print(
        f"Epoch: {epoch}\t"
        f"Time: {time.time() - epoch_start_time:.4f}\n"
        f"Train Loss: {epoch_loss / len(train_loader):.4f}\t"
        f"Learning Rate First {optimizer.param_groups[0]['lr']:.8f}\t")

    # Log images
    log_img = torchvision.utils.make_grid(
        torch.cat([
            input_img[0:2, ...].cpu() * 0.5 + 0.5,
            output_img[0:2, ...].detach().cpu() * 0.5 + 0.5,
            target_img[0:2, ...].cpu() * 0.5 + 0.5], dim=-1), nrow=1)

    log_img = log_img.permute(1, 2, 0)

    log_img_difference = torchvision.utils.make_grid(
        torch.cat([
            torch.abs(input_img[0:2, ...].cpu() - target_img[0:2, ...].cpu()) / 2,
            torch.abs(input_img[0:2, ...].cpu() - output_img[0:2, ...].detach().cpu()) / 2,
            torch.abs(output_img[0:2, ...].detach().cpu() - target_img[0:2, ...].cpu()) / 2
        ], dim=-1), nrow=1)

    log_img_difference = log_img_difference.permute(1, 2, 0)

    # Show image
    show_img(
        img=log_img,
        title=f'Train Result (input, out, gt) in epoch {epoch}')
    show_img(
        img=log_img_difference,
        title=f'Train Result (input-gt, input-out, out-gt) in epoch {epoch}',
        figsize=(9, 9))

    ##############
    # Validation #
    ##############

    if epoch % params['val_epoch'] == 0:
        model.eval()
        epoch_start_time = time.time()
        epoch_loss = 0

        val_loop = tqdm(val_loader, leave=False, position=0)
        val_loop.set_description('Val Epoch')
        for batch_idx, batch_data in enumerate(val_loop):

            # Load data
            input_img = batch_data['input_img'].cuda()
            target_img = batch_data['target_img'].cuda()

            # Check the current step
            current_step = epoch * len(val_loader) + batch_idx

            # Forward pass of model
            with torch.no_grad():

                output_img, output_feature = model(input_img, target_img)

                # Calculate losses between pseudo-gt and output
                loss = 0
                if params['l1_loss_weight']:
                    loss_l1 = criterion_l1(output_img, target_img)
                    loss += params['l1_loss_weight'] * loss_l1
                    loss_l1_log = loss_l1.item()

                if params['ssim_loss_weight']:
                    loss_ssim = criterion_neg_ssim(output_img, target_img)
                    loss += params['ssim_loss_weight'] * loss_ssim
                    loss_ssim_log = loss_ssim.item()

                epoch_loss += loss.item()
                loss_log = loss.item()

                # Tensorboard
                if (current_step % params['tensorboard_log_step_val']) == 0:

                    # Log loss
                    writer.add_scalar('loss/val', loss_log, current_step)

                    # Seperate loss
                    if params['l1_loss_weight']:
                        writer.add_scalar('l1_loss/val', loss_l1_log, current_step)
                    if params['ssim_loss_weight']:
                        writer.add_scalar('ssim_loss/val', loss_ssim_log, current_step)

                    # Log images
                    log_img = torchvision.utils.make_grid(
                        torch.cat([
                            input_img[0:2, ...].cpu() * 0.5 + 0.5,
                            output_img[0:2, ...].cpu() * 0.5 + 0.5,
                            target_img[0:2, ...].cpu() * 0.5 + 0.5], dim=-1), nrow=1)
                    writer.add_image('images/val (input-out-gt)', log_img, current_step)

        # Print info
        avg_val_loss = epoch_loss / len(val_loader)
        print(
            f"Val Epoch\t"
            f"Time: {time.time() - epoch_start_time:.4f}\t"
            f"Val Loss: {avg_val_loss:.4f}")

        # Log images
        log_img = torchvision.utils.make_grid(
            torch.cat([
                input_img[0:2, ...].cpu() * 0.5 + 0.5,
                output_img[0:2, ...].cpu() * 0.5 + 0.5,
                target_img[0:2, ...].cpu() * 0.5 + 0.5], dim=-1), nrow=1)
        log_img = log_img.permute(1, 2, 0)

        # Show image
        show_img(
            img=log_img,
            title=f'Val Result (input, out, gt) in epoch {epoch}')

    # Move the scheduler forward
    scheduler.step()

    # Save every few epochs
    if epoch % params['save_every'] == 0:
        print('Saving...')
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()},
            os.path.join(params['save_dir'], f'model_epoch_{epoch}.pth'))

    # Tensorboard
    writer.flush()
# Close tensorboard
writer.close()
