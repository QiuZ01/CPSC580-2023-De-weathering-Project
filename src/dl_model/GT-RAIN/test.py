import torch
from PIL import Image
from natsort import natsorted
from glob import glob

from tqdm.notebook import tqdm

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import numpy as np
import os
from model import GTRainModel

params = {
    'load_checkpoint': './model/model_checkpoint.pth',  # Dir to load model weights
    'input_path': '../../data/rain/GT-RAIN_test',
    'gt_path': '',
    'save_path': './outputs/',
    'init_type': 'normal',  # Initialization type
    'norm_layer_type': 'batch',  # Normalization type
    'activation_func': torch.nn.LeakyReLU(negative_slope=0.10, inplace=True),  # Activation function
    'upsample_mode': 'bilinear',  # Mode for upsampling
    'ngf': 64,
    'n_blocks': 9,
    'use_mode': 'test',
}

os.makedirs(params['save_path'], exist_ok=True)

# Make the model
model = GTRainModel(
    ngf=params['ngf'],
    n_blocks=params['n_blocks'],
    norm_layer_type=params['norm_layer_type'],
    activation_func=params['activation_func'],
    upsample_mode=params['upsample_mode'],
    init_type=params['init_type'])

print(model)
model.cuda()

# Load model weights
print('Loading weights:', params['load_checkpoint'])
checkpoint = torch.load(params['load_checkpoint'])  # , map_location=torch.device('cpu')
model.load_state_dict(checkpoint['state_dict'], strict=True)  # strict=True
model.eval()

# # Section for running with generic test sets
# if params['gt_path']:
#     total_PSNR_input = 0
#     total_SSIM_input = 0
#     total_PSNR_output = 0
#     total_SSIM_output = 0
#     clean_img_paths = natsorted(glob(params['gt_path']))
#
# rainy_img_paths = natsorted(glob(params['input_path']))
# num_paths = len(rainy_img_paths)
#
# for i in tqdm(range(num_paths)):
#     filename = rainy_img_paths[i].split('/')[-1][:-4]
#     img = Image.open(rainy_img_paths[i])
#     img = np.array(img, dtype=np.float32)
#     img *= 1 / 255
#     height, width = img.shape[:2]
#
#     img = img[:height - height % 4, :width - width % 4, :]
#     input = torch.from_numpy(img).permute((2, 0, 1)) * 2 - 1
#     input = torch.unsqueeze(input, 0).cuda()
#     output = (model(input)[0] * 0.5 + 0.5).squeeze().permute((1, 2, 0))
#     output = output.detach().cpu().numpy()
#
#     if params['gt_path']:
#         gt_img = Image.open(clean_img_paths[i])
#         gt_img = np.array(gt_img, dtype=np.float32)
#         gt_img *= 1 / 255
#         gt_img = gt_img[:height - height % 4, :width - width % 4, :]
#         total_PSNR_input += psnr(gt_img, img)
#         total_SSIM_input += ssim(gt_img, img, multichannel=True)
#         total_PSNR_output += psnr(gt_img, output)
#         total_SSIM_output += ssim(gt_img, output, multichannel=True)
#
#     # USE THIS BLOCK TO SAVE
#     im = Image.fromarray((output * 255).astype(np.uint8))
#     im.save(f"{params['save_path']}/{filename}.png")
#
# if params['gt_path']:
#     print(f"PSNR Input: {total_PSNR_input / num_paths}")
#     print(f"SSIM Input: {total_SSIM_input / num_paths}")
#     print(f"PSNR Output: {total_PSNR_output / num_paths}")
#     print(f"SSIM Output: {total_SSIM_output / num_paths}")

# Section for running on GT-RAIN test set
total_PSNR_input = 0
total_SSIM_input = 0
total_PSNR_output = 0
total_SSIM_output = 0

rain_acc_total_PSNR_input = 0
rain_acc_total_SSIM_input = 0
rain_acc_total_PSNR_output = 0
rain_acc_total_SSIM_output = 0
rain_acc_num_scenes = 0

dense_streak_total_PSNR_input = 0
dense_streak_total_SSIM_input = 0
dense_streak_total_PSNR_output = 0
dense_streak_total_SSIM_output = 0
dense_streak_num_scenes = 0

scene_paths = natsorted(glob(f"{params['input_path']}/*"))
for scene_path in tqdm(scene_paths):
    scene_name = scene_path.split('/')[-1]
    clean_img_path = glob(scene_path + '/*C-000.png')[0]
    rainy_img_paths = natsorted(glob(scene_path + '/*R-*.png'))
    scene_PSNR_input = 0
    scene_SSIM_input = 0
    scene_PSNR_output = 0
    scene_SSIM_output = 0

    for i in tqdm(range(len(rainy_img_paths))):
        filename = rainy_img_paths[i].split('/')[-1][:-4]
        img = Image.open(rainy_img_paths[i])
        gt_img = Image.open(clean_img_path)
        img = np.array(img, dtype=np.float32)
        img *= 1 / 255
        gt_img = np.array(gt_img, dtype=np.float32)
        gt_img *= 1 / 255
        height, width = img.shape[:2]

        img = img[:height - height % 4, :width - width % 4, :]

        gt_img = gt_img[:height - height % 4, :width - width % 4, :]

        input = torch.from_numpy(img).permute((2, 0, 1)) * 2 - 1
        input = torch.unsqueeze(input, 0).cuda()
        output = (model(input)[0] * 0.5 + 0.5).squeeze().permute((1, 2, 0))
        output = output.detach().cpu().numpy()

        # USE THIS BLOCK TO SAVE
        im = Image.fromarray((output * 255).astype(np.uint8))
        im.save(f"{params['save_path']}/{scene_name}/{filename}.png")

        scene_PSNR_input += psnr(gt_img, img)
        scene_SSIM_input += ssim(gt_img, img, multichannel=True)
        scene_PSNR_output += psnr(gt_img, output)
        scene_SSIM_output += ssim(gt_img, output, multichannel=True)
    print(f"Scene: {scene_name}")
    print(f"Scene PSNR Input: {scene_PSNR_input / len(rainy_img_paths)}")
    print(f"Scene SSIM Input: {scene_SSIM_input / len(rainy_img_paths)}")
    print(f"Scene PSNR Output: {scene_PSNR_output / len(rainy_img_paths)}")
    print(f"Scene SSIM Output: {scene_SSIM_output / len(rainy_img_paths)}")

    total_PSNR_input += scene_PSNR_input / len(rainy_img_paths)
    total_SSIM_input += scene_SSIM_input / len(rainy_img_paths)
    total_PSNR_output += scene_PSNR_output / len(rainy_img_paths)
    total_SSIM_output += scene_SSIM_output / len(rainy_img_paths)

    if scene_name in ["Oinari_0-0", "M1135_0-0", "Table_Rock_0-0"]:
        rain_acc_total_PSNR_input += scene_PSNR_input / len(rainy_img_paths)
        rain_acc_total_SSIM_input += scene_SSIM_input / len(rainy_img_paths)
        rain_acc_total_PSNR_output += scene_PSNR_output / len(rainy_img_paths)
        rain_acc_total_SSIM_output += scene_SSIM_output / len(rainy_img_paths)
        rain_acc_num_scenes += 1
    else:
        dense_streak_total_PSNR_input += scene_PSNR_input / len(rainy_img_paths)
        dense_streak_total_SSIM_input += scene_SSIM_input / len(rainy_img_paths)
        dense_streak_total_PSNR_output += scene_PSNR_output / len(rainy_img_paths)
        dense_streak_total_SSIM_output += scene_SSIM_output / len(rainy_img_paths)
        dense_streak_num_scenes += 1
num_scenes = len(scene_paths)
print(f"Total PSNR Input: {total_PSNR_input / (num_scenes)}")
print(f"Total SSIM Input: {total_SSIM_input / num_scenes}")
print(f"Total PSNR Output: {total_PSNR_output / num_scenes}")
print(f"Total SSIM Output: {total_SSIM_output / num_scenes}")

print(f"rain accumulation Total PSNR Input: {rain_acc_total_PSNR_input / (rain_acc_num_scenes)}")
print(f"rain accumulation Total SSIM Input: {rain_acc_total_SSIM_input / rain_acc_num_scenes}")
print(f"rain accumulation Total PSNR Output: {rain_acc_total_PSNR_output / rain_acc_num_scenes}")
print(f"rain accumulation Total SSIM Output: {rain_acc_total_SSIM_output / rain_acc_num_scenes}")

print(f"dense streak Total PSNR Input: {dense_streak_total_PSNR_input / (dense_streak_num_scenes)}")
print(f"dense streak Total SSIM Input: {dense_streak_total_SSIM_input / dense_streak_num_scenes}")
print(f"dense streak Total PSNR Output: {dense_streak_total_PSNR_output / dense_streak_num_scenes}")
print(f"dense streak Total SSIM Output: {dense_streak_total_SSIM_output / dense_streak_num_scenes}")