import random
import os
import shutil


def random_select_images(source_folder, number_of_images):
    # 获取所有图像文件
    all_images = [file for file in os.listdir(source_folder) if file.endswith(('.png', '.jpg', '.jpeg'))]

    # 如果图像数量少于所需数量，则全部选择
    if len(all_images) <= number_of_images:
        return all_images

    # 随机选择指定数量的图像
    selected_images = random.sample(all_images, number_of_images)
    return selected_images

def copy_selected_images(source_folder, destination_folder, selected_images):
    # 检查并创建目标文件夹
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # 复制选中的图像到目标文件夹
    for image in selected_images:
        source_image_path = os.path.join(source_folder, image)
        destination_image_path = os.path.join(destination_folder, image)
        shutil.copy(source_image_path, destination_image_path)

    return "图像复制完成"

# 设置文件夹路径和图像数量
hazy_folder = './data/rain/test/hazy'
number_of_images = 500
destination_folder = './data/classify/GT-RAIN'

# 调用函数进行随机选择
selected_images = random_select_images(hazy_folder, number_of_images)
result = copy_selected_images(hazy_folder, destination_folder, selected_images)