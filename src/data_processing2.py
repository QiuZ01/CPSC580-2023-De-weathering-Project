import os
import shutil

def process_images(source_folder, destination_folder):
    # 创建目标文件夹
    hazy_folder = os.path.join(destination_folder, "hazy")
    gt_folder = os.path.join(destination_folder, "GT")

    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    if not os.path.exists(hazy_folder):
        os.makedirs(hazy_folder)
    if not os.path.exists(gt_folder):
        os.makedirs(gt_folder)

    # 遍历子文件夹
    for subdir in os.listdir(source_folder):
        subdir_path = os.path.join(source_folder, subdir)
        if os.path.isdir(subdir_path):
            c_000_image = None

            # 找到以C-000.png结尾的图像
            for file in os.listdir(subdir_path):
                if file.endswith("C-000.png"):
                    c_000_image = file
                    break

            if c_000_image:
                c_000_path = os.path.join(subdir_path, c_000_image)

                # 处理其他图像
                for file in os.listdir(subdir_path):
                    file_path = os.path.join(subdir_path, file)
                    if file != c_000_image:
                        # 移动到hazy文件夹
                        shutil.copy(file_path, os.path.join(hazy_folder, file))

                        # 复制C-000图像到GT文件夹
                        new_gt_name = file
                        shutil.copy(c_000_path, os.path.join(gt_folder, new_gt_name))

    return "处理完成"

# 设置文件夹路径
source_folder = './data/rain/GT-RAIN_test'
destination_folder = './data/rain/test'

# 调用函数进行处理
process_images(source_folder, destination_folder)
