import os
import random

dataset_dir = '/home/suepr20/luofan/my_lanedetection/mytraining_data_example'
# 指定原始图片目录、二值化分割目录和实例分割目录
train_txt_path = os.path.join(dataset_dir, 'train.txt')
test_txt_path = os.path.join(dataset_dir, 'test.txt')
image_dir = os.path.join(dataset_dir, "image")
binary_seg_dir = os.path.join(dataset_dir, "gt_binary_image")
instance_seg_dir = os.path.join(dataset_dir, "gt_instance_image")

# 获取图片文件列表
image_files = os.listdir(image_dir)
num_images = len(image_files)

# 计算训练集和测试集的数量
train_ratio = 0.9
num_train = int(num_images * train_ratio)
num_test = num_images - num_train

# 随机打乱图片文件列表
random.shuffle(image_files)

# 划分训练集和测试集
train_files = image_files[:num_train]
print(f"train_files ={train_files}")
test_files = image_files[num_train:]
print(f"test_files ={test_files}")
# 创建.txt文件并写入训练集和测试集的图片信息
with open(train_txt_path, 'w') as f_train, open(test_txt_path, 'w') as f_test:
    for image_file in train_files:
        file_name = os.path.splitext(image_file)[0]
        binary_seg_file = os.path.join(binary_seg_dir, file_name + '.png')
        instance_seg_file = os.path.join(instance_seg_dir, file_name + '.png')
        f_train.write(f"{os.path.join(image_dir, image_file)} {binary_seg_file} {instance_seg_file}\n")
    for image_file in test_files:
        file_name = os.path.splitext(image_file)[0]
        binary_seg_file = os.path.join(binary_seg_dir, file_name + '.png')
        instance_seg_file = os.path.join(instance_seg_dir, file_name + '.png')
        f_test.write(f"{os.path.join(image_dir, image_file)} {binary_seg_file} {instance_seg_file}\n")
