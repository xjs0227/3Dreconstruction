import cv2
import numpy as np
import os
from scipy.ndimage import zoom

#### Experiment Setting

##### eagle-8um 200 slices
 
# image_folder = '/home/hui/xjs/3Dreconstruction/results/eagle/3DReconLog-8um-120/1'
# start_index = 95
# end_index = 120
# target_spacing = 16.0
# original_spacing = 140.0
# output_folder = '/home/hui/xjs/yyc_workspace/results/interpolated_images/3DReconLog-8um-120/1'

##### eagle-8um 500 slices

# image_folder = '/home/hui/xjs/3Dreconstruction/results/eagle/3DReconLog-8um-120/2'
# start_index = 245
# end_index = 270
# target_spacing = 16.0
# original_spacing = 140.0
# output_folder = '/home/hui/xjs/yyc_workspace/results/interpolated_images/3DReconLog-8um-120/2'

##### eagle-25um 200 slices

# image_folder = '/home/hui/xjs/3Dreconstruction/results/eagle/3DReconLog-25um-120/2'
# start_index = 97
# end_index = 114
# target_spacing = 50.0
# original_spacing = 236.0
# start_pos = -2000
# end_pos = 1500
# output_folder = '/home/hui/xjs/yyc_workspace/results/interpolated_images/3DReconLog-25um-120/2'

##### e5-120 

# image_folder = '/home/hui/xjs/3Dreconstruction/results/e5-120'
# start_index = 118
# end_index = 124
# target_spacing = 50.0
# original_spacing = 150.0
# start_pos = -950
# end_pos = 50
# output_folder = '/home/hui/xjs/yyc_workspace/results/interpolated_images/e5-120'

##### 20231207

# image_folder = '/home/hui/xjs/3Dreconstruction/results/20231207/bga'
# start_index = 97
# end_index = 114
# target_spacing = 50.0
# original_spacing = 236.0
# start_pos = -2000
# end_pos = 1500
# output_folder = '/home/hui/xjs/yyc_workspace/results/interpolated_images/3DReconLog-25um-120/2'



def interpolate_images(images, target_spacing, original_spacing):
    # 获取图像数量和原始间距
    num_images = len(images)

    # 计算缩小比例
    scale_factor = target_spacing / original_spacing

    # 计算新的图像数量和间距
    new_num_images = int(num_images / scale_factor)
    new_spacing = original_spacing * scale_factor

    # 创建新的图像序列
    new_images = []

    # 对每个新的Z轴位置进行插值
    for i in range(new_num_images):
        # 计算插值的原始图像索引
        idx_low = int(i * scale_factor)
        idx_high = min(idx_low + 1, num_images - 1)

        # 计算插值权重
        weight_high = i * scale_factor - idx_low
        weight_low = 1.0 - weight_high

        # 进行插值
        interpolated_image = cv2.addWeighted(images[idx_low], weight_low, images[idx_high], weight_high, 0)

        # 添加插值后的图像到新的图像序列
        new_images.append(interpolated_image)

    return new_images

# def interpolate_images(image_list, z_spacing, target_spacing):
#     # 计算当前图像深度和目标图像深度
#     current_depth = len(image_list)
#     target_depth = int(current_depth * z_spacing / target_spacing)

#     # 创建当前图像的三维数组
#     image_array = np.stack(image_list, axis=-1)

#     # 计算Z轴上的缩放因子
#     z_scale = target_depth / current_depth
#     # import pdb; pdb.set_trace()

#     # 在Z轴上进行重采样
#     resampled_array = zoom(image_array, (1, 1, z_scale), order=1)
#     # 将重采样后的数组重新拆分为图像列表
#     resampled_images = np.split(resampled_array, target_depth, axis=-1)

#     # 返回重采样后的图像列表
#     return [resampled_image.squeeze() for resampled_image in resampled_images]


# images = []

# for i in range(start_index, end_index + 1):
#     image_path = image_folder + f"/reco{i:04d}.png"  # 根据文件名格式构建图像路径
#     # import pdb; pdb.set_trace()
#     image = cv2.imread(image_path, 0)
#     images.append(image)


# interpolated_images = interpolate_images(images, original_spacing, target_spacing)
# os.makedirs(output_folder, exist_ok=True)

# # 保存插值后的图像

# for i, image in enumerate(interpolated_images):
#     output_path = os.path.join(output_folder, f'interpolated_image_{start_pos+i*target_spacing}.png')
#     cv2.imwrite(output_path, image)

