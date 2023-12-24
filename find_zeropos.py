import cv2
import numpy as np




def remove_black_border(image):

    # 获取图像的尺寸
    height, width = image.shape[:2]

    # 从图像四周开始往中间遍历像素
    top = 0
    bottom = height - 1
    left = 0
    right = width - 1

    # 遍历顶部行，查找黑色边框
    while top < bottom:
        if np.all(image[top, :] == 0):
            top += 1
        else:
            break

    # 遍历底部行，查找黑色边框
    while bottom > top:
        if np.all(image[bottom, :] == 0):
            bottom -= 1
        else:
            break

    # 遍历左侧列，查找黑色边框
    while left < right:
        if np.all(image[:, left] == 0):
            left += 1
        else:
            break

    # 遍历右侧列，查找黑色边框
    while right > left:
        if np.all(image[:, right] == 0):
            right -= 1
        else:
            break

    # 裁剪图像，去除黑色边框
    cropped_image = image[top:bottom+1, left:right+1]

    return cropped_image

def is_padding(image):
    crop_image = remove_black_border(image)
    # import pdb; pdb.set_trace()
    if crop_image.shape != image.shape:
        return True
    else:
        return False

start_idx = 0
slices = 100
image_folder = ''
for frame_index in range(slices):
    frame_id = "{:04d}".format(frame_index)
    image_path = f"{image_folder}/reco{frame_id}.png"  # 图像文件路径
    image = cv2.imread(image_path)
    if is_padding(image):
        start_idx = frame_index
        break
  
