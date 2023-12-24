import cv2
import numpy as np

def invert_pixels_below_threshold(gray, threshold):
    # 将图像转换为灰度图像
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 将低于阈值的像素取反
    # import pdb; pdb.set_trace()
    mask = gray < threshold
    inverted_pixels = np.where(mask, 255 - gray, gray)

    # 将取反后的像素重新赋值给图像
    # inverted_image = image.copy()
    # inverted_image[..., 0] = inverted_pixels
    # inverted_image[..., 1] = inverted_pixels
    # inverted_image[..., 2] = inverted_pixels

    return inverted_pixels

# 读取图像
# image = cv2.imread('/home/hui/xjs/yyc_workspace/hm.png')

# # 设置阈值
# # import pdb; pdb.set_trace()
# threshold = 59

# # 对低于阈值的像素进行取反
# result = invert_pixels_below_threshold(image, threshold)
# cv2.imwrite("result.png", result)