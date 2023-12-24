# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# image = cv2.imread('/home/hui/xjs/yyc_workspace/results/3DReconLog-bga-60/reco0045.png', cv2.IMREAD_GRAYSCALE)
# hist = cv2.calcHist([image], [0], None, [256], [0, 256])

# plt.figure()
# plt.title('Grayscale Histogram')
# plt.xlabel('Bins')
# plt.ylabel('Frequency')
# plt.plot(hist)
# plt.xlim([0, 256])
# plt.savefig("highlight.png")

import cv2
import numpy as np
import matplotlib.pyplot as plt


import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def remove_highlight(image):
    # image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    height, width = image.shape
    center_x = width // 2
    center_y = height // 2

    # 计算每个像素到图像中心的距离
    distances = np.sqrt((np.arange(height)[:, np.newaxis] - center_y) ** 2 + (np.arange(width) - center_x) ** 2)

    # 计算高斯分布的密度函数
    sigma = max(center_x, center_y) / 10  # 标准差
    gaussian_weights = norm.pdf(distances, scale=sigma)

    # 根据高斯权重调整像素灰度值
    # import pdb; pdb.set_trace()
    adjusted_image = image - (image * gaussian_weights * 40).astype(np.uint8) + np.mean(gaussian_weights*40)
    return adjusted_image

    # output_filename = 'processed_image.jpg'
    # cv2.imwrite(output_filename, adjusted_image)
    # print(f"Processed image saved as {output_filename}")

# plt.figure(figsize=(10, 5))

# plt.subplot(1, 2, 1)
# plt.imshow(image, cmap='gray')
# plt.title('Original Image')
# plt.axis('off')

# plt.subplot(1, 2, 2)
# plt.imshow(adjusted_image, cmap='gray')
# plt.title('Adjusted Image')
# plt.axis('off')

# plt.tight_layout()
# plt.show()

