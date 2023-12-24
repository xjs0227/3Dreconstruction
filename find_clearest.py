import cv2
import numpy as np
import matplotlib.pyplot as plt
import os



def find_clearest(image_folder):
    def estimate_blur(image):
        # gray = crop_image(image)
        gray = image
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        # print(laplacian_var)
        return laplacian_var

    # image_folder = '/home/hui/xjs/yyc_workspace/output2'  # 图像文件夹路径
    # image_folder = '/home/hui/xjs/yyc_workspace/results/3DReconLog-bga-120'
    #### Find the init index

    cur_max_res = 0
    clearest_idx = 0
    res = []

    files = os.listdir(image_folder)
    for i,file in enumerate(files):
        image = cv2.imread(os.path.join(image_folder,file), cv2.IMREAD_GRAYSCALE)
        # height, width = image.shape[0], image.shape[1]
        # image = image[0:int(0.9*height), 0:int(0.9*width)]
        blur_score = estimate_blur(image)
        if blur_score > cur_max_res:
            cur_max_res = blur_score
            clearest_idx = i
        res.append(blur_score)
    
    return clearest_idx
# for frame_index in range(start_frame, end_frame):
#     frame_id = "{:04d}".format(frame_index)
#     image_path = f"{image_folder}/reco{frame_id}.png"   # 图像文件路径
#     image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

#     blur_score = estimate_blur(image)
#     if blur_score > cur_max_res:
#         cur_max_res = blur_score
#         clearest_idx = frame_index
#     res.append(blur_score)

# plt.plot(res)

# plt.savefig('gray_mean_curve2.png')  # 保存图像为gray_mean_curve.png

# print("clearest index is", files[clearest_idx])


    
    




