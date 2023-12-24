from __future__ import division
import os
import shutil
import numpy as np
from os.path import join
from imageio import get_writer, imread, imwrite
import cv2
import time

import astra
import math

# PROJECTION_RESULTS = r'D:\study\graduate\research\Study\project\3Dreconstruction\3D重建\data\Temp-e5\3DReconLog_e5_120\Run1'
# File = r"D:\st udy\graduate\research\Study\project\3Dreconstruction\3D重建\data\Temp-e4\3DReconLog_e4_23\Run76\Cal.txt"

PROJECTION_RESULTS = "/home/hui/xjs/data/20231207/bga/3DReconLog/Run1"  # "data/e5"
File = "/home/hui/xjs/data/20231207/bga/3DReconLog/Run1/Cal.txt" #"data/e5/Cal.txt"

RECONSTRUCTION_RESULTS = "/home/hui/xjs/3Dreconstruction/results/20231207/bga"
os.makedirs(RECONSTRUCTION_RESULTS, exist_ok=True)

# scale = 6
# X_INIT = 3096
# Y_INIT = 3104
# n = 120
# slices = 500
# resol_cam = 9.4
# r = 2
# # resolution = resol_cam * scale 
# resolution =  0.99 * 15.9 / resol_cam
# iteration = 1
scale = 6

X_INIT = 3096
Y_INIT = 3104
n = 120
slices = 200
resol_cam = 15.9
# resolution = resol_cam * scale 
resolution =  scale * 0.99 * 15.9 / resol_cam
iteration = 1

t1 = time.time()

print("saved in:", RECONSTRUCTION_RESULTS)
print("scale:", scale)
print("resolution:", resolution)
print("slices:", slices)
print("iteration:", iteration)

vectors = np.zeros((n, 12))
try:
    file = open(File, 'r')
except FileNotFoundError: 
    print('File is not found')
else:
    lines = file.readlines()
    flag = 0
    i = 0
    for line in lines:
        a = line.split()
        if a:
            if a[0] == "1":
                flag = 1
            if flag:
                z_source = int(a[9].split(".")[0]) / 1000

                x = int(a[13].split(".")[0]) / 1000
                y = int(a[14].split(".")[0]) / 1000
                z = int(a[15].split(".")[0]) / 1000

                x_source = x * z_source / z
                y_source = y * z_source / z

                # theta2 = (math.pi / 2 - math.atan2(math.sqrt(x ** 2 + y ** 2), -z))
                # ry = [[math.cos(theta2), 0, math.sin(theta2)],
                #       [0, 1, 0],
                #       [-math.sin(theta2), 0, math.cos(theta2)]]
                # rx = [[1, 0, 0],
                #       [0, math.cos(theta2), -math.sin(theta2)],
                #       [0, math.sin(theta2), math.cos(theta2)]]
                # rz = [[math.cos(theta2), -math.sin(theta2), 0],
                #       [math.sin(theta2), math.cos(theta2), 0],
                #       [0, 0, 1]]
                # rx = np.array(rx)
                # ry = np.array(ry)
                # rz = np.array(rz)

                vectors[i][3] = -y
                vectors[i][4] = -x
                vectors[i][5] = z

                # theta = math.atan2(abs(vectors[i][4]), abs(vectors[i][3]))

                vectors[i][0] = -y_source
                vectors[i][1] = -x_source
                vectors[i][2] = z_source

                vectors[i][9] = resolution# math.cos(theta) * vectors[i][3] / abs(vectors[i][3]) * resolution * 6 #resolution * 6
                vectors[i][10] = 0#math.sin(theta) * vectors[i][4] / abs(vectors[i][4]) * resolution * 6 #resolution * 6
                vectors[i][11] = 0

                vectors[i][6] = vectors[i][10]
                vectors[i][7] = -vectors[i][9]
                vectors[i][8] = 0
                # vectors[i][6] = vectors[i][7] = 0.099
                # vectors[i][11] = 0.099
                # print(vectors[0])

                # [vectors[i][3], vectors[i][4], vectors[i][5]] = np.matmul(ry.T, np.array([vectors[i][3], vectors[i][4], vectors[i][5]]))
                # [vectors[i][0], vectors[i][1], vectors[i][2]] = np.matmul(ry.T, np.array([vectors[i][0], vectors[i][1], vectors[i][2]]))
                # [vectors[i][6], vectors[i][7], vectors[i][8]] = np.matmul(ry.T, np.array([vectors[i][6], vectors[i][7], vectors[i][8]]))
                # [vectors[i][9], vectors[i][10], vectors[i][11]] = np.matmul(ry.T, np.array([vectors[i][9], vectors[i][10], vectors[i][11]]))

                i += 1
                # print(i)


file.close()

# exit()

class Virtual_Cbct():
    def __init__(self):
        # Configuration.
        self.distanceFromSourceToOrigin = 70.9  # [mm]
        self.distanceFromOriginToDetector = 370  # [mm]
        self.pixelSize = resolution * 6  # [mm]
        self.detectorRows = Y_INIT  # Vertical size of detector [pixels].
        self.detectorColumns = X_INIT  # Horizontal size of detector [pixels].
        self.numberOfProjections = n

        # Configuration.
        # self.distanceFromSourceToOrigin = 300  # [mm]
        # self.distanceFromOriginToDetector = 100  # [mm]
        # self.pixelSize = 1.05  # [mm]
        # self.detectorRows = 200  # Vertical size of detector [pixels].
        # self.detectorColumns = 200  # Horizontal size of detector [pixels].
        # self.numberOfProjections = 180

    def start_run(self):
        # self.phantom = phantom
        # self.angles = np.linspace(
        #     0, 2 * np.pi, num=self.numberOfProjections, endpoint=False)
        # self.projectionGeometry = astra.create_proj_geom(
        #     'cone', 1, 1,
        #     self.detectorRows,
        #     self.detectorColumns,
        #     self.angles,
        #     self.distanceFromSourceToOrigin / self.pixelSize,
        #     self.distanceFromOriginToDetector / self.pixelSize
        # )

        self.vectors = vectors
        # self.vectors = np.zeros((self.numberOfProjections, 12))
        # for i in range(self.numberOfProjections):
        #     self.vectors[i][0] = math.sin(self.angles[i]) * self.distanceFromSourceToOrigin
        #     self.vectors[i][1] = -math.cos(self.angles[i]) * self.distanceFromSourceToOrigin
        #
        #     self.vectors[i][3] = -math.sin(self.angles[i]) * self.distanceFromOriginToDetector
        #     self.vectors[i][4] = math.cos(self.angles[i]) * self.distanceFromOriginToDetector
        #
        #     self.vectors[i][6] = math.cos(self.angles[i]) * self.pixelSize
        #     self.vectors[i][7] = math.sin(self.angles[i]) * self.pixelSize
        #
        #     self.vectors[i][11] = self.pixelSize

        self.projectionGeometry = astra.create_proj_geom(
            'cone_vec',
            self.detectorRows//scale,
            self.detectorColumns//scale,
            self.vectors,
        )
        self.vol_geom = astra.creators.create_vol_geom(
            self.detectorColumns//scale, self.detectorColumns//scale, slices)
        # self.create_projections()
        self.create_reconstructions()

    def create_projections(self):
        # Projections are created as if phantom is rotated clockwise.
        phantomId = astra.data3d.create('-vol', self.vol_geom, data=self.phantom)

        projectionId, projections = astra.creators.create_sino3d_gpu(
            phantomId, self.projectionGeometry, self.vol_geom)

        projections /= np.max(projections)

        # Apply Poisson noise.
        projections = np.random.poisson(projections * 10000) / 10000
        projections[projections > 1.1] = 1.1
        projections /= 1.1

        # Save projections.
        projections = np.round(projections * 65535).astype(np.uint16)

        # If output path exists (from previous run) delete and create new folder
        output_dir = PROJECTION_RESULTS
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)

        for i in range(self.numberOfProjections):
            projection = projections[:, i, :]
            with get_writer(join(output_dir, 'proj%04d.tif' % i)) as writer:
                writer.append_data(projection, {'compress': 9})

        # Cleanup.
        astra.data3d.delete(projectionId)
        astra.data3d.delete(phantomId)

    def create_reconstructions(self):
        # Set result directories
        input_dir = PROJECTION_RESULTS
        output_dir = RECONSTRUCTION_RESULTS
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)

        projections = np.zeros((self.detectorRows//scale, self.numberOfProjections, self.detectorColumns//scale))
        j = 1
        for i in range(1, self.numberOfProjections+1):
            path = join(input_dir, 'Prep{}.png'.format(i))
            if not os.path.exists(path):
                continue
            im = imread(path).astype(float)
            im /= np.max(im)
            
            h,w = im.shape

            im = cv2.resize(im,(w//scale,h//scale))

            projections[:, j-1, :] = im
            j += 1


        t_in = time.time()

        # Copy projection images into ASTRA Toolbox.
        projectionId = astra.data3d.create('-proj3d', self.projectionGeometry, projections)

        # Create reconstruction.
        reconstruction_id = astra.data3d.create('-vol', self.vol_geom, data=0)

        alg_cfg = astra.astra_dict('SIRT3D_CUDA')
        alg_cfg['ProjectionDataId'] = projectionId
        alg_cfg['ReconstructionDataId'] = reconstruction_id
        algorithm_id = astra.algorithm.create(alg_cfg)
        astra.algorithm.run(algorithm_id, iteration)
        reconstruction = astra.data3d.get(reconstruction_id)
        
        print("reconstruction finished")

        t_recon = time.time()

        # Limit and scale reconstruction.
        reconstruction[reconstruction < 0] = 0
        reconstruction /= np.max(reconstruction)
        reconstruction = np.round(reconstruction * 255).astype(np.uint8)


        # Save reconstruction.
        for i in range(slices):
            im = reconstruction[i, :, :]
            im = np.flipud(im)
            
            # im = im.T
            # im = im[::-1,::-1]
            im = im.T
            
            # im = black_edge(im)
            
            im = cv2.resize(im,(w//scale,h//scale))

            # a = np.max(im)
            # if a >0:
            #     print(i, a)
            imwrite(join(output_dir, 'reco%04d.png' % i), im)
        
        t2 = time.time()
        
        print("overall time:{},input time:{},recon time:{},output time:{}".format(t2-t1,t_in-t1,t_recon-t_in,t2-t_recon))

        # Cleanup.
        astra.algorithm.delete(algorithm_id)
        astra.data3d.delete(reconstruction_id)
        astra.data3d.delete(projectionId)


def black_edge(img):
    
    b = cv2.threshold(img, 3, 255, cv2.THRESH_BINARY)[1] #调整裁剪效果

    edges_y, edges_x = np.where(b==255) ##h, w
    bottom = min(edges_y)             
    top = max(edges_y) 
    height = top - bottom            
                                   
    left = min(edges_x)           
    right = max(edges_x)             
    height = top - bottom 
    width = right - left

    res_image = img[bottom:bottom+height, left:left+width]
    
    return res_image


if __name__ == "__main__":
    import time 
    start_time = time.time()
    recon = Virtual_Cbct()
    recon.start_run()
    end_time = time.time()
    print("recon time", end_time - start_time)
    from find_clearest import find_clearest
    image_folder = "/home/hui/xjs/3Dreconstruction/results/20231207/bga/"
    clearest_id = find_clearest(image_folder)
    print("clearest_id", clearest_id)
    end_time = time.time()
    print("find clearest time", end_time - start_time)
    clearest_id = 35
    from interpolate import interpolate_images
    image_name_list = [f"{image_folder}/reco{(clearest_id-1):04d}.png", 
                    f"{image_folder}/reco{clearest_id:04d}.png",
                    f"{image_folder}/reco{(clearest_id+1):04d}.png"]
    image_list = []
    for i in range(1, -2, -1):
        name = f"{image_folder}/reco{(int(clearest_id-i)):04d}.png"
        image = cv2.imread(name, 0)
        image_list.append(image)
    # import pdb; pdb.set_trace()
    target_spacing = 50
    origin_spacing = 150 / resolution * scale
    # import pdb; pdb.set_trace()
    resampled_images = interpolate_images(image_list, target_spacing, origin_spacing)
    images = []
    from remove_highlight import remove_highlight
    from revert_component import invert_pixels_below_threshold
    for image in resampled_images:
        image = remove_highlight(image)
        # image = invert_pixels_below_threshold(image, 30)
        images.append(image)
    # import pdb; pdb.set_trace()

    for i, image in enumerate(images):
        output_path = os.path.join(f'final_results/interpolated_image_{i}.png')
        cv2.imwrite(output_path, image)

    end_time = time.time()
    print("Time:", end_time-start_time)








