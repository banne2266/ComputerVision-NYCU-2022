import cv2
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import math
from scipy.sparse import csr_matrix, diags
import scipy
from scipy.sparse.linalg import svds
from sqlalchemy import true

image_row = 0 
image_col = 0

# visualizing the mask (size : "image width" * "image height")
def mask_visualization(M):
    mask = np.copy(np.reshape(M, (image_row, image_col)))
    plt.figure()
    plt.imshow(mask, cmap='gray')
    plt.title('Mask')

# visualizing the unit normal vector in RGB color space
# N is the normal map which contains the "unit normal vector" of all pixels (size : "image width" * "image height" * 3)
def normal_visualization(N):
    # converting the array shape to (w*h) * 3 , every row is a normal vetor of one pixel
    N_map = np.copy(np.reshape(N, (image_row, image_col, 3)))
    # Rescale to [0,1] float number
    N_map = (N_map + 1.0) / 2.0
    plt.figure()
    plt.imshow(N_map)
    plt.title('Normal map')

# visualizing the depth on 2D image
# D is the depth map which contains "only the z value" of all pixels (size : "image width" * "image height")
def depth_visualization(D):
    D_map = np.copy(np.reshape(D, (image_row,image_col)))
    # D = np.uint8(D)
    plt.figure()
    plt.imshow(D_map)
    plt.colorbar(label='Distance to Camera')
    plt.title('Depth map')
    plt.xlabel('X Pixel')
    plt.ylabel('Y Pixel')

# convert depth map to point cloud and save it to ply file
# Z is the depth map which contains "only the z value" of all pixels (size : "image width" * "image height")
def save_ply(Z,filepath):
    Z_map = np.reshape(Z, (image_row,image_col)).copy()
    data = np.zeros((image_row*image_col,3),dtype=np.float32)
    # let all point float on a base plane 
    baseline_val = np.min(Z_map)
    Z_map[np.where(Z_map == 0)] = baseline_val
    for i in range(image_row):
        for j in range(image_col):
            idx = i * image_col + j
            data[idx][0] = j
            data[idx][1] = i
            data[idx][2] = Z_map[image_row - 1 - i][j]
    # output to ply file
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data)
    o3d.io.write_point_cloud(filepath, pcd,write_ascii=True)

# show the result of saved ply file
def show_ply(filepath):
    pcd = o3d.io.read_point_cloud(filepath)
    o3d.visualization.draw_geometries([pcd])

# read the .bmp file
def read_bmp(filepath):
    global image_row
    global image_col
    image = cv2.imread(filepath,cv2.IMREAD_GRAYSCALE)
    image_row , image_col = image.shape
    return image


def read_light_source(filepath):
    ret = np.zeros((6, 3), np.float)
    with open(filepath, "r") as infile:
        for i, line in enumerate(infile):
            str = line[7:-2]
            sum = 0.0
            for j, item in enumerate(str.split(',')):
                sum += float(item) ** 2
            sum = math.sqrt(sum)
            for j, item in enumerate(str.split(',')):
                ret[i][j] = float(item) / sum
    return ret



if __name__ == '__main__':
    files = ['bunny', 'star', 'venus']
    for name in files:
        imgs = []
        for i in range(1, 7):
            path = './test/' + name + '/pic' + str(i) + '.bmp'
            img = read_bmp(path)
            img = np.reshape(img, (image_row * image_col))
            imgs.append(img)
        I = np.array(imgs)
        L = read_light_source('./test/' + name + '/LightSource.txt')

        I = np.asmatrix(I)
        L = np.asmatrix(L)
        KdN = (L.T * L).I * L.T * I
        
        N = np.array(KdN, np.float)
        nor = np.linalg.norm(N, ord=2, axis=0)
        for idx in range(nor.shape[0]):
            if nor[idx] == 0:
                nor[idx] = 1
        
        N = N / nor
        N = N.T
        N = N.reshape((image_row , image_col, 3))
        normal_visualization(N)
        plt.show()

        z = np.zeros((image_row , image_col), np.float)
        for i in range(image_row):
            for j in range(image_col):
                if (i == 0 and j == 0) or (N[i][j][2] == 0 and N[i][j][1] == 0 and N[i][j][0] == 0):
                    z[i][j] = 0
                elif i == 0:
                    z[i][j] = z[i][j-1] + (-N[i][j][0] / N[i][j][2])
                elif j == 0:
                    z[i][j] = z[i-1][j] + (N[i][j][1] / N[i][j][2])
                else:
                    z[i][j] = (z[i-1][j] + (N[i][j][1] / N[i][j][2]) + z[i][j-1] + (-N[i][j][0] / N[i][j][2])) / 2
        
        z1 = np.zeros((image_row , image_col), np.float)
        for i in range(image_row-1, -1, -1):
            for j in range(image_col-1, -1, -1):
                if (i == image_row-1 and j == image_col-1) or (N[i][j][2] == 0 and N[i][j][1] == 0 and N[i][j][0] == 0):
                    z1[i][j] = 0
                elif i == 0:
                    z1[i][j] = z1[i][j+1] + (N[i][j][0] / N[i][j][2])
                elif j == image_col-1:
                    z1[i][j] = z1[i+1][j] + (-N[i][j][1] / N[i][j][2])
                else:
                    z1[i][j] = (z1[i+1][j] + (-N[i][j][1] / N[i][j][2]) + z1[i][j+1] + (N[i][j][0] / N[i][j][2])) / 2
        
        z2 = np.zeros((image_row , image_col), np.float)
        for i in range(image_row-1, -1, -1):
            for j in range(image_col):
                if (i == image_row-1 and j == 0) or (N[i][j][2] == 0 and N[i][j][1] == 0 and N[i][j][0] == 0):
                    z2[i][j] = 0
                elif i == 0:
                    z2[i][j] = z2[i][j-1] + (-N[i][j][0] / N[i][j][2])
                elif j == 0:
                    z2[i][j] = z2[i+1][j] + (-N[i][j][1] / N[i][j][2])
                else:
                    z2[i][j] = (z2[i+1][j] + (-N[i][j][1] / N[i][j][2]) + z2[i][j-1] + (-N[i][j][0] / N[i][j][2])) / 2

        z3 = np.zeros((image_row , image_col), np.float)
        for i in range(image_row):
            for j in range(image_col-1, -1, -1):
                if (i == 0 and j == image_col-1) or (N[i][j][2] == 0 and N[i][j][1] == 0 and N[i][j][0] == 0):
                    z3[i][j] = 0
                elif i == 0:
                    z3[i][j] = z3[i][j+1] + (N[i][j][0] / N[i][j][2])
                elif j == 0:
                    z3[i][j] = z3[i-1][j] + (N[i][j][1] / N[i][j][2])
                else:
                    z3[i][j] = (z3[i-1][j] + (N[i][j][1] / N[i][j][2]) + z3[i][j+1] + (N[i][j][0] / N[i][j][2])) / 2

        z = (z + z1 + z2 + z3) / 4 #integrate from 4 corners and average them

        depth_visualization(z)
        plt.show()

        save_path = './ply/' + name + '-1.ply'
        save_ply(z, save_path)
        show_ply(save_path)

        row = []
        col = []
        data = []
        S = image_row*image_col
        for i in range(image_row):
            for j in range(image_col):
                if j + 1 < image_col:
                    row.append(i * image_col + j)
                    col.append(i * image_col + j + 1)
                    data.append(1)

                    row.append(i * image_col + j)
                    col.append(i * image_col + j)
                    data.append(-1)

                if i + 1 < image_row:
                    row.append(i * image_col + j + S)
                    col.append((i+1) * image_col + j)
                    data.append(1)

                    row.append(i * image_col + j + S)
                    col.append(i * image_col + j)
                    data.append(-1)

        M = csr_matrix((data, (row, col)), shape=(S*2, S))

        V = np.zeros((S*2), np.float)
        for i in range(image_row):
            for j in range(image_col):
                    V[i * image_col + j] = (-N[i][j][0] / N[i][j][2]) if N[i][j][2] != 0 else 0
                    V[i * image_col + j + S] = (N[i][j][1] / N[i][j][2]) if N[i][j][2] != 0 else 0

        z = scipy.sparse.linalg.lsmr(M, V)[0]
        z = np.reshape(z, (image_row, image_col))
        for i in range(image_row):
            for j in range(image_col):
                if (i == 0 and j == 0) or (N[i][j][2] == 0 and N[i][j][1] == 0 and N[i][j][0] == 0):
                    z[i][j] = 0
        depth_visualization(z)
        plt.show()

        save_path = './ply/' + name + '-2.ply'
        save_ply(z, save_path)
        show_ply(save_path)

