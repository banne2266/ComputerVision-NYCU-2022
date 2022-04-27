import cv2
import numpy as np
import random
import math
import sys

DEBUG = 0
WIDTH = 1008
HEIGHT = 756
START_INDEX = 0
IMG_NUM = 9

# read the image file & output the color & gray image
def read_img(path):
    # opencv read image in BGR color space
    img = cv2.imread(path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, img_gray

# the dtype of img must be "uint8" to avoid the error of SIFT detector
def img_to_gray(img):
    if img.dtype != "uint8":
        print("The input image dtype is not uint8 , image type is : ",img.dtype)
        return
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img_gray

# create a window to show the image
# It will show all the windows after you call im_show()
# Remember to call im_show() in the end of main
def creat_im_window(window_name,img):
    cv2.imshow(window_name,img)

# show the all window you call before im_show()
# and press any key to close all windows
def im_show():
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def get_SIFT_des(image):
    SIFT_Detector = cv2.SIFT_create()
    kp, des = SIFT_Detector.detectAndCompute(image, None)
    return kp, des

def l2_distance(p1, p2):
    p1 = np.array(p1)
    p2 = np.array(p2)
    dis = np.linalg.norm(p1-p2)
    return dis

def knn(target, kps, points, k):
    neighbor_pts = []
    
    for i, pt in enumerate(points):
        distance = l2_distance(target, pt)
        neighbor_pts.append((pt, distance, kps[i]))

    neighbor_pts.sort(key=lambda tup: tup[1])
    neighbor_pts = neighbor_pts[0:k]

    distances = [i[1] for i in neighbor_pts]
    neighbor_pt = [i[0] for i in neighbor_pts]
    neighbor_kps = [i[2] for i in neighbor_pts]
    return neighbor_pt, distances, neighbor_kps

def find_good_match(kp1, kp2, des1, des2, Lowe_ratio = 0.7):
    matches = []
    points_matches = []
    for i, item in enumerate(des1):
        neighbor_pt, distances, neighbor_kps = knn(item, kp2, des2, 2)
        if distances[0] < distances[1] * Lowe_ratio:
            matches.append((item, neighbor_pt[0]))
            points_matches.append((kp1[i], neighbor_kps[0]))

    return matches, points_matches

def find_homography(pts1, pts2):
    a = np.zeros((8, 9), np.float)
    for i in range(4):
        a[i*2][0] = -pts1[i][0]
        a[i*2][1] = -pts1[i][1]
        a[i*2][2] = -1

        a[i*2+1][3] = -pts1[i][0]
        a[i*2+1][4] = -pts1[i][1]
        a[i*2+1][5] = -1

        a[i*2][6] = pts1[i][0] * pts2[i][0]
        a[i*2][7] = pts1[i][1] * pts2[i][0]

        a[i*2+1][6] = pts1[i][0] * pts2[i][1]
        a[i*2+1][7] = pts1[i][1] * pts2[i][1]

        a[i*2][8] = pts2[i][0]
        a[i*2+1][8] = pts2[i][1]

    u, s, vh = np.linalg.svd(a, full_matrices=True)
    homography = vh[-1].reshape(3, 3)
    homography /= homography[2][2]
    return homography


def ransac(matches, iter = 1000):
    match_num = len(matches)
    pt1 = [item[0].pt for item in matches]
    pt2 = [item[1].pt for item in matches]
    
    pt1 = np.array(pt1)
    pt2 = np.array(pt2)
    pt1 = np.c_[ pt1, np.ones(match_num) ]

    best = 0
    best_homography = 0

    for i in range(iter):
        l = []
        while len(l) < 4:
            temp = np.random.randint(match_num)
            if temp not in l:
                l.append(temp)
        homography = find_homography(pt1[l], pt2[l])
        a = homography @ pt1.T

        divide = np.array(a[2], np.float)
        a = a / divide
        a = a[0:2]
        a = a.T

        temp = np.linalg.norm(a-pt2, axis=1)
        temp = temp < 10
        temp = np.sum(temp)

        if temp / match_num > 0.5:
            best_homography = homography
            break
        if temp > best:
            best = temp
            best_homography = homography

    return best_homography
    
    


if __name__ == '__main__':
    # the example of image window
    # creat_im_window("Result",img)
    # im_show()

    # you can use this function to store the result
    # cv2.imwrite("result.jpg",img)
    image_name = ['m1.jpg', 'm2.jpg', 'm3.jpg', 'm4.jpg', 'm5.jpg', 'm6.jpg', 'm7.jpg', 'm8.jpg', 'm9.jpg', 'm10.jpg']
    imgs = []
    img_grays = []

    for f in image_name:
        path = './test/' + f
        img, img_gray = read_img(path)
        imgs.append(img)
        img_grays.append(img_gray)

    size = (WIDTH, HEIGHT)
    
    for i in range(IMG_NUM-1):
        img1 = imgs[START_INDEX] if i == 0 else img2
        img2 = imgs[START_INDEX + 1 + i]
        img_gray1 = img_to_gray(img1)
        img_gray2 = img_to_gray(img2)

        kp0, des0 = get_SIFT_des(img_gray1)
        kp1, des1 = get_SIFT_des(img_gray2)

        bf = cv2.BFMatcher(crossCheck = True)
        matches = bf.match(des0,des1)
        print(len(matches))

        pt1  = np.float32([kp0[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
        pt2  = np.float32([kp1[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
        homography, mask = cv2.findHomography(pt1, pt2, cv2.RANSAC, 5.0)

        '''matches, points_matches = find_good_match(kp0, kp1, des0, des1)
        homography = ransac(points_matches, 50000)'''

        corner = np.array([[0, 0, 1],[0, size[1], 1],[size[0], 0, 1],[size[0], size[1], 1]])
        corner = homography @ corner.T
        corner = corner / np.array(corner[2], np.float)
        x1 = int(min(np.min(corner[0]), 0))
        y1 = int(min(np.min(corner[1]), 0))
        print(x1, y1)
        size = (WIDTH + abs(x1), HEIGHT + abs(y1))
        
        A = np.array([[1, 0 , -x1],  [0, 1, -y1], [0, 0, 1]], np.float)
        homography = A @ homography
        img1 = cv2.warpPerspective(img1, homography, size) 
        
        A = np.array([[1, 0 ,-x1], [0, 1, -y1], [0, 0, 1]], np.float)
        img2 = cv2.warpPerspective(img2, A, size)
        img2 = cv2.copyTo(img1, img1, img2)


    creat_im_window("result", img2)
    im_show()


    cv2.imwrite('result.jpg', img2)


