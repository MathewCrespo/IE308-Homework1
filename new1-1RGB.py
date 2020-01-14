import cv2
import matplotlib.pyplot as plt
import numpy as np
import math


def get_hist (img):     ##get the hist of the image
    a, b = img.shape
    hist = [0] * 256

    for i in range(a):
        for j in range(b):
            hist[int(img[i][j])] += 1
    # normalize
    for i in range(256):
        hist[i] = float(hist[i]) / (a*b)
    return hist

def histmatch(initial, target):  ## regulize the hist
    ini = get_hist(initial)
    tar = get_hist(target)
    tmp1 = 0
    tmp2 = 0
    ini_acc = ini.copy()
    tar_acc = tar.copy()
    for i in range(256):  # get the cfd function
        tmp1 += ini[i]
        ini_acc[i] = tmp1
        tmp2 += tar[i]
        tar_acc[i] = tmp2

    M = np.zeros(256)
    for i in range(256):
        idex = 0
        minv = 1
        for j in range(256):
            if (np.fabs(tar_acc[j] - ini_acc[i]) < minv):
                minv = np.fabs(tar_acc[j] - ini_acc[i])
                idex = int(j)
        M[i] = idex    ##get the funtion of the two hists
    print (M)
    des = M[initial]
    return des


if __name__ == "__main__":   ##main function
    group = [0]*256
    for i in range(256):
        group[i]=i
    img = cv2.imread("F:\\anaconda\\Scripts\\lion.JPG", 1)  # read the image and convert into gray image
    img = np.array(img)  # transform into array
    ima_r = img[:, :, 2]
    ima_g = img[:, :, 1]
    ima_b = img[:, :, 0]

    target= cv2.imread("./car.JPG", 1)
    target = np.array(target)
    target_r = target[:, :, 2]
    target_g = target[:, :, 1]
    target_b = target[:, :, 0]


    de_r = histmatch(ima_r, target_r)
    de_g = histmatch(ima_g, target_g)
    de_b = histmatch(ima_b, target_b)

    final = np.zeros(img.shape)
    final[:, :, 0] = de_b
    final[:, :, 1] = de_g
    final[:, :, 2] = de_r
    print(de_r)
    #cv2.imshow("final", final)
    #cv2.waitKey(10000)
    a= get_hist(ima_r)
    b= get_hist(target_r)
    c= get_hist(de_r)
    plt.figure()
    plt.subplot(1,3,1)
    plt.title("original R")
    plt.bar(group,a)

    plt.subplot(1,3,2)
    plt.title("target R")
    plt.bar(group, b)

    plt.subplot(1,3,3)
    plt.title("after match R")
    plt.bar(group, c)
    plt.show()

    plt.figure()
    plt.imshow(final,cmap='PRGn')
    plt.show()



