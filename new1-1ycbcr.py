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

    ima_y = 0.256789 * ima_r + 0.504129 * ima_g + 0.097906 * ima_b + 16
    ima_cb = -0.148223 * ima_r - 0.290992 * ima_g + 0.439215 * ima_b + 128
    ima_cr = 0.439215 * ima_r - 0.367789 * ima_g - 0.071426 * ima_b + 128
    ima_y = ima_y.astype(np.int)
    ima_cb = ima_cb.astype(np.int)
    ima_cr = ima_cr.astype(np.int)

    target = cv2.imread("./car.JPG", 1)
    target = np.array(target)
    target_r = target[:, :, 2]
    target_g = target[:, :, 1]
    target_b = target[:, :, 0]
    tar_y = 0.256789 * target_r + 0.504129 * target_g + 0.097906 * target_b + 16
    tar_cb = -0.148223 * target_r - 0.290992 * target_g + 0.439215 * target_b + 128
    tar_cr = 0.439215 * target_r - 0.367789 * target_g - 0.071426 * target_b + 128
    tar_y = tar_y.astype(np.int)
    tar_cb = tar_cb.astype(np.int)
    tar_cr = tar_cr.astype(np.int)



    de_y = histmatch(ima_y, tar_y)
    de_cb = histmatch(ima_cb, tar_cb)
    de_cr = histmatch(ima_cr, tar_cr)

    a = get_hist(ima_y)
    b = get_hist(tar_y)
    c = get_hist(de_y)


    final = np.zeros(img.shape)
    final[:, :, 0] = 1.164383 * (de_y - 16) + 1.596027 * (de_cr - 128)
    final[:, :, 1] = 1.164383 * (de_y - 16) - 0.391762 * (de_cb - 128) - 0.812969 * (de_cr - 128)
    final[:, :, 2] = 1.164383 * (de_y - 16) + 2.017230 * (de_cb - 128)
    final = final.astype(np.int)
    plt.figure()
    plt.subplot(1,3,1)
    plt.title("original")
    plt.bar(group,a)
    plt.subplot(1, 3, 2)
    plt.title("target")
    plt.bar(group, b)
    plt.subplot(1, 3, 3)
    plt.title("after")
    plt.bar(group, c)
    plt.show()

    plt.imshow(final,cmap='PRGn')
    plt.show()



