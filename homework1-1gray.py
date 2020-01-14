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
    img = cv2.imread("F:\\anaconda\\Scripts\\lion.JPG", 0)  # read the image and convert into gray image
    #cv2.imshow("input image", img)   # to show the original image
    cv2.waitKey(10000)
    img = np.array(img)  # transform into array

    target= cv2.imread("./car.JPG", 0)
    target = np.array(target)
    de = histmatch(img, target)
    plt.figure()
    plt.subplot(2, 3, 1)
    plt.title("original")
    plt.imshow(img,cmap='gray')

    plt.subplot(2, 3, 2)
    plt.title("target")
    plt.imshow(target,cmap='gray')

    plt.subplot(2, 3, 3)
    plt.title("after")
    plt.imshow(de, cmap='gray')

    plt.subplot(2, 3, 4)
    plt.title("original hist")
    hist1=get_hist(img)
    plt.bar(group, hist1)



    plt.subplot(2, 3, 5)
    plt.title("target hist")
    hist2=get_hist(target)
    plt.bar(group, hist2)


    plt.subplot(2, 3, 6)
    plt.title("after hist")
    hist3=get_hist(de)
    plt.bar(group, hist3)
    plt.show()






