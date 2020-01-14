import cv2
import matplotlib.pyplot as plt
import numpy as np

import math

def gauss_filter (image, sigma1, sigma2):
    gau_sum = 0
    gauss_func = np.zeros([3, 3])
    for i in range(3):
        for j in range(3):
            gauss_func[i, j] = (1/(2*math.pi*sigma2*sigma1)) * math.exp((-1/(2*sigma1*sigma2))
                                *(np.square(i-2) + np.square(j-2)))
            gau_sum = gau_sum + gauss_func[i, j]

    gauss_func = gauss_func / gau_sum

    a, b = image.shape
    new_image = np.zeros([a-3, b-3])

    for i in range(a-3):
        for j in range(b-3):
            new_image[i, j] = np.sum(np.multiply(image[i:i+3, j:j+3] , gauss_func))

    return  new_image

def gra(new_image):   # calculate gra at each point of the image
    a, b = new_image.shape
    dx = np.zeros([a-1, b-1])
    dy = np.zeros([a-1, b-1])
    Mag = np.zeros([a-1, b-1])

    for i in range(a-1):
        for j in range(b-1):
            dx[i, j] = new_image[i+1, j] - new_image[i, j]  # gra at x direction
            dy[i, j] = new_image[i, j+1] - new_image[i, j]  # gra at y direction
            Mag[i, j] = np.sqrt(np.square(dx[i, j]) + np.square(dy[i, j]))  # magnitude of the gra
    return  dx, dy, Mag

def nms(dx, dy, M):
    a, b = M.shape
    d = np.copy(M)
    nms = np.copy(d)
    nms[0, :] = nms[a-1, :] = nms[:, 0] = nms[:, b-1] = 0
    ## for the edge of the picture

    for i in range(1, a-1):
        for j in range(1, b-1):
            if M[i, j] == 0:
                nms[i, j] = 0
            else:
                gradx = dx[i, j]
                grady = dy[i, j]
                gradM = d[i, j]
                ## gradient y larger than gradient x
                if np.abs(grady) > np.abs(gradx):
                    coe = np.abs(gradx) / np.abs(grady)   #defination of the coeficient
                    grad2 = d[i-1, j]
                    grad4 = d[i+1, j]
                    if gradx * grady > 0:
                        grad1 = d[i - 1, j - 1]
                        grad3 = d[i + 1, j + 1]
                    else:
                        grad1 = d[i - 1, j + 1]
                        grad3 = d[i + 1, j - 1]
                else:  ## gradient x larger than gradient y
                    coe = np.abs(grady) / np.abs(gradx)
                    grad2 = d[i, j-1]
                    grad4 = d[i, j+1]
                    if gradx * grady > 0:
                        grad1 = d[i + 1, j - 1]
                        grad3 = d[i - 1, j + 1]
                    else:
                        grad1 = d[i-1, j-1]
                        grad3 = d[i+1, j+1]
                gradtemp1 = coe * grad1 + (1 - coe) * grad2
                gradtemp2 = coe * grad3 + (1 - coe) * grad4
                if gradM >= gradtemp1 and gradM >= gradtemp2:
                    nms[i, j] = gradM
                else:
                    nms [i, j] = 0
    return  nms

def dual_threshold(nms,high_coe, low_coe):
    a,b =  nms.shape
    edge = np.zeros([a, b])
    high = high_coe * np.max(nms)
    low  = low_coe *  np.max(nms)

    for i in range(1, a-1):
        for j in range(1, b-1):
            if (nms[i, j] < low):
                edge[i, j] = 0
            elif (nms[i, j] > high):    ## obvious edge
                edge[i, j] = 1
            elif (nms[i-1, j-1:j+1] < high).any() or (nms[i+1, j-1:j+1].any()
                        or (nms[i, [j-1, j+1]] < high).any()):
                edge[i,j] = 1    ## connect within the area
    return edge

if __name__ == "__main__":
    img = cv2.imread("F:\\anaconda\\Scripts\\lion.JPG", 0)
    img = np.array(img)
    new_image = gauss_filter(img, 7, 7)
    plt.imshow(new_image, cmap='gray')
    plt.show()
    #blurred = cv2.GaussianBlur(new_image, (3, 3), 0)
    #plt.imshow(blurred, cmap= 'gray')
    #plt.show()

    x, y, M = gra(new_image)
    nms = nms(M, x, y)
    dt = dual_threshold(nms, 0.24, 0.08)
    plt.imshow(dt, cmap='gray')
    plt.show()




