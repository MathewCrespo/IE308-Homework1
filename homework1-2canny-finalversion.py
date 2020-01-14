import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

m1 = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
m2 = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])

def standard_canny (image):
    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    edge_output = cv2.Canny(blurred, 50, 150)
    return  edge_output


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

def gra(image):   # calculate gra and angle at each point of the image

    gradM = np.zeros(image.shape, dtype="uint8")
    theta = np.zeros(image.shape, dtype="float")
    img = cv2.copyMakeBorder(image, 1, 1, 1, 1, borderType=cv2.BORDER_REPLICATE)
    rows, cols = img.shape
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            # Gy
            Gy = (np.dot(np.array([1, 1, 1]), (m1 * img[i - 1:i + 2, j - 1:j + 2]))).dot(np.array([[1], [1], [1]]))
            # Gx
            Gx = (np.dot(np.array([1, 1, 1]), (m2 * img[i - 1:i + 2, j - 1:j + 2]))).dot(np.array([[1], [1], [1]]))
            if Gx[0] == 0:
                theta[i - 1, j - 1] = 90
                continue
            else:  # angle transfer into rad
                temp = (np.arctan(Gy[0] / Gx[0])) * 180 / np.pi
            if Gx[0] * Gy[0] > 0:
                if Gx[0] > 0:
                    theta[i - 1, j - 1] = np.abs(temp)
                else:   # negative angle
                    theta[i - 1, j - 1] = (np.abs(temp) - 180)
            if Gx[0] * Gy[0] < 0:
                if Gx[0] > 0:  # negative angle
                    theta[i - 1, j - 1] = (-1) * np.abs(temp)
                else:
                    theta[i - 1, j - 1] = 180 - np.abs(temp)
            gradM[i - 1, j - 1] = (np.sqrt(Gx ** 2 + Gy ** 2))
    for i in range(1, rows - 2):
        for j in range(1, cols - 2):    ## to regularize the angles
            if (((theta[i, j] >= -22.5) and (theta[i, j] < 22.5)) or
                    ((theta[i, j] <= -157.5) and (theta[i, j] >= -180)) or
                    ((theta[i, j] >= 157.5) and (theta[i, j] < 180))):
                theta[i, j] = 0.0
            elif (((theta[i, j] >= 22.5) and (theta[i, j] < 67.5)) or
                  ((theta[i, j] <= -112.5) and (theta[i, j] >= -157.5))):
                theta[i, j] = -45.0
            elif (((theta[i, j] >= 67.5) and (theta[i, j] < 112.5)) or
                  ((theta[i, j] <= -67.5) and (theta[i, j] >= -112.5))):
                theta[i, j] = 90.0
            elif (((theta[i, j] >= 112.5) and (theta[i, j] < 157.5)) or
                  ((theta[i, j] <= -22.5) and (theta[i, j] >= -67.5))):
                theta[i, j] = 45.0
    return  theta, gradM

def nms(img1, theta):
    img2 = np.zeros(img1.shape)
    rows, cols = img2.shape
    # to discuss different situations of the nms
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            if (theta[i, j] == 0.0) and (img1[i, j] == np.max([img1[i, j], img1[i + 1, j], img1[i - 1, j]])):
                img2[i, j] = img1[i, j]

            if (theta[i, j] == -45.0) and img1[i, j] == np.max([img1[i, j], img1[i - 1, j - 1], img1[i + 1, j + 1]]):
                img2[i, j] = img1[i, j]

            if (theta[i, j] == 90.0) and img1[i, j] == np.max([img1[i, j], img1[i, j + 1], img1[i, j - 1]]):
                img2[i, j] = img1[i, j]

            if (theta[i, j] == 45.0) and img1[i, j] == np.max([img1[i, j], img1[i - 1, j + 1], img1[i + 1, j - 1]]):
                img2[i, j] = img1[i, j]
    return  img2

def dual_threshold(img2,low, high):
    img3 = np.zeros(img2.shape)
    rows, cols = img3.shape
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            if img2[i, j] < low:
                img3[i, j] = 0
            elif img2[i, j] > high:
                img3[i, j] = 255
            elif ((img2[i + 1, j] < high) or (img2[i - 1, j] < high) or (img2[i, j + 1] < high) or
                  (img2[i, j - 1] < high) or (img2[i - 1, j - 1] < high) or (img2[i - 1, j + 1] < high) or
                  (img2[i + 1, j + 1] < high) or (img2[i + 1, j - 1] < high)):
                img3[i, j] = 255
    return  img3

if __name__ == "__main__":
    img = cv2.imread("F:\\anaconda\\Scripts\\lion.JPG", 0)
    img = np.array(img)
    new_image = gauss_filter(img, 7, 7)

    theta , M = gra(new_image)

    nms = nms(M, theta)
    dt1 = dual_threshold(nms,50,100)
    dt2 = dual_threshold(nms,75,150)
    dt3 = dual_threshold(nms,30, 60)
    std = standard_canny(img)

    plt.figure()
    plt.subplot(2,2,1)
    plt.title("low=50,high=100")
    plt.imshow(dt1,cmap='gray')
    plt.subplot(2,2,2)
    plt.title("low=75,high=150")
    plt.imshow(dt2,cmap='gray')
    plt.subplot(2,2,3)
    plt.title("low=30,high=60")
    plt.imshow(dt3,cmap='gray')
    plt.subplot(2,2,4)
    plt.title("Standard Canny")
    plt.imshow(std, cmap='gray')
    plt.show()



  #  cv2.imshow("theta", theta)
  #  cv2.imshow("magnitude", M)
  #  cv2.imshow("nms", nms)
  #  cv2.waitKey(4000)




