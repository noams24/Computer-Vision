import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import cv2 as cv
from scipy.signal import convolve2d
import scipy.ndimage.filters as filters
from LoG_filter import log_filt
import os

warnings.filterwarnings("ignore")

# constants for Q2:

NUM_OF_PYRAMIDS = 15
SIGMA = 2
K = 2 ** 0.25
MAX_MIN_THRESHOLD = 15
PIC_SIZE = 512


def Q1():
    # a:
    gaussian_matrix = np.random.normal(loc=10, scale=5, size=(100, 100))
    plt.figure()
    plt.imshow(gaussian_matrix, cmap='gray')
    plt.show()

    # b:
    sns.distplot(gaussian_matrix)  # plotting the historgram
    plt.show()

    # c:
    img_path = os.path.join('images', 'messi.jpg')
    image = cv.imread(img_path)
    gray_image = cv.imread('messi.jpg', 0)

    cv.imshow('Original image', image)
    cv.imshow('Gray image', gray_image)
    cv.waitKey(0)
    cv.destroyAllWindows()

    # d:
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    edges1 = cv.Canny(image, 100, 200)
    edges2 = cv.Canny(image, 150, 250)
    edges3 = cv.Canny(image, 200, 300)

    plt.figure(figsize=(9, 9))
    plt.subplot(2, 2, 1), plt.imshow(image)
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 2), plt.imshow(edges1)
    plt.title('Canny Edge 100,200'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 3), plt.imshow(edges2)
    plt.title('Canny Edge 150,250'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 4), plt.imshow(edges3)
    plt.title('Canny Edge 200,300'), plt.xticks([]), plt.yticks([])

    plt.show()

    # e:
    img_path = os.path.join('images', 'damka.jpg')
    img = cv.imread(img_path)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    harris = cv.cornerHarris(gray, 2, 3, 0.04)
    img[harris > 0.01 * harris.max()] = [0, 0, 255]  # Threshold for an optimal value

    img2 = cv.imread(img_path)
    gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    harris = cv.cornerHarris(gray, 3, 5, 0.07)
    img2[harris > 0.01 * harris.max()] = [0, 0, 255]  # Threshold for an optimal value

    cv.imshow('Harris corners with parameter set: 2,3,0.04', img)
    cv.imshow('Harris corners with parameter set: 3,5,0.07', img2)
    cv.waitKey(0)
    cv.destroyAllWindows()


def Q2(picture_name):
    # open and resize the images:
    img_path = os.path.join('images', picture_name)
    image = cv.imread(img_path)
    gray_image = cv.imread(img_path, 0)
    image = cv.resize(image, (PIC_SIZE, PIC_SIZE))
    gray_image = cv.resize(gray_image, (PIC_SIZE, PIC_SIZE))

    # create filters:

    filter_arr = []
    scales_arr = []
    current_scale = SIGMA

    for i in range(NUM_OF_PYRAMIDS):
        filter_size = 2 * np.ceil(3 * current_scale) + 1
        filter = log_filt(ksize=filter_size, sig=current_scale)
        filter *= (current_scale ** 2)  # normalize
        filter_arr.append(filter)
        scales_arr.append(current_scale)
        current_scale *= K

    # create pyramids:

    pyramids = np.zeros((PIC_SIZE, PIC_SIZE, NUM_OF_PYRAMIDS), dtype=float)
    for i, filt in enumerate(filter_arr):
        pyramids[:, :, i] = convolve2d(in1=gray_image, in2=filt, mode='same')

    # finding non maximum supression:

    suppression_diameter = np.median(
        scales_arr)  # chose something that would presumable be adaptive and won't need tuning

    data_max = filters.maximum_filter(pyramids, suppression_diameter)# finding max points
    maxima_mask = np.logical_and((pyramids == data_max), data_max > MAX_MIN_THRESHOLD)
    data_min = filters.minimum_filter(pyramids, suppression_diameter) #finding min points
    minima_mask = np.logical_and((pyramids == data_min), data_min < -MAX_MIN_THRESHOLD)

    true_max_locations = np.where(maxima_mask)
    true_min_locations = np.where(minima_mask)

    # draw blobs with scales on color image
    for maxima_locations_x, maximum_location_y, mask_ind in zip(true_max_locations[1], true_max_locations[0],
                                                                true_max_locations[2]):
        cv.circle(image, (maxima_locations_x, maximum_location_y), int(np.ceil(scales_arr[mask_ind])),
                  (0, 255, 0), 1)

    for minima_locations_x, minimum_location_y, mask_ind in zip(true_min_locations[1], true_min_locations[0],
                                                                true_min_locations[2]):
        cv.circle(image, (minima_locations_x, minimum_location_y), int(np.ceil(scales_arr[mask_ind])),
                  (0, 0, 255), 1)
    cv.imshow("blob detection", image)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':
    Q1()
    pictures = next(os.walk('images'), (None, None, []))[2]
    for picture_name in pictures:
        Q2(picture_name)
