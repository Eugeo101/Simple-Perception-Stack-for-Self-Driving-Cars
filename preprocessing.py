import cv2
import matplotlib.pyplot as plt
import numpy as np


def show_image(image, title="Image", cmap_type='gray'):
    plt.imshow(image, cmap_type)
    plt.title(title)
#     plt.axis("off")


def show_image_BGR(image, title="Image", cmap_type='gray'):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image, cmap_type)
    plt.title(title)
#     plt.axis("off")


def binary_array(array, thresh, value=0):
    """
    Return a 2D binary array (mask) in which all pixels are either 0 or 1

    :param array: NumPy 2D array that we want to convert to binary values
    :param thresh: Values used for thresholding (inclusive)
    :param value: Output value when between the supplied threshold
    :return: Binary 2D array...
             number of rows x number of columns =
             number of pixels from top to bottom x number of pixels from
               left to right
    """
    if value == 0:
        # Create an array of ones with the same shape and type as
        # the input 2D array.
        binary = np.ones_like(array)

    else:
        # Creates an array of zeros with the same shape and type as
        # the input 2D array.
        binary = np.zeros_like(array)
        value = 255

    # If value == 0, make all values in binary equal to 0 if the
    # corresponding value in the input array is between the threshold
    # (inclusive). Otherwise, the value remains as 1. Therefore, the pixels
    # with the high Sobel derivative values (i.e. sharp pixel intensity
    # discontinuities) will have 0 in the corresponding cell of binary.
    binary[(array >= thresh[0]) & (array <= thresh[1])] = value

    return binary


def mag_thresh(image, sobel_kernel=3, thresh=(0, 255), value=1):
    """
    Implementation of Sobel edge detection

    :param image: 2D or 3D array to be blurred
    :param sobel_kernel: Size of the small matrix (i.e. kernel)
                         i.e. number of rows and columns
    :return: Binary (black and white) 2D mask image
    """
    # Get the magnitude of the edges that are vertically aligned on the image
    #sobel = cv2.Sobel(img_channel, cv2.CV_64F, 1, 0, sobel_kernel)
    sobelx = np.absolute(cv2.Sobel(image, cv2.CV_64F, 1, 0, sobel_kernel))

    # Get the magnitude of the edges that are horizontally aligned on the image
    sobely = np.absolute(cv2.Sobel(image, cv2.CV_64F, 1, 0, sobel_kernel))

    # Find areas of the image that have the strongest pixel intensity changes
    # in both the x and y directions. These have the strongest gradients and
    # represent the strongest edges in the image (i.e. potential lane lines)
    # mag is a 2D array .. number of rows x number of columns = number of pixels
    # from top to bottom x number of pixels from left to right
    mag = np.sqrt(sobelx ** 2 + sobely ** 2)

    # Return a 2D array that contains 0s and 1s
    return binary_array(mag, thresh, value)


def canny_detection(frame, thresh=(0, 255), plot=False):
    """
    using canny and thershold will make it binary then we make wrap prespective
    output: Edges of image in binary Image
    """
    edges = cv2.Canny(frame, thresh[0], thresh[1])
    if plot == True:
        copy_edges = edges.copy()
        show_image(frame, "canny edges")
    return edges
