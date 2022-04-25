import shared_data
import cv2
import matplotlib.pyplot as plt
import numpy as np

def show_image(image, title= "Image", cmap_type = 'gray'):
    plt.imshow(image, cmap_type)
    plt.title(title)
#     plt.axis("off")

def show_image_BGR(image, title= "Image", cmap_type = 'gray'):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image, cmap_type)
    plt.title(title)
#     plt.axis("off")

def plot_ROI(frame = None, plot = False):
    """
    input: image
    description: using cv2.polylines
    output: get image with ROI
    """
    if frame == None:
        frame = shared_data.orig_frame.copy()
    if plot == True:
        #cv2.polylines(image, [pts], isClosed, color, thickness)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        roi_img = cv2.polylines(frame, np.int32([shared_data.roi_points]), True, (217, 14, 23), 3)
        plt.imshow(roi_img)
        plt.title("ROI Image")
        plt.axis("off")
    else:
        return

def Wrap_Presspective(frame = None, plot = False):
    """
    input: binary image
    description: wrap image ot birdview
    output: image with lanes only
    """
    if frame is None:
        frame = shared_data.orig_frame.copy()
        shared_data.transformation_matrix = cv2.getPerspectiveTransform(shared_data.src, shared_data.dist)
        shared_data.inverse_transformation_matrix = cv2.getPerspectiveTransform(shared_data.dist, shared_data.src)
        shared_data.warped_frame = cv2.warpPerspective(frame, shared_data.transformation_matrix, frame.shape[1::-1], flags=(cv2.INTER_LINEAR))

    if plot == True:
        copy_wraped = shared_data.warped_frame.copy()
        plot_ROI(src=shared_data.dist, frame= copy_wraped, plot = True)
        
    return shared_data.warped_frame


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
        value = 1

    # If value == 0, make all values in binary equal to 0 if the
    # corresponding value in the input array is between the threshold
    # (inclusive). Otherwise, the value remains as 1. Therefore, the pixels
    # with the high Sobel derivative values (i.e. sharp pixel intensity
    # discontinuities) will have 0 in the corresponding cell of binary.
    binary[(array >= thresh[0]) & (array <= thresh[1])] = value

    return binary


def mag_thresh(image, sobel_kernel=3, thresh=(0, 255), value = 1):
    """
    Implementation of Sobel edge detection

    :param image: 2D or 3D array to be blurred
    :param sobel_kernel: Size of the small matrix (i.e. kernel)
                         i.e. number of rows and columns
    :return: Binary (black and white) 2D mask image
    """
    # Get the magnitude of the edges that are vertically aligned on the image
    #sobel = cv2.Sobel(img_channel, cv2.CV_64F, 1, 0, sobel_kernel)
    sobelx = np.absolute(cv2.Sobel(image, cv2.CV_64F, 1, 0, sobel_kernel)) #

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

def canny_detection(frame = None, thresh = (0, 255), plot = False):
    """
    using canny and thershold will make it binary then we make wrap prespective
    output: Edges of image in binary Image
    """
    if frame == None:
        frame = shared_data.orig_frame
    edges = cv2.Canny(frame, thresh[0], thresh[1])
    if plot == True:
        copy_edges = edges.copy()
        show_image(frame, "canny edges")
    return edges


def Preprocessing(frame = None, plot = False):
    """
    input: orignal image
    descripiton: Contrast - Color - GaussianBlur
    output: image with yellow and white lines only
    """
    from skimage import morphology
#     if frame == None:
#         frame = shared_data.orig_frame
    hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
    
    # define range of yellow color in HSV
    lower_yellow = np.array([20,50,50])
    upper_yellow = np.array([40,255,255])
    # Threshold the HLS image to get only yellow colors
    mask_yellow = cv2.inRange(hls, lower_yellow, upper_yellow)
    # Bitwise-AND mask and original image
    result_yellow = cv2.bitwise_and(frame, frame, mask= mask_yellow)        

    b, g, r = cv2.split(result_yellow)
    result_yellow = g
    _, result_yellow = cv2.threshold(result_yellow, 120, 255, cv2.THRESH_BINARY)
    

    lower_white = np.array([0,0,255])
    upper_white = np.array([180,255,255])
    # Threshold the HLS image to get only yellow colors
    mask_white = cv2.inRange(hls, lower_white, upper_white)
    # Bitwise-AND mask and original image
    result_white = cv2.bitwise_and(frame, frame, mask= mask_white)


    b, g, r = cv2.split(result_white)
    result_white = g
    _, result_white = cv2.threshold(result_white, 120, 255, cv2.THRESH_BINARY)
    
    result_yw = cv2.bitwise_or(result_white, result_yellow)
    
    if plot == True:
        show_image_BGR(frame, 'orignal frame')
        plt.show()
    #         show_image(mask_yellow, 'yellow mask')
    #         plt.show()
    #         show_image(result_yellow, 'yellow of orignal frame')
    #         plt.show()
        show_image(result_yellow, 'yellow thersholded of orignal frame')
        plt.show()
    #         show_image(mask_white, 'white mask')
    #         plt.show()
    #         show_image(result_white, 'white of orignal frame')
    #         plt.show()
        show_image(result_white, 'white thersholded of orignal frame')
        plt.show()
        show_image(result_yw, 'yellow or white result')
        plt.show()
    ###############################yellow or white image is output now get lane defination#################################
    

    ################### Isolate possible lane line edges ######################
    l_channel = hls[:, :, 1]
    
    if plot == True:
        show_image(l_channel, 'L_channel')
        plt.show()

    _, L_channel_binary = cv2.threshold(l_channel, 120, 255, cv2.THRESH_BINARY)
    
    if plot == True:
        show_image(L_channel_binary, 'binary L_channel')
        plt.show()

    L_channel_binary = cv2.GaussianBlur(L_channel_binary, (3, 3), 0)  # Reduce noise
    
    if plot == True:
        show_image(L_channel_binary, "gaussian Binary L_channel")
        plt.show()


    # 1s will be in the cells with the highest Sobel derivative values
    # (i.e. strongest lane line edges)
    # L_edges_binary = mag_thresh(L_channel_binary, sobel_kernel=3, thresh=(110, 255), value = 1)
    L_edges_binary = canny_detection(L_channel_binary, thresh = (110, 230))
    
    if plot == True:
        show_image(L_edges_binary, 'Edges L_channel Thersholded')
        plt.show()

    #####################################remove noise of binary image by open morphology#####################################
#     kernel = np.ones((2, 2), np.uint8)
#     opened_edges = cv2.morphologyEx(L_edges_binary, cv2.MORPH_OPEN, kernel)
#     show_image(opened_edges, "Morphology")
#     plt.show()

    # for i in range(5):
    #     opened_edges = morphology.binary_erosion(opened_edges)
    #     show_image(opened_edges, i)
    #     plt.show()
    
    # assumed lane lines are pure color (solid white, solid yellow)
    # White in the regions with the purest hue colors (e.g. >80...play with
    # this value for best results).
    s_channel = hls[:, :, 2]  # use only the saturation channel data
    
    if plot == True:
        show_image(s_channel, 's channel')
        plt.show()

    _, s_binary = cv2.threshold(s_channel, 80, 255, cv2.THRESH_BINARY)
    
    if plot == True:
        show_image(s_binary, 'S Binary Thersholded')
        plt.show()

    # White in the regions with the richest red channel values (e.g. >120).
    # pure white is bgr(255, 255, 255), Pure yellow is bgr(0, 255, 255).
    # Both have high red and green channel values.

    b, g, r = cv2.split(frame)
   
    if plot == True:
        show_image(g, 'G channel')
        plt.show()

    _, g_thresh = cv2.threshold(g, 120, 255, cv2.THRESH_BINARY)
    
    if plot == True:
        show_image(g_thresh, 'G Binary Thersholded')
        plt.show()

    # Lane lines should be pure in color and have high red channel values
    # Bitwise AND operation to reduce noise and black-out any pixels that
    # don't appear to be nice, pure, solid colors (like white or yellow lane
    # lines.)
    gs_binary = cv2.bitwise_and(s_binary, g_thresh)

    if plot == True:
        show_image(gs_binary, 'G & S Binary Thersholded')
        plt.show()

    ### Combine the possible lane lines with the possible lane line edges #####
    # If you show rs_binary visually, you'll see that it is not that different
    # from this return value. The edges of lane lines are thin lines of pixels.


    lane_line_markings = cv2.bitwise_or(gs_binary, L_edges_binary.astype(np.uint8))

    if plot == True:
        show_image(lane_line_markings, "final image")
        plt.show()
    
    result_image = cv2.bitwise_and(lane_line_markings, result_yw)
    if plot == True:
        show_image(result_image)
    
    shared_data.lane_line_markings = lane_line_markings
    return lane_line_markings
