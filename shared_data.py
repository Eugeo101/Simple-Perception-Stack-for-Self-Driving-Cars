import numpy as np
import cv2

# Holds the path to the image 
# The value is just for testing
orig_frame_path = "./tests/test_images/test2.jpg"

# Holds the Image 
orig_frame = cv2.imread(orig_frame_path)

# This will hold an image with the lane lines
lane_line_markings = None

# This will hold the image after perspective transformation
warped_frame = None
transformation_matrix = None
inv_transformation_matrix = None

# (Width, Height) of the original video frame (or image)
orig_image_size = orig_frame.shape[::-1][1:]

width = orig_image_size[0]
height = orig_image_size[1]
width = width
height = height

# Four corners of the trapezoid-shaped region of interest
# You need to find these corners manually.
roi_points = np.float32([
    (562,  441),  # Top-left corner
    (310, 677),  # Bottom-left corner
    (1079, 670),  # Bottom-right corner
    (665, 442)  # Top-right corner
])

# The desired corner locations  of the region of interest
# after we perform perspective transformation.
# Assume image width of 600, padding == 150.
# padding from side of the image in pixels
padding = int(0.25 * width)
desired_roi_points = np.float32([
    [padding, 0],  # Top-left corner
    [padding, orig_image_size[1]],  # Bottom-left corner
    [orig_image_size[
        0]-padding, orig_image_size[1]],  # Bottom-right corner
    [orig_image_size[0]-padding, 0]  # Top-right corner
])

# Histogram that shows the white pixel peaks for lane line detection
histogram = None

# Sliding window parameters
no_of_windows = 10
margin = int((1/12) * width)  # Window width is +/- margin
# Min no. of pixels to recenter window
minpix = int((1/24) * width)

# Best fit polynomial lines for left line and right line of the lane
left_fit = None
right_fit = None
left_lane_inds = None
right_lane_inds = None
ploty = None
left_fitx = None
right_fitx = None
leftx = None
rightx = None
lefty = None
righty = None

# Pixel parameters for x and y dimensions
YM_PER_PIX = 10.0 / 1000  # meters per pixel in y dimension
XM_PER_PIX = 3.7 / 781  # meters per pixel in x dimension

# Radii of curvature and offset
left_curvem = None
right_curvem = None
center_offset = None
