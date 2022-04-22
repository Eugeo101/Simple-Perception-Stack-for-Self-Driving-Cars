import shared_data
import cv2
import numpy as np


def overlay_lane_lines():
    """
        output: area within the lanes filled in nonWarped perspective 
                and in the bird-eye view    
    """
    # image to draw the lane lines on Considered as a plank channel
    warp_zero = np.zeros_like(shared_data.warped_frame)

    # Stacking the channels to make a colored image
    colored_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Contains the points that surrounds the lanes
    # The points must be in a specific order so we can fill it
    # Bottom Left -> Top Left -> Top Right -> Bottom Right
    perimeter = []

    # Appending The points of the left lane
    # Bottom Left -> Top Left
    left_pts = []
    for index, value in np.ndenumerate(shared_data.left_fitx):
        left_pts.append([value, index[0]])

    # coloring the left lane line with red
    color_warp = cv2.polylines(
        color_warp, np.int32([np.array(left_pts)]), isClosed=False, color=(0, 0, 255), thickness=20)

    # Appending The points of the right lane
    # Top Right -> Bottom Right, hence the points of the right lane is reversed
    right_pts = []
    for index, value in reversed(list(np.ndenumerate(shared_data.right_fitx))):
        right_pts.append([value, index[0]])

    # coloring the right lane line with blue
    color_warp = cv2.polylines(
        color_warp, np.int32([np.array(right_pts)]), isClosed=False, color=(255, 0, 0), thickness=20)

    # Concatinating then Converting to numpy array
    perimeter = np.array(left_pts + right_pts)

    # Fill the area within the parameter with green
    # Bird-view with colored area
    cv2.fillPoly(colored_warp, np.int_([perimeter]), (0, 255, 0))

    # Transform From Bird-eye view back to the original view
    new_warp = cv2.warpPerspective(
        colored_warp, shared_data.inv_transformation_matrix, shared_data.orig_image_size)

    # Combine the result with the original image, with alpha of the lane area equals .3
    overlayed_frame = cv2.addWeighted(
        shared_data.orig_frame, 1, new_warp, 0.3, 0)

    return overlayed_frame, colored_warp


def calculate_car_position():
    """
        Output: The Offset between car position and the center of the lane
    """
    # The car position is in the center of the frame
    car_position = shared_data.orig_frame.shape[1] / 2

    # the width of the lane in PIXLES
    # Taking the lane width from the left lane and the right lane
    lane_width_px = shared_data.right_fitx[-1] - shared_data.left_fitx[-1]

    # Calculating the Position of the center of the lane
    lane_center_px = lane_width_px / 2 + shared_data.left_fitx[-1]

    # The ratio between Meters and Pixels
    M_PER_PX = shared_data.LANE_WIDTH_M / lane_width_px

    # The Difference between The center of the car and the center of the lane in PIXELS
    diff_car_pos_lane_center_px = car_position - lane_center_px

    # Calculating the offset from the center to the lane in CM
    center_offset = diff_car_pos_lane_center_px * M_PER_PX * 100

    shared_data.center_offset = center_offset

    return center_offset
