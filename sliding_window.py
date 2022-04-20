# import shared_data

# """
#     To access Shared data:
#         shared_data.orig_frame
# """

import cv2
import numpy as np
import matplotlib.pyplot as plt

def sliding_windows(self):
    
    """
    Using the sliding windows technique, 
    indices of the lane lines pixels are obtained.
    """

    sliding_window_frame = self.warped_frame.copy()

    window_height = np.int(self.warped_frame.shape[0]/self.no_of_windows)

    # x and y coordinates of all white pixels in the frame.
    white = self.warped_frame.nonzero()
    whitey = np.array(white[0])
    whitex = np.array(white[1])

    # To save the lane lines pixel indices
    left_lane_indices = []
    right_lane_indices = []

    # Starting postions for pixel indices for each window.
    leftx_base, rightx_base = self.histogram_peak()
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Windows loop.
    for window in range(self.no_of_windows):

        # Window boumdaries.
        win_y_low = self.warped_frame.shape[0] - (window + 1) * window_height
        win_y_high = self.warped_frame.shape[0] - window * window_height
        win_xleft_low = leftx_current - self.margin
        win_xleft_high = leftx_current + self.margin
        win_xright_low = rightx_current - self.margin
        win_xright_high = rightx_current + self.margin
        cv2.rectangle(sliding_window_frame,(win_xleft_low,win_y_low),(
            win_xleft_high,win_y_high), (255,255,255), 2)
        cv2.rectangle(sliding_window_frame,(win_xright_low,win_y_low),(
            win_xright_high,win_y_high), (255,255,255), 2)

        # Find the white pixels in x and y inside the window.
        good_left_indices = ((whitey >= win_y_low) & (whitey < win_y_high) & 
                            (whitex >= win_xleft_low) & 
                            (whitex < win_xleft_high)).nonzero()[0]
        good_right_indices = ((whitey >= win_y_low) & (whitey < win_y_high) & 
                            (whitex >= win_xright_low) & 
                            (whitex < win_xright_high)).nonzero()[0]

        left_lane_indices.append(good_left_indices)
        right_lane_indices.append(good_right_indices)

        # If you found > minpix pixels, recenter next window on mean position
        if len(good_left_indices) > self.minpix:
            leftx_current = np.int(np.mean(whitex[good_left_indices]))
        if len(good_right_indices) > self.minpix:        
            rightx_current = np.int(np.mean(whitex[good_right_indices]))

    left_lane_indices = np.concatenate(left_lane_indices)
    right_lane_indices = np.concatenate(right_lane_indices)

    leftx = whitex[left_lane_indices]
    lefty = whitey[left_lane_indices] 
    rightx = whitex[right_lane_indices] 
    righty = whitey[right_lane_indices]

    left_fit = None
    right_fit = None
            
    global prev_leftx
    global prev_lefty 
    global prev_rightx
    global prev_righty
    global prev_left_fit
    global prev_right_fit

    if len(leftx)==0 or len(lefty)==0 or len(rightx)==0 or len(righty)==0:
        leftx = prev_leftx
        lefty = prev_lefty
        rightx = prev_rightx
        righty = prev_righty

    # Fitting a second order curve to the coordinates for both lines.
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    prev_left_fit.append(left_fit)
    prev_right_fit.append(right_fit)

    # Calculate the moving average	
    if len(prev_left_fit) > 10:
      prev_left_fit.pop(0)
      prev_right_fit.pop(0)
      left_fit = sum(prev_left_fit) / len(prev_left_fit)
      right_fit = sum(prev_right_fit) / len(prev_right_fit)

    self.left_fit = left_fit
    self.right_fit = right_fit
		
    prev_leftx = leftx
    prev_lefty = lefty 
    prev_rightx = rightx
    prev_righty = righty

    return self.left_fit, self.right_fit