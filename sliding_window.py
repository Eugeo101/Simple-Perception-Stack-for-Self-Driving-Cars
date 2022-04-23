# import shared_data

# """
#     To access Shared data:
#         shared_data.orig_frame
# """

import cv2
import numpy as np
import matplotlib.pyplot as plt

def sliding_windows(self, plot=False):
    
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

    left_lane_inds = ((whitex > (left_fit[0]*(whitey**2) + left_fit[1]*whitey + left_fit[2] - self.margin)) & 
                    (whitex < (left_fit[0]*(whitey**2) + left_fit[1]*whitey + left_fit[2] + self.margin))) 
    right_lane_inds = ((whitex > (right_fit[0]*(whitey**2) + right_fit[1]*whitey + right_fit[2] - self.margin)) & 
                    (whitex < (right_fit[0]*(whitey**2) + right_fit[1]*whitey + right_fit[2] + self.margin))) 			
    self.left_lane_inds = left_lane_inds
    self.right_lane_inds = right_lane_inds

    leftx = whitex[left_lane_inds]
    lefty = whitey[left_lane_inds] 
    rightx = whitex[right_lane_inds]
    righty = whitey[right_lane_inds]	

    global prev_leftx2
    global prev_lefty2 
    global prev_rightx2
    global prev_righty2
    global prev_left_fit2
    global prev_right_fit2

    if len(leftx)==0 or len(lefty)==0 or len(rightx)==0 or len(righty)==0:
      leftx = prev_leftx2
      lefty = prev_lefty2
      rightx = prev_rightx2
      righty = prev_righty2

    self.leftx = leftx
    self.rightx = rightx
    self.lefty = lefty
    self.righty = righty
		
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    prev_left_fit2.append(left_fit)
    prev_right_fit2.append(right_fit)

    # Calculate the moving average	
    if len(prev_left_fit2) > 10:
      prev_left_fit2.pop(0)
      prev_right_fit2.pop(0)
      left_fit = sum(prev_left_fit2) / len(prev_left_fit2)
      right_fit = sum(prev_right_fit2) / len(prev_right_fit2)

    self.left_fit = left_fit
    self.right_fit = right_fit
		
    prev_leftx2 = leftx
    prev_lefty2 = lefty 
    prev_rightx2 = rightx
    prev_righty2 = righty

    # Creating y and x values of the plot.
    ploty = np.linspace(0, sliding_window_frame.shape[0]-1, sliding_window_frame.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    self.ploty = ploty
    self.left_fitx = left_fitx
    self.right_fitx = right_fitx

    if plot==True:
		
      # Result image
      out_img = np.dstack((sliding_window_frame, sliding_window_frame, (sliding_window_frame))) * 255
			
      window_image = np.zeros_like(out_img)

      # Left line in red and right one in blue.
      out_img[whitey[left_lane_indices], whitex[left_lane_indices]] = [255, 0, 0]
      out_img[whitey[right_lane_indices], whitex[right_lane_indices]] = [0, 0, 255]

      # Show the area of the search window
      # Creating the usable format for the fillpoly function.

      left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-self.margin, ploty]))])
      left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+self.margin, ploty])))])
      left_line_pts = np.hstack((left_line_window1, left_line_window2))
      right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-self.margin, ploty]))])
      right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+self.margin, ploty])))])
      right_line_pts = np.hstack((right_line_window1, right_line_window2))

      cv2.fillPoly(window_image, np.int_([left_line_pts]), (0,255, 0))
      cv2.fillPoly(window_image, np.int_([right_line_pts]), (0,255, 0))
      result = cv2.addWeighted(out_img, 1, window_image, 0.3, 0)

      # Plot the figure with the sliding windows and the detected lanes.
      figure, axes = plt.subplots(3,2)
      figure.set_size_inches(10, 10)
      figure.tight_layout(pad=3.0)
      axes[0][1].set_visible(False)
      axes[0][0].imshow(cv2.cvtColor(self.orig_frame, cv2.COLOR_BGR2RGB))
      axes[1][0].imshow(sliding_window_frame, cmap='gray')
      axes[2][0].imshow(out_img)
      axes[2][0].plot(left_fitx, ploty, color='yellow')
      axes[2][0].plot(right_fitx, ploty, color='yellow')
      axes[1][1].imshow(self.warped_frame, cmap='gray')
      axes[2][1].imshow(result)
      axes[2][1].plot(left_fitx, ploty, color='yellow')
      axes[2][1].plot(right_fitx, ploty, color='yellow')
      axes[0][0].set_title("Original Frame")  
      axes[1][0].set_title("Warped Frame with Sliding Windows")
      axes[2][0].set_title("Detected Lane Lines with Sliding Windows")
      axes[1][1].set_title("Warped Frame")
      axes[2][1].set_title("Warped Frame With Search Window")
      plt.show()
