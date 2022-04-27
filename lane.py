import cv2  # Import the OpenCV library to enable computer vision
import numpy as np  # Import the NumPy scientific computing library
import preprocessing as pre  # Handles the detection of lane lines
import matplotlib.pyplot as plt  # Used for plotting and error checking


# Global variables
prev_leftx = None
prev_lefty = None
prev_rightx = None
prev_righty = None
prev_left_fit = []
prev_right_fit = []

prev_leftx2 = None
prev_lefty2 = None
prev_rightx2 = None
prev_righty2 = None
prev_left_fit2 = []
prev_right_fit2 = []

total_curvature = 0
avg_curvature = 0
total_offset = 0
avg_offset = 0
frame_count = 0

filename = None

class Lane:
    """
    Represents a lane on a road.
    """

    def __init__(self, orig_frame):
        """
              Default constructor

        :param orig_frame: Original camera image (i.e. frame)
        """
        self.orig_frame = orig_frame

        # This will hold an image with the lane lines
        self.lane_line_markings = None

        # This will hold the image after perspective transformation
        self.warped_frame = None
        self.transformation_matrix = None
        self.inv_transformation_matrix = None

        # (Width, Height) of the original video frame (or image)
        self.orig_image_size = self.orig_frame.shape[::-1][1:]

        width = self.orig_image_size[0]
        height = self.orig_image_size[1]
        self.width = width
        self.height = height

        # Four corners of the trapezoid-shaped region of interest
        # You need to find these corners manually.
        self.roi_points = np.float32([
            (600, 450),
            (285, 655),
            (1092, 636),
            (756, 450)
        ])
        self.roi_img = None
        # The desired corner locations  of the region of interest
        # after we perform perspective transformation.
        # Assume image width of 600, padding == 150.
        # padding from side of the image in pixels
        self.padding = int(0.25 * width)
        self.desired_roi_points = np.float32([
            [self.padding, 0],  # Top-left corner
            [self.padding, self.orig_image_size[1]],  # Bottom-left corner
            [self.orig_image_size[
                0]-self.padding, self.orig_image_size[1]],  # Bottom-right corner
            [self.orig_image_size[0]-self.padding, 0]  # Top-right corner
        ])

        # Histogram that shows the white pixel peaks for lane line detection
        self.histogram = None

        # Sliding window parameters
        self.no_of_windows = 10
        self.margin = int((1/12) * width)  # Window width is +/- margin
        # Min no. of pixels to recenter window
        self.minpix = int((1/24) * width)

        self.s_threshold = 80

        if (filename == 'challenge_video.mp4'):
            self.roi_points = np.float32([(639, 473),  # Top-left corner
                                          (327, 686),  # Bottom-left corner
                                          (1062, 690),  # Bottom-right corner
                                          (738, 474)  # Top-right corner
                                          ])  # challenge
            self.s_threshold = 10
            self.margin = int((1/48) * width)
        if (filename == 'project_video.mp4'):
            self.roi_points = np.float32([(562, 441),  # Top-left corner
                                          (310, 677),  # Bottom-left corner
                                          (1079, 670),  # Bottom-right corner
                                          (665, 442)  # Top-right corner
                                          ])  # project
        if (filename == 'harder_challenge_video.mp4'):
            self.roi_points = np.float32([(592, 500),  # Top-left corner
                                          (319, 684),  # Bottom-left corner
                                          (1048, 685),  # Bottom-right corner
                                          (768, 498)  # Top-right corner
                                          ])  # hard
            self.s_threshold = 150

        # Best fit polynomial lines for left line and right line of the lane
        self.left_fit = None
        self.right_fit = None
        self.left_lane_inds = None
        self.right_lane_inds = None
        self.ploty = None
        self.left_fitx = None
        self.right_fitx = None
        self.leftx = None
        self.rightx = None
        self.lefty = None
        self.righty = None

        # contains Thresholded Warped Lanes with fitted lines overlayed over them
        self.warped_with_lines = None
        self.sliding_window_frame = None
        # Pixel parameters for x and y dimensions
        self.YM_PER_PIX = 7.0 / 400  # meters per pixel in y dimension
        self.XM_PER_PIX = 3.7 / 255  # meters per pixel in x dimension

        # the width of lane meters
        self.LANE_WIDTH_M = 3

        # Radii of curvature and offset
        self.left_curvem = None
        self.right_curvem = None
        self.center_offset = None
        self.curvature = None

    def preProcessing(self, frame=None, plot=False):
        """
        input: orignal image
        descripiton: Contrast - Color - GaussianBlur
        output: image with yellow and white lines only
        """
        if frame is None:
            frame = self.orig_frame
        hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)

        if plot == True:
            pre.show_image_BGR(frame, 'orignal frame')
            plt.show()

        # define range of yellow color in HLS
        lower_yellow = np.array([20, 102, 60])
        upper_yellow = np.array([40, 255, 130])
        # Threshold the HLS image to get only yellow colors
        mask_yellow = cv2.inRange(hls, lower_yellow, upper_yellow)
        if plot == True:
            pre.show_image(mask_yellow, 'yellow mask')
            plt.show()

        # Bitwise-AND mask and original image
        result_yellow = cv2.bitwise_and(frame, frame, mask=mask_yellow)

        if plot == True:
            pre.show_image_BGR(result_yellow, 'yellow of orignal frame')
            plt.show()

        b, g, r = cv2.split(result_yellow)
        result_yellow = g
        _, result_yellow = cv2.threshold(
            result_yellow, 120, 255, cv2.THRESH_BINARY)

        if plot == True:
            pre.show_image(
                result_yellow, 'yellow thersholded of orignal frame')
            plt.show()

        lower_white = np.array([0, 200, 0])
        upper_white = np.array([40, 255, 50])

#         lower_white = np.array([0, 110, 0])
#         upper_white = np.array([40, 255, 50])

        # Threshold the HLS image to get only yellow colors
        mask_white = cv2.inRange(hls, lower_white, upper_white)

        if plot == True:
            pre.show_image(mask_white, 'white mask')
            plt.show()

        # Bitwise-AND mask and original image
        result_white = cv2.bitwise_and(frame, frame, mask=mask_white)

        if plot == True:
            pre.show_image_BGR(result_white, 'white of orignal frame')
            plt.show()

        b, g, r = cv2.split(result_white)
        result_white = g
        _, result_white = cv2.threshold(
            result_white, 130, 255, cv2.THRESH_BINARY)

        if plot == True:
            pre.show_image(result_white, 'white thersholded of orignal frame')
            plt.show()

        result_yw = cv2.bitwise_or(result_white, result_yellow)

        if plot == True:
            pre.show_image(result_yw, 'yellow or white result')
            plt.show()


        ###############################yellow or white image is output now get lane defination#################################

        ################### Isolate possible lane line edges ######################
        l_channel = hls[:, :, 1]

        if plot == True:
            pre.show_image(l_channel, 'L_channel')
            plt.show()

        _, L_channel_binary = cv2.threshold(
            l_channel, 120, 255, cv2.THRESH_BINARY)

        if plot == True:
            pre.show_image(L_channel_binary, 'binary L_channel')
            plt.show()

        L_channel_binary = cv2.GaussianBlur(
            L_channel_binary, (3, 3), 0)  # Reduce noise

        if plot == True:
            pre.show_image(L_channel_binary, "gaussian Binary L_channel")
            plt.show()

        # 1s will be in the cells with the highest Sobel derivative values
        # (i.e. strongest lane line edges)
        L_edges_binary = pre.mag_thresh(
            L_channel_binary, sobel_kernel=3, thresh=(110, 255), value=1)  # -ve -> +ve
#         L_edges_binary = canny_detection(L_channel_binary, thresh = (50, 150)) #0 -> 255

        if plot == True:
            pre.show_image(L_edges_binary, 'Edges L_channel Thersholded')
            plt.show()

        #####################################remove noise of binary image by open morphology#####################################
        # assumed lane lines are pure color (solid white, solid yellow)
        # White in the regions with the purest hue colors (e.g. >80...play with
        # this value for best results).
        s_channel = hls[:, :, 2]  # use only the saturation channel data

        if plot == True:
            pre.show_image(s_channel, 's channel')
            plt.show()

        _, s_binary = cv2.threshold(
            s_channel, self.s_threshold, 255, cv2.THRESH_BINARY)

        if plot == True:
            pre.show_image(s_binary, 'S Binary Thersholded')
            plt.show()

        # White in the regions with the richest red channel values (e.g. >120).
        # pure white is bgr(255, 255, 255), Pure yellow is bgr(0, 255, 255).
        # Both have high red and green channel values.

        b, g, r = cv2.split(frame)

        if plot == True:
            pre.show_image(r, 'G channel')
            plt.show()

        _, g_thresh = cv2.threshold(r, 120, 255, cv2.THRESH_BINARY)

        if plot == True:
            pre.show_image(g_thresh, 'G Binary Thersholded')
            plt.show()

        # Lane lines should be pure in color and have high red channel values
        # Bitwise AND operation to reduce noise and black-out any pixels that
        # don't appear to be nice, pure, solid colors (like white or yellow lane
        # lines.)
        gs_binary = cv2.bitwise_and(s_binary, g_thresh)

        if plot == True:
            pre.show_image(gs_binary, 'G & S Binary Thersholded')
            plt.show()

        ### Combine the possible lane lines with the possible lane line edges #####
        # If you show rs_binary visually, you'll see that it is not that different
        # from this return value. The edges of lane lines are thin lines of pixels.

        self.lane_line_markings = cv2.bitwise_or(
            result_yw, L_edges_binary.astype(np.uint8))
        mask = np.zeros_like(self.lane_line_markings)
        roi_points = self.roi_points.copy()
        roi_points[0, 0] = roi_points[0, 0] - 20
        roi_points[1, 0] = roi_points[1, 0] - 20
        roi_points[2, 0] = roi_points[2, 0] - 20
        roi_points[3, 0] = roi_points[3, 0] + 20

        roi_points[0, 1] = roi_points[0, 1] - 20
        roi_points[1, 1] = roi_points[1, 1] + 20
        roi_points[2, 1] = roi_points[2, 1] + 20
        roi_points[3, 1] = roi_points[3, 1] - 20
        cv2.fillPoly(mask, np.int_([roi_points]), 255)
        self.lane_line_markings = cv2.bitwise_and(
            self.lane_line_markings, mask)
        if plot == True:
            pre.show_image(self.lane_line_markings, "edges or yw in ROI image")
            plt.show()
        result_image = cv2.bitwise_or(self.lane_line_markings, gs_binary)
        if plot == True:
            pre.show_image(result_image, "YorW or lines_with_edges")
            plt.show()
        self.lane_line_markings = result_image

        return self.lane_line_markings


    def Wrap_Presspective(self, frame=None, plot=False):
        """
        input: binary image
        description: wrap image ot birdview
        output: image with lanes only
        """
        if frame is None:
            frame = self.lane_line_markings

        self.transformation_matrix = cv2.getPerspectiveTransform(
            self.roi_points, self.desired_roi_points)

        self.inv_transformation_matrix = cv2.getPerspectiveTransform(
            self.desired_roi_points, self.roi_points)

        self.warped_frame = cv2.warpPerspective(
            frame, self.transformation_matrix, frame.shape[1::-1], flags=(cv2.INTER_LINEAR))

        (thresh, binary_warped) = cv2.threshold(
            self.warped_frame, 127, 255, cv2.THRESH_BINARY)
        self.warped_frame = binary_warped

        if plot == True:
            warped_copy = self.warped_frame.copy()
            warped_copy = cv2.cvtColor(warped_copy, cv2.COLOR_BGR2RGB)
            roi_img = cv2.polylines(warped_copy, np.int32(
                [self.desired_roi_points]), True, (217, 14, 23), 3)
            self.preProcessingshow_image(roi_img, "Bird View")
        return self.warped_frame

    def plot_ROI(self, frame=None, plot=False):
        """
        input: image
        description: using cv2.polylines
        output: get image with ROI
        """
        if frame == None:
            frame = self.orig_frame.copy()

        #cv2.polylines(image, [pts], isClosed, color, thickness)
        roi_img = cv2.polylines(frame, np.int32(
            [self.roi_points]), True, (23, 14, 217), 3)
        self.roi_img = roi_img

        if plot == True:
            roi_img = cv2.cvtColor(roi_img, cv2.COLOR_BGR2RGB)
            plt.imshow(roi_img)
            plt.title("ROI Image")
            plt.axis("off")

            cv2.destroyAllWindows()

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
            win_y_low = self.warped_frame.shape[0] - \
                (window + 1) * window_height
            win_y_high = self.warped_frame.shape[0] - window * window_height
            win_xleft_low = leftx_current - self.margin
            win_xleft_high = leftx_current + self.margin
            win_xright_low = rightx_current - self.margin
            win_xright_high = rightx_current + self.margin
            cv2.rectangle(sliding_window_frame, (win_xleft_low, win_y_low), (
                win_xleft_high, win_y_high), (255, 255, 255), 2)
            cv2.rectangle(sliding_window_frame, (win_xright_low, win_y_low), (
                win_xright_high, win_y_high), (255, 255, 255), 2)

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

        if len(leftx) == 0 or len(lefty) == 0 or len(rightx) == 0 or len(righty) == 0:
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

        if len(leftx) == 0 or len(lefty) == 0 or len(rightx) == 0 or len(righty) == 0:
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
        ploty = np.linspace(
            0, sliding_window_frame.shape[0]-1, sliding_window_frame.shape[0])
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        self.ploty = ploty
        self.left_fitx = left_fitx
        self.right_fitx = right_fitx

        # lanes with sliding windows on top of the lanes
        self.sliding_window_frame = sliding_window_frame

        # Result image
        out_img = np.dstack(
            (sliding_window_frame, sliding_window_frame, (sliding_window_frame))) * 255
        window_image = np.zeros_like(out_img)
        lane_line_img = np.zeros_like(out_img)

        # Left line in red and right one in blue.
        out_img[whitey[left_lane_indices],
                whitex[left_lane_indices]] = [0, 0, 255]
        out_img[whitey[right_lane_indices],
                whitex[right_lane_indices]] = [255, 0, 0]

        # Show the area of the search window
        # Creating the usable format for the fillpoly function.

        left_line_window1 = np.array(
            [np.transpose(np.vstack([left_fitx-self.margin, ploty]))])
        left_line_window2 = np.array(
            [np.flipud(np.transpose(np.vstack([left_fitx+self.margin, ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array(
            [np.transpose(np.vstack([right_fitx-self.margin, ploty]))])
        right_line_window2 = np.array(
            [np.flipud(np.transpose(np.vstack([right_fitx+self.margin, ploty])))])
        right_line_pts = np.hstack(
            (right_line_window1, right_line_window2))

        cv2.fillPoly(window_image, np.int_([left_line_pts]), (0, 255, 0))
        cv2.fillPoly(window_image, np.int_([right_line_pts]), (0, 255, 0))
        result = cv2.addWeighted(out_img, 1, window_image, 0.3, 0)
        
        # Show The lines
        left_line_window1 = np.array(
            [np.transpose(np.vstack([left_fitx-3, ploty]))])
        left_line_window2 = np.array(
            [np.flipud(np.transpose(np.vstack([left_fitx+3, ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array(
            [np.transpose(np.vstack([right_fitx-3, ploty]))])
        right_line_window2 = np.array(
            [np.flipud(np.transpose(np.vstack([right_fitx+3, ploty])))])
        right_line_pts = np.hstack(
            (right_line_window1, right_line_window2))

        cv2.fillPoly(lane_line_img, np.int_([left_line_pts]), (0,255,255))
        cv2.fillPoly(lane_line_img, np.int_([right_line_pts]),(0,255,255))
        self.warped_with_lines = cv2.addWeighted(
            result, 1, lane_line_img, 1, 0)
    
        if plot == True:
            # Plot the figure with the sliding windows and the detected lanes.
            figure, axes = plt.subplots(3, 2)
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

    def histogram_peak(self, frame=None, plot=False):
        """
        Find the histogram to obtain left and right peaks in white pixel count.

        Returns the x index for the left and right white pixels count peaks in the histogram.
        """
        if frame is None:
            frame = self.warped_frame

        # Find the histogram
        self.histogram = np.sum(frame[int(frame.shape[0]/2):, :], axis=0)

        # x indices of left and right peaks.
        midpoint = np.int(self.histogram.shape[0]/2)
        leftx_base = np.argmax(self.histogram[:midpoint])
        rightx_base = np.argmax(self.histogram[midpoint:]) + midpoint

        if plot == True:

            # Plot the image and the corresponding histogram.
            figure, (ax1, ax2) = plt.subplots(2, 1)
            figure.set_size_inches(10, 5)
            ax1.imshow(frame, cmap='gray')
            ax1.set_title("Warped Binary Frame")
            ax2.plot(self.histogram)
            ax2.set_title("Histogram Peaks")
            plt.show()

        return leftx_base, rightx_base

    def calculate_car_position(self):
        """
            Output: The Offset between car position and the center of the lane
        """
        global total_offset
        global frame_count
        global avg_offset

        # The car position is in the center of the frame
        car_position = self.orig_frame.shape[1] / 2

        # the width of the lane in PIXLES
        height = self.orig_frame.shape[0]
        bottom_left = self.left_fit[0] * height ** 2 + \
            self.left_fit[1] * height + self.left_fit[2]
        bottom_right = self.right_fit[0] * height ** 2 + \
            self.right_fit[1] * height + self.right_fit[2]

        # Calculating the lane width from the left lane and the right lane
        lane_width_px = bottom_right - bottom_left

        # Calculating the Position of the center of the lane
        lane_center_px = lane_width_px / 2 + bottom_left

        # The ratio between Meters and Pixels
        self.XM_PER_PIX = self.LANE_WIDTH_M / lane_width_px

        # The Difference between The center of the car and the center of the lane in PIXELS
        diff_car_pos_lane_center_px = car_position - lane_center_px

        # Calculating the offset from the center to the lane in CM
        center_offset = diff_car_pos_lane_center_px * self.XM_PER_PIX * 100

        if frame_count == 1:
            avg_offset = center_offset

        # Averaging the readings each 5 frames to elemnate noise as much as possible
        total_offset += center_offset
        if frame_count % 5 == 0:
            self.center_offset = total_offset/frame_count
            avg_offset = self.center_offset
            total_offset = avg_offset
            frame_count = 1
        else:
            self.center_offset = avg_offset

        return center_offset

    def calculate_curvature(self):

        global frame_count
        global total_curvature
        global avg_curvature
        frame_count += 1

        # Fit polynomial curves on data with near real measurements
        left_fit_cr = np.polyfit(self.lefty * self.YM_PER_PIX,
                                 self.leftx * self.XM_PER_PIX, 2)
        right_fit_cr = np.polyfit(self.righty * self.YM_PER_PIX,
                                  self.rightx * self.XM_PER_PIX, 2)

        # Calculate the radii of curvature of each lane using the formula in the next link
        # https://mathworld.wolfram.com/RadiusofCurvature.html
        left_curvem = ((1 + (2 * left_fit_cr[0] * self.YM_PER_PIX +
                             left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit_cr[0])
        right_curvem = ((1 + (2 * right_fit_cr[0] * self.YM_PER_PIX +
                        right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit_cr[0])

        self.left_curvem = left_curvem
        self.right_curvem = right_curvem
        curvature = (left_curvem + right_curvem) / 2

        if frame_count == 1:
            avg_curvature = curvature

        # Averaging the readings each 5 frames to elemnate noise as much as possible
        total_curvature += curvature
        if frame_count % 5 == 0:
            self.curvature = total_curvature/frame_count
            avg_curvature = self.curvature
            total_curvature = avg_curvature
        else:
            self.curvature = avg_curvature

        return self.curvature

    def fill_area_within_lanes(self):
        """
            output: area within the lanes filled in nonWarped perspective 
                    and in the bird-eye view all in Black Background
        """
        # image to draw the lane lines on Considered as a plank channel
        warp_zero = np.zeros_like(self.warped_frame)

        # Stacking the channels to make a colored image
        self.colored_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Contains the points that surrounds the lanes
        # The points must be in a specific order so we can fill it
        # Bottom Left -> Top Left -> Top Right -> Bottom Right
        perimeter = []

        # Appending The points of the left lane
        # Bottom Left -> Top Left
        left_pts = []
        for index, value in np.ndenumerate(self.left_fitx):
            left_pts.append([value, index[0]])

        # coloring the left lane line with red
        self.colored_warp = cv2.polylines(
            self.colored_warp, np.int32([np.array(left_pts)]), isClosed=False, color=(0, 0, 255), thickness=20)

        # Appending The points of the right lane
        # Top Right -> Bottom Right, hence the points of the right lane is reversed
        right_pts = []
        for index, value in reversed(list(np.ndenumerate(self.right_fitx))):
            right_pts.append([value, index[0]])

        # coloring the right lane line with blue
        self.colored_warp = cv2.polylines(
            self.colored_warp, np.int32([np.array(right_pts)]), isClosed=False, color=(255, 0, 0), thickness=20)

        # Concatinating then Converting to numpy array
        perimeter = np.array(left_pts + right_pts)

        # Fill the area within the parameter with green
        # Bird-view with colored area
        cv2.fillPoly(self.colored_warp, np.int_([perimeter]), (0, 255, 0))

        # Transform From Bird-eye view back to the original view
        self.new_warp = cv2.warpPerspective(
            self.colored_warp, self.inv_transformation_matrix, self.orig_image_size)

    def overlay_over_original_image(self, plot=False):
        """
        Display curvature, offset and the filled lane area on the image
        :return: Image with filled lane area, curvature and offset
        """
        image_copy = self.orig_frame.copy()

        cv2.putText(image_copy, 'Curve Radius: '+str(self.curvature)[:7]+' m', (int((
            5/600)*self.width), int((
                20/338)*self.height)), cv2.FONT_HERSHEY_SIMPLEX, (float((
                    0.5/600)*self.width)), (
            255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image_copy, 'Center Offset: '+str(
            self.center_offset)[:7]+' cm', (int((
                5/600)*self.width), int((
                    40/338)*self.height)), cv2.FONT_HERSHEY_SIMPLEX, (float((
                        0.5/600)*self.width)), (
            255, 255, 255), 2, cv2.LINE_AA)

        image_copy = cv2.addWeighted(image_copy, 1, self.new_warp, 0.3, 0)

        if plot == True:
            cv2.imshow("Image with Curvature and Offset", image_copy)

        return image_copy
