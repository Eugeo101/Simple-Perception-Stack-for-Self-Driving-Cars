#!/user/bin/python
import sys
import cv2
from lane import Lane
import warnings
from tqdm import tqdm
import lane
import cnn

# To Ignore Warnings
warnings.filterwarnings("ignore")

# rescales a frame by a specific percentage
def rescale_frame(frame, percent=75):

    width = int(frame.shape[1] * percent / 100)
    height = int(frame.shape[0] * percent / 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

# Contains The main Pipeline of the Program
def main(input_video_path, output_video_path, car_detection, debug_mode):

    # Setting the filename
    lane.filename = input_video_path

    # Create a VideoCapture object and read from input file path
    cap = cv2.VideoCapture(input_video_path)

    # The number of frames in the video
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Check if The input video path is valid
    if (cap.isOpened() == False):
        print("Error opening video stream or file")

    # Default resolutions of the frame are obtained.The default resolutions are system dependent.
    # We convert the resolutions from float to integer.
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    # If we are debugging the output video resolution will be 3/4 of the original
    if debug_mode == '1':
        frame_width = int(cap.get(3)) * 3//4
        frame_height = int(cap.get(4)) * 3//4

    # Define the codec to be mp4v and create VideoWriter object.
    # The output is stored in 'outputvideopath.mp4' file.
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(
        *"mp4v"), 20, (frame_width, frame_height))

    # Read until video is completed
    # tqdm is used to show th progress bar
    for i in tqdm(range(n_frames - 1)):
        if(cap.isOpened()):

            # Capture frame by frame
            ret, original_frame = cap.read()

            if ret == True:
                # Create a Lane object
                lane_obj = Lane(orig_frame=original_frame)

                # frames_objects.append(lane_obj)

                # Perform thresholding to isolate lane lines
                lane_line_markings = lane_obj.preProcessing()

                # Plot the region of interest on the image
                lane_obj.plot_ROI(plot=False)

                # Perform the perspective transform to generate a bird's eye view
                # If Plot == True, show image with new region of interest
                warped_frame = lane_obj.Wrap_Presspective(plot=False)

                # Generate the image histogram to serve as a starting point
                # Find lane line pixels using the sliding window method
                lane_obj.sliding_windows()

                # fills the area within lanes
                lane_obj.fill_area_within_lanes()

                # Calculate lane line curvature (left and right lane lines)
                lane_obj.calculate_curvature()

                # Calculate center offset
                lane_obj.calculate_car_position()

                # Display Filled Lanes, curvature and center offset on image
                image_with_lanes = lane_obj.overlay_over_original_image()

                if car_detection == "yolo":
                    # Running YOLOv3
                    final_image = lane_obj.yolo_car_detection(image_with_lanes)
                else:
                    # Running Locally Trained CNN
                    final_image = cnn.cnn_detect_cars(original_frame, image_with_lanes)

                if debug_mode == '1':
                    # rescaling the stage images of the pipeline
                    roi_img25 = rescale_frame(lane_obj.roi_img, percent=25)

                    # Sliding Window
                    sliding_window = cv2.merge((lane_obj.sliding_window_frame,
                                                lane_obj.sliding_window_frame, lane_obj.sliding_window_frame))
                    sliding_window25 = rescale_frame(
                        sliding_window, percent=25)

                    # Marked Lanes in the bird-eye view
                    warped_with_lines25 = rescale_frame(
                        lane_obj.warped_with_lines, percent=25)

                    # Filled area within Lanes in bird-eye view
                    colored_warp25 = rescale_frame(
                        lane_obj.colored_warp, percent=25)

                    # Marked Lanes in the normal view
                    lane_line_markings = cv2.merge(
                        (lane_obj.lane_line_markings, lane_obj.lane_line_markings, lane_obj.lane_line_markings))
                    lane_line_markings25 = rescale_frame(
                        lane_line_markings, percent=25)

                    # Concatinating pipeline images
                    verticalright = cv2.vconcat(
                        [roi_img25, lane_line_markings25])
                    horizontalBottom = cv2.hconcat(
                        [colored_warp25, warped_with_lines25,  sliding_window25])

                    # rescaling the output image in normal view
                    final_img50 = rescale_frame(final_image, percent=50)

                    # Conactinating all the images together
                    debug_final_img = cv2.hconcat([final_img50, verticalright])
                    debug_final_img = cv2.vconcat(
                        [debug_final_img, horizontalBottom])

                    # writing the debug final image
                    out.write(debug_final_img)
                else:
                    # Write the final image only
                    out.write(final_image)

                # Press Q on keyboard to  exit
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break

            # Break the loop
            else:
                break

    # When everything done, release the video capture object
    cap.release()
    out.release()

    # Closes all the frames
    cv2.destroyAllWindows()


# Takes the input data from the bash script
input = sys.argv

# Print Error Message If the debug mode is not specified
try:
    input_video_path, output_video_path, car_detection, debug_mode = input[1:]
    main(input_video_path, output_video_path,
         car_detection.lower(), debug_mode)
    print("done\U0001F973\U0001F389")
except Exception as e:
    print(e)
    print("ERROR!!!\U0001F494\U0001F622")
