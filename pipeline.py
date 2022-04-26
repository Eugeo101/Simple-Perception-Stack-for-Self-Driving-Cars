#!/user/bin/python
import sys
import cv2
from lane import Lane
import warnings
from tqdm import tqdm
import emoji as emoji

# To Ignore Warnings
warnings.filterwarnings("ignore")
# Contains The main Pipeline of the Program


def main(input_video_path, output_video_path, debug_mode):
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

    # Define the codec to be mp4v and create VideoWriter object.
    # The output is stored in 'outputvideopath.mp4' file.
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(
        *"mp4v"), 10, (frame_width, frame_height))

    # Read until video is completed
    # tqdm is used to show th progress bar
    for i in tqdm(range(n_frames - 10)):
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
                frame_with_lane_lines2 = lane_obj.overlay_over_original_image()

                # Write the frame into the file 'output.avi'
                out.write(frame_with_lane_lines2)

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
    input_video_path, output_video_path, debug_mode = input[1:]
    main(input_video_path, output_video_path, debug_mode)
    # Get printed if the program ran successfully 
    print("done\U0001F973\U0001F389")
except:
    print("ERROR: please enter valid input")
