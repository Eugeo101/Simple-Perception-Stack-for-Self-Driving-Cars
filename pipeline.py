#!/user/bin/python
import sys
import cv2

# Contains The main Pipeline of the Program
def main(input_video_path, output_video_path, debug_mode):
    # Create a VideoCapture object and read from input file path
    cap = cv2.VideoCapture(input_video_path)

    # Check if camera opened successfully
    if (cap.isOpened() == False):
        print("Error opening video stream or file")

    # Default resolutions of the frame are obtained.The default resolutions are system dependent.
    # We convert the resolutions from float to integer.
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    # Define the codec to be mp4v and create VideoWriter object.
    # The output is stored in 'outputvideopath.mp4' file.
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), 10, (frame_width,frame_height))

    # Read until video is completed
    while(cap.isOpened()):
        
        # Capture frame by frame
        ret, original_frame = cap.read()
        
        if ret == True:    
            # if the debug mode is 0
            if debug_mode == 0:
                pass
            # if the debug mode is 1
            else:
                pass
            
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
except:
    print("ERROR: please enter valid input")
