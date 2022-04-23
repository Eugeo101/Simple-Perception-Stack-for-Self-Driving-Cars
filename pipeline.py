#!/user/bin/python
import sys

# Contains The main Pipeline of the Program
def main(input_video_path, output_video_path, debug_mode):
    pass


# Takes the input data from the bash script
input = sys.argv

# Print Error Message If the debug mode is not specified
try:
    input_video_path, output_video_path, debug_mode = input[1:]
    main(input_video_path, output_video_path, debug_mode)
except:
    print("ERROR: please enter valid input")
