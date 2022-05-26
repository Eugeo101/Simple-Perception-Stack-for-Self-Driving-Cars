# Simple-Perception-Stack-for-Self-Driving-Cars

## Required Libraries
`OpenCV, Matplotlib, Numpy, tqdm, Scipy, Tensorflow`
To make sure you are fully equipped copy the following lines and paste them in terminal
```bash
pip3 install tensorflow
sudo apt-get install python3-scipy
sudo apt install python3-opencv
sudo apt install python-numpy
sudo apt-get install python3-matplotlib
sudo apt install python3-tqdm   

```

## YOLOv3 Wieghts
please download the weights of YOLOv3-416 and add it into Yolo folder

# Running the code
Add The Input video to the project folder and then run one these lines in terminal
## No Debugging
```bash
bash perception.sh input_video.mp4 ouput_video.mp4 yolo debug_mode=0  
```
<p align="left">
  <img src="https://media.giphy.com/media/zG0jYjCGNxpJiBGvvq/giphy.gif" alt="animated" />
</p>

## Debugging
```bash
bash perception.sh input_video.mp4 ouput_video.mp4 yolo debug_mode=1
```
<p align="left">
  <img src="https://media.giphy.com/media/osWJixQTj7XwJlxHhg/giphy.gif" alt="animated" />
</p>
