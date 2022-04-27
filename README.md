# Simple-Perception-Stack-for-Self-Driving-Cars

## Required Libraries
`OpenCV, Matplotlib, Numpy, tqdm`
To make sure you are fully equipped copy the following lines and paste them in terminal
```bash
sudo apt install python3-opencv
sudo apt install python-pip
sudo apt-get install python3-matplotlib
sudo apt install python3-tqdm   

```
You Add The Input video to the project folder and then run this line in terminal
## No Debugging
```bash
bash perception.sh input_video.mp4 ouput_video.mp4 debug_mode=0  
```
<p align="left">
  <img src="https://media.giphy.com/media/hJkMAghvJjHrSW9wgZ/giphy.gif" alt="animated" />
</p>

## Debugging
```bash
bash perception.sh input_video.mp4 ouput_video.mp4 debug_mode=1
```
<p align="left">
  <img src="https://media.giphy.com/media/rDXE2l9W76fz2Ec3Vn/giphy.gif" alt="animated" />
</p>
