def histogram_peak(self,frame=None,plot=False):
    """
    Find the histogram to obtain left and right peaks in white pixel count.
		
    Returns the x index for the left and right white pixels count peaks in the histogram.
    """
    if frame is None:
      frame = self.warped_frame
			
    # Find the histogram
    self.histogram = np.sum(frame[int(frame.shape[0]/2):,:], axis=0)

    # x indices of left and right peaks.
    midpoint = np.int(self.histogram.shape[0]/2)
    leftx_base = np.argmax(self.histogram[:midpoint])
    rightx_base = np.argmax(self.histogram[midpoint:]) + midpoint

    if plot == True:
		
      # Plot the image and the corresponding histogram.
      figure, (ax1, ax2) = plt.subplots(2,1)
      figure.set_size_inches(10, 5)
      ax1.imshow(frame, cmap='gray')
      ax1.set_title("Warped Binary Frame")
      ax2.plot(self.histogram)
      ax2.set_title("Histogram Peaks")
      plt.show()

    return leftx_base, rightx_base
