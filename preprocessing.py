import shared_data

def plot_ROI(frame = None, plot = False):
    #input img
    #output get image with ROI
    
    if frame == None:
        frame = shared_data.orig_frame.copy()
    if plot == True:
        #cv2.polylines(image, [pts], isClosed, color, thickness)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        roi_img = cv2.polylines(frame, np.int32([shared_data.roi_points]), True, (217, 14, 23), 3)
        plt.imshow(roi_img)
        plt.title("ROI Image")
        plt.axis("off")
    else:
        return
