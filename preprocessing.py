import shared_data

"""
    To access Shared data:
        shared_data.orig_frame
"""
def plot_ROI(src, frame = None, plot = False):
    #input img
    #output get image with ROI
    
    if plot == True:
        #cv2.polylines(image, [pts], isClosed, color, thickness)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        roi_img = cv2.polylines(frame, [src], True, (217, 14, 23), 3)
        plt.imshow(roi_img)
        
    else:
        return
