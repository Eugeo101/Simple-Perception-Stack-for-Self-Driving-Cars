import shared_data

def plot_ROI(frame = None, plot = False):
    """
    input: image
    description: using cv2.polylines
    output: get image with ROI
    """
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

def Wrap_Presspective(frame = None, plot = False):
"""
input: binary image
description: wrap image ot birdview
output: image with lanes only
"""
    if frame is None:
#             frame = Preprocessing(frame)
shared_data.transformation_matrix = cv2.getPerspectiveTransform(shared_data.src, shared_data.dist)
shared_data.inverse_transformation_matrix = cv2.getPerspectiveTransform(shared_data.dist, shared_data.src)
shared_data.warped_frame = cv2.warpPerspective(frame, shared_data.transformation_matrix, frame.shape[1::-1], flags=(cv2.INTER_LINEAR))

if plot == True:
    copy_wraped = warped_frame.copy()
    plot_ROI(src=dist, frame= copy_wraped, plot = True)
return warped_frame
