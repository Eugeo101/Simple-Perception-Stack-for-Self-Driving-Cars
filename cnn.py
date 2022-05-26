import cv2
import numpy as np
import tensorflow as tf
from scipy.ndimage.measurements import label
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Conv2D, MaxPooling2D, Lambda

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def create_model(inputShape=(64, 64, 3)):

    model = Sequential()
    # Center and normalize our data
    model.add(Lambda(lambda x: x / 255.,
              input_shape=inputShape, output_shape=inputShape))
    # Block 0
    model.add(Conv2D(filters=16, kernel_size=(3, 3), activation='relu', name='cv0',
                     input_shape=inputShape, padding="same"))
    model.add(Dropout(0.5))

    # Block 1
    model.add(Conv2D(filters=32, kernel_size=(3, 3),
              activation='relu', name='cv1', padding="same"))
    model.add(Dropout(0.5))

    # block 2
    model.add(Conv2D(filters=64, kernel_size=(3, 3),
              activation='relu', name='cv2', padding="same"))
    model.add(MaxPooling2D(pool_size=(8, 8)))
    model.add(Dropout(0.5))

    # binary 'classifier'
    model.add(Conv2D(filters=1, kernel_size=(8, 8),
              name='fcn', activation="sigmoid"))

    return model


# Init a version of our network with another resolution without the flatten layer
heatmodel = create_model((260, 1280, 3))
# print(heatmodel.count_params())
# Load the weights
heatmodel.load_weights('./CNN/new_model.h5')


def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    draw_img = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(draw_img, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return draw_img


def search_cars(img):
    # We crop the image to 440-660px in the vertical direction
    cropped = img[400:660, 0:1280]
    # plt.imshow(cropped)
    heat = heatmodel.predict(cropped.reshape(
        1, cropped.shape[0], cropped.shape[1], cropped.shape[2]),verbose=0)
    # This finds us rectangles that are interesting
    xx, yy = np.meshgrid(np.arange(heat.shape[2]), np.arange(heat.shape[1]))
    x = (xx[heat[0, :, :, 0] > 0.95])
    y = (yy[heat[0, :, :, 0] > 0.95])
    hot_windows = []
    # We save those rects in a list
    for i, j in zip(x, y):
        hot_windows.append(((i*8, 400 + j*8), (i*8+64, 400 + j*8+64)))
    return hot_windows


def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
    # Return updated heatmap
    return heatmap  # Iterate through list of bboxes


def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap


def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)),
                (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0, 255, 0), 3)
    # Return the image
    return img


def cnn_detect_cars(img, image_with_lanes):
    # Search for our windows
    hot_windows = search_cars(img)
    # Draw the found boxes on the test image
    window_img = draw_boxes(img, hot_windows, (0, 255, 0), 6)

    # Show the image with the windows on top
    # fig = plt.figure(figsize=(12,20))
    # plt.imshow(window_img)
    # Create image for the heat similar to one shown above
    heat = np.zeros_like(img[:, :, 0]).astype(np.float)

    # Add heat to each box in box list
    heat = add_heat(heat, hot_windows)

    # Apply threshold to help remove false positives
    heat = apply_threshold(heat, 10)

    # Visualize the heatmap when displaying
    heatmap = np.clip(heat, 0, 255)

    # Find final boxes from heatmap using label function
    boxes = label(heatmap)

    # Create the final image
    draw_img = draw_labeled_bboxes(np.copy(image_with_lanes), boxes)

    return draw_img
