import cv2
import math
import numpy as np
import mediapipe as mp
from collections import deque
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


p_face_mesh = mp.solutions.face_mesh

# Initialize the mediapipe hands class.
mp_hands = mp.solutions.hands

# Initializing mediapipe pose class.
mp_pose = mp.solutions.pose

# Initialize the mediapipe drawing class.
mp_drawing = mp.solutions.drawing_utils

# Initialize the mediapipe drawing styles class.
mp_drawing_styles = mp.solutions.drawing_styles
# Initialize the mediapipe hands class.

# Set up the Hands functions for images and videos.
hands = mp_hands.Hands(static_image_mode=True,
                       max_num_hands=2, min_detection_confidence=0.3)
hands_videos = mp_hands.Hands(
    static_image_mode=False, max_num_hands=2, min_detection_confidence=0.8)

# Initialize the mediapipe drawing class.
mp_drawing = mp.solutions.drawing_utils


def detectHandsLandmarks(image, hands, draw=True, display=True):
    '''
    This function performs hands landmarks detection on an image.
    Args:
        image:   The input image with prominent hand(s) whose landmarks needs to be detected.
        hands:   The Mediapipe's Hands function required to perform the hands landmarks detection.
        draw:    A boolean value that is if set to true the function draws hands landmarks on the output image. 
        display: A boolean value that is if set to true the function displays the original input image, and the output 
                 image with hands landmarks drawn if it was specified and returns nothing.
    Returns:
        output_image: A copy of input image with the detected hands landmarks drawn if it was specified.
        results:      The output of the hands landmarks detection on the input image.
    '''

    # Create a copy of the input image to draw landmarks on.
    output_image = image.copy()

    # Convert the image from BGR into RGB format.
    imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Perform the Hands Landmarks Detection.
    results = hands.process(imgRGB)

    # Check if landmarks are found and are specified to be drawn.
    if results.multi_hand_landmarks and draw:

        # Iterate over the found hands.
        for hand_landmarks in results.multi_hand_landmarks:

            # Draw the hand landmarks on the output image.
            mp_drawing.draw_landmarks(image=output_image,
                                      landmark_list=hand_landmarks,
                                      connections=mp_hands.HAND_CONNECTIONS,
                                      landmark_drawing_spec=mp_drawing.
                                      DrawingSpec(color=(255, 255, 255),
                                                  thickness=6, circle_radius=6),
                                      connection_drawing_spec=mp_drawing.
                                      DrawingSpec(color=(0, 255, 0),
                                                  thickness=4, circle_radius=4))

    # Check if the original input image and the output image are specified to be displayed.
    if display:

        # Display the original input image and the output image.
        plt.figure(figsize=[15, 15])
        plt.subplot(121)
        plt.imshow(image[:, :, ::-1])
        plt.title("Sample Image")
        plt.axis('off')
        plt.subplot(122)
        plt.imshow(output_image[:, :, ::-1])
        plt.title("Output Image")
        plt.axis('off')

        # Iterate over the found hands.
        for hand_world_landmarks in results.multi_hand_world_landmarks:

            # Plot the hand landmarks in 3D.
            mp_drawing.plot_landmarks(
                hand_world_landmarks, mp_hands.HAND_CONNECTIONS, azimuth=5)

    # Otherwise
    else:

        # Return the output image and results of hands landmarks detection.
        return output_image, results


def countFingers(image, results, draw=True, display=True):
    '''
    This function will count the number of fingers up for each hand in the image.
    Args:
        image:   The image of the hands on which the fingers counting is required to be performed.
        results: The output of the hands landmarks detection performed on the image.
        draw:    A boolean value that is if set to true the function writes the total count of 
                 fingers up, of the hands on the image.
        display: A boolean value that is if set to true the function displays the resultant image
                 and returns nothing.
    Returns:
        count:            A dictionary containing the count of the fingers that are up, of both hands.
        fingers_statuses: A dictionary containing the status (i.e., up or down) of each finger of both hands.
        tips_landmarks:   A dictionary containing the landmarks of the tips of the fingers of both hands.
    '''

    # Get the height and width of the input image.
    height, width, _ = image.shape

    # Store the indexes of the tips landmarks of each finger of a hand in a list.
    fingers_tips_ids = [mp_hands.HandLandmark.INDEX_FINGER_TIP,
                        mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                        mp_hands.HandLandmark.RING_FINGER_TIP,
                        mp_hands.HandLandmark.PINKY_TIP]

    # Initialize a dictionary to store the status
    # (i.e., True for open and False for close) of each finger of both hands.
    fingers_statuses = {'RIGHT_THUMB': False, 'RIGHT_INDEX': False,
                        'RIGHT_MIDDLE': False, 'RIGHT_RING': False,
                        'RIGHT_PINKY': False, 'LEFT_THUMB': False,
                        'LEFT_INDEX': False, 'LEFT_MIDDLE': False,
                        'LEFT_RING': False, 'LEFT_PINKY': False}

    # Initialize a dictionary to store the count of fingers of both hands.
    count = {'RIGHT': 0, 'LEFT': 0}

    # Initialize a dictionary to store the tips landmarks of each finger of the hands.
    tips_landmarks = {'RIGHT': {'THUMB': (None, None), 'INDEX': (None, None),
                                'MIDDLE': (None, None), 'RING': (None, None),
                                'PINKY': (None, None)},
                      'LEFT': {'THUMB': (None, None), 'INDEX': (None, None),
                               'MIDDLE': (None, None), 'RING': (None, None),
                               'PINKY': (None, None)}}

    # Iterate over the found hands in the image.
    for hand_index, hand_info in enumerate(results.multi_handedness):

        # Retrieve the label of the found hand i.e. left or right.
        hand_label = hand_info.classification[0].label

        # Retrieve the landmarks of the found hand.
        hand_landmarks = results.multi_hand_landmarks[hand_index]

        # Iterate over the indexes of the tips landmarks of each finger of the hand.
        for tip_index in fingers_tips_ids:

            # Retrieve the label (i.e., index, middle, etc.) of the
            # finger on which we are iterating upon.
            finger_name = tip_index.name.split("_")[0]

            # Store the tip landmark of the finger in the dictionary.
            tips_landmarks[hand_label.upper()][finger_name] = \
                (int(hand_landmarks.landmark[tip_index].x*width),
                 int(hand_landmarks.landmark[tip_index].y*height))

            # Check if the finger is up by comparing the y-coordinates
            # of the tip and pip landmarks.
            if (hand_landmarks.landmark[tip_index].y <
                    hand_landmarks.landmark[tip_index - 2].y):

                # Update the status of the finger in the dictionary to true.
                fingers_statuses[hand_label.upper()+"_"+finger_name] = True

                # Increment the count of the fingers up of the hand by 1.
                count[hand_label.upper()] += 1

        # Store the tip landmark of the thumb in the dictionary.
        tips_landmarks[hand_label.upper()]['THUMB'] = \
            (int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x*width),
             int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y*height))

        # Retrieve the x-coordinates of the tip and mcp landmarks of the thumb of the hand.
        thumb_tip_x = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x
        thumb_mcp_x = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP - 2].x

        # Check if the thumb is up by comparing the hand label and
        # the x-coordinates of the retrieved landmarks.
        if (hand_label == 'Right' and (thumb_tip_x < thumb_mcp_x)) or \
                (hand_label == 'Left' and (thumb_tip_x > thumb_mcp_x)):

            # Update the status of the thumb in the dictionary to true.
            fingers_statuses[hand_label.upper()+"_THUMB"] = True

            # Increment the count of the fingers up of the hand by 1.
            count[hand_label.upper()] += 1

    # Check if the total count of the fingers of both hands are specified to be written on the image.
    if draw:

        # Write the total count of the fingers of both hands on the image.
        cv2.putText(image, " Total Fingers: ", (10, 55),
                    cv2.FONT_HERSHEY_COMPLEX, 2, (20, 255, 155), 3)
        cv2.putText(image, str(sum(count.values())), (width//2-150, 240), cv2.FONT_HERSHEY_SIMPLEX,
                    8.9, (20, 255, 155), 10, 10)

    # Check if the image is specified to be displayed.
    if display:

        # Display the resultant image.
        plt.figure(figsize=[10, 10])
        plt.imshow(image[:, :, ::-1])
        plt.title("Output Image")
        plt.axis('off')

    # Otherwise.
    else:

        # Return the count of fingers up, each finger status, and tips landmarks.
        return count, fingers_statuses, tips_landmarks


def recognizeGestures(image, results, hand_label='LEFT', draw=True, display=True):
    '''
    This function will determine the gesture a hand in the image.
    Args:
        image:      The image of the hands on which the hand gesture recognition is required to be performed.
        results:    The output of the hands landmarks detection performed on the image.
        hand_label: The label of the hand i.e. left or right, of which the gesture is to be recognized.      
        draw:       A boolean value that is if set to true the function writes the gesture of the hand on the
                    image, after recognition.
        display:    A boolean value that is if set to true the function displays the resultant image and 
                    returns nothing.
    Returns:
        hands_gestures:        The recognized gesture of the specified hand in the image.
        fingers_tips_position: The fingers tips landmarks coordinates of the other hand in the image.
    '''

    # Initialize a variable to store the gesture of the hand in the image.
    hand_gesture = 'UNKNOWN'

    # Initialize a variable to store the color we will use to write the hand gesture on the image.
    # Initially it is red which represents that the gesture is not recognized.
    color = (0, 0, 255)

    # Get the count of fingers up, fingers statuses, and tips landmarks of the detected hand(s).
    count, fingers_statuses, fingers_tips_position = countFingers(image, results, draw=False,
                                                                  display=False)

    # Check if the number of the fingers up of the hand is 1 and the finger that is up,
    # is the index finger.
    
    if count[hand_label] == 1 and fingers_statuses[hand_label+'_INDEX']:

        # Set the gesture recognized of the hand to INDEX POINTING UP SIGN.
        hand_gesture = 'INDEX POINTING UP'

        # Update the color value to green.
        color = (0, 255, 0)

    # Check if the number of fingers up of the hand is 2 and the fingers that are up,
    # are the index and the middle finger.
    elif count[hand_label] == 2 and fingers_statuses[hand_label+'_INDEX'] and fingers_statuses[hand_label+'_MIDDLE']:

        # Set the gesture recognized of the hand to VICTORY SIGN.
        hand_gesture = 'VICTORY'

        # Update the color value to green.
        color = (0, 255, 0)
    
    elif count[hand_label] == 3 and fingers_statuses[hand_label+'_INDEX'] and fingers_statuses[hand_label+'_PINKY'] and fingers_statuses[hand_label+'_THUMB']:

        # Set the gesture recognized of the hand to VICTORY SIGN.
        hand_gesture = 'SPIDERMAN'

        # Update the color value to green.
        color = (0, 255, 0)

    # Check if the number of fingers up of the hand is 3 and the fingers that are up,
    # are the index, pinky, and the thumb.
    elif count[hand_label] == 3 and fingers_statuses[hand_label+'_INDEX'] and fingers_statuses[hand_label+'_MIDDLE'] and fingers_statuses[hand_label+'_RING']:

        # Set the gesture recognized of the hand to SPIDERMAN SIGN.
        hand_gesture = 'THREE'

        # Update the color value to green.
        color = (0, 255, 0)

    # Check if the number of fingers up of the hand is 5.
    # elif count[hand_label] == 5:

    #     # Set the gesture recognized of the hand to HIGH-FIVE SIGN.
    #     hand_gesture = 'HIGH-FIVE'

    #     # Update the color value to green.
    #     color = (0, 255, 0)
    
    if count[hand_label] == 4  and fingers_statuses[hand_label+'_PINKY'] and fingers_statuses[hand_label+'_INDEX'] and fingers_statuses[hand_label+'_MIDDLE']:
        
        # Set the gesture recognized of the hand to VICTORY SIGN.
        hand_gesture =  'FOURS'
        
        # Update the color value to green.
        color=(0,255,0)
    
    if count[hand_label] == 5  and fingers_statuses[hand_label+'_PINKY'] and fingers_statuses[hand_label+'_INDEX'] and fingers_statuses[hand_label+'_MIDDLE'] and fingers_statuses[hand_label+'_RING'] and fingers_statuses[hand_label+'_THUMB']:
        
        # Set the gesture recognized of the hand to VICTORY SIGN.
        hand_gesture =  'HIGH-FIVE'
        
        # Update the color value to green.
        color=(0,255,0)

    # Check if the recognized hand gesture is specified to be written.
    if draw:

        # Write the recognized hand gesture on the image.
        cv2.putText(image, hand_label + ' HAND: ' + hand_gesture, (10, 60),
                    cv2.FONT_HERSHEY_PLAIN, 4, color, 5)

    # Check if the image is specified to be displayed.
    if display:

        # Display the resultant image.
        plt.figure(figsize=[10, 10])
        plt.imshow(image[:, :, ::-1])
        plt.title("Output Image")
        plt.axis('off')

    # Otherwise
    else:

        # Return the hand gesture name and the fingers tips position of the other hand.
        return hand_gesture, fingers_tips_position['LEFT' if hand_label == 'LEFT' else 'RIGHT']


def calculateDistance(image, point1, point2, draw=True, display=True):
    '''
    This function will calculate distance between two points on an image.
    Args:
        image:   The image on which the two points are.
        point1:  A point with x and y coordinates values on the image.
        point2:  Another point with x and y coordinates values on the image.
        draw:    A boolean value that is if set to true the function draws a line between the 
                 points and write the calculated distance on the image
        display: A boolean value that is if set to true the function displays the output image 
                 and returns nothing.
    Returns:
        distance: The calculated distance between the two points.

    '''

    # Initialize the value of the distance variable.
    distance = None

    # Get the x and y coordinates of the points.
    x1, y1 = point1
    x2, y2 = point2

    # Check if all the coordinates values are processable.
    if isinstance(x1, int) and isinstance(y1, int) \
            and isinstance(x2, int) and isinstance(y2, int):

        # Calculate the distance between the two points.
        distance = math.hypot(x2 - x1, y2 - y1)

        # Check if the distance is greater than the upper threshold.
        if distance > 230:

            # Set the distance to the upper threshold.
            distance = 230

        # Check if the distance is lesser than the lower threshold.
        elif distance < 30:

            # Set the distance to the lower threshold.
            distance = 30

        if draw:

            # Draw a line between the two points on the image.
            cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)),
                     (255, 0, 255), 4)

            # Draw a circle on the first point on the image.
            cv2.circle(image, (int(x1), int(y1)), 20, (0, 255, 0), -1)

            # Draw a circle on the second point on the image.
            cv2.circle(image, (int(x2), int(y2)), 20, (0, 255, 0), -1)

            # Write the calculated distance between the two points on the image.
            cv2.putText(image, f'Distance: {round(distance, 2)}', (10, 30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

    # Check if the image is specified to be displayed.
    if display:

        # Display the resultant image.
        plt.figure(figsize=[10, 10])
        plt.imshow(image[:, :, ::-1])
        plt.title("Output Image")
        plt.axis('off')

    # Return the calculated distance.
    return distance


def changeSatValue(image, scale_factor, channel, display=True):
    '''
    This function will increase/decrease the Saturation or Brighness of an image.
    Args:
        image:        The image whose Saturation or Brighness is to be changed.
        scale_factor: A number that will multiply/scale the required channel of the image.
        channel:      The channel either Saturation or Value whose needed to be modified.
        display:      A boolean value that is if set to true the function displays the original image,
                      and the output image with the modified Saturation or Brighness and returns nothing.
    Returns:
        output_image: A copy of the input image with the Saturation or Brighness modified.

    '''

    # Convert the image from BGR into HSV format.
    image_hsv = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2HSV)

    # Convert the pixel values of the image into float.
    image_hsv = np.array(image_hsv, dtype=np.float64)

    # Split the hue, saturation, and value channel of the image.
    hue_channel, saturation_channel, value_channel = cv2.split(image_hsv)

    # Check if the channel that is needed to be changed is Saturation.
    if channel == 'Saturation':

        # Scale up or down the pixel values of the channel utilizing the scale factor.
        saturation_channel *= scale_factor

    # Check if the channel that is needed to be changed is Value.
    elif channel == 'Value':

        # Scale up or down the pixel values of the channel utilizing the scale factor.
        value_channel *= scale_factor

    # Merge the Hue, Saturation, and Value channel.
    image_hsv = cv2.merge((hue_channel, saturation_channel, value_channel))

    # Set values > 255 to 255.
    image_hsv[image_hsv > 255] = 255

    # Set values < 0 to 0.
    image_hsv[image_hsv < 0] = 0

    # Convert the image into uint8 type and BGR format.
    output_image = cv2.cvtColor(
        np.array(image_hsv, dtype=np.uint8), cv2.COLOR_HSV2BGR)

    # Check if the original input image and the output image are specified to be displayed.
    if display:

        # Display the original input image and the output image.
        plt.figure(figsize=[15, 15])
        plt.subplot(121)
        plt.imshow(image[:, :, ::-1])
        plt.title("Sample Image")
        plt.axis('off')
        plt.subplot(122)
        plt.imshow(output_image[:, :, ::-1])
        plt.title("Output Image")
        plt.axis('off')

    # Otherwise
    else:

        # Return the output image.
        return output_image


def changeContrast(image, scale_factor, display=True):
    '''
    This function will modify the Contrast of an image.
    Args:
        image:        The image whose Contrast is to be changed.
        scale_factor: A number that will scale the Contrast of the image.
        display:      A boolean value that is if set to true the function displays the original image,
                      and the output image with the modified Contrast and returns nothing.
    Returns:
        output_image: A copy of the input image with the Contrast modified.

    '''

    # Change the contrast of a copy of the image.
    output_image = cv2.convertScaleAbs(
        image.copy(), alpha=float(scale_factor), beta=0)

    # Check if the original input image and the output image are specified to be displayed.
    if display:

        # Display the original input image and the output image.
        plt.figure(figsize=[15, 15])
        plt.subplot(121)
        plt.imshow(image[:, :, ::-1])
        plt.title("Sample Image")
        plt.axis('off')
        plt.subplot(122)
        plt.imshow(output_image[:, :, ::-1])
        plt.title("Output Image")
        plt.axis('off')

    # Otherwise
    else:

        # Return the output image.
        return output_image


def gammaCorrection(frame, scale_factor, display=True):
    '''
    This function will perform the Gamma Correction of an image.
    Args:
        image:        The image on which Gamma Correction is to be performed.
        scale_factor: A number that will be used to calculate the required gamma value.
        display:      A boolean value that is if set to true the function displays the original image,
                      and the output image with the modified Contrast and returns nothing.
    Returns:
        output_image: A copy of the input image with the gamma corrected. 
    '''

    # Calculate the gamma value from the passed scale factor.
    gamma = 1.0/scale_factor

    # Initialize the look-up table of 256 elements with values zero.
    table = np.zeros(shape=(1, 256), dtype=np.uint8)

    # Iterate the number of times equal to the number of columns (256) of the look-up table.
    for i in range(table.shape[1]):

        # Calculate the value of the ith column of the look-up table.
        # And clip (limit) the values between 0 and 255.
        table[0, i] = np.clip(a=pow(i/255.0, gamma)*255.0, a_min=0, a_max=255)

    # Perform look-up table transform of the image.
    output_image = cv2.LUT(frame, table)

    # Check if the original input image and the output image are specified to be displayed.
    if display:

        # Display the original input image and the output image.
        plt.figure(figsize=[15, 15])
        plt.subplot(121)
        plt.imshow(frame[:, :, ::-1])
        plt.title("Sample Image")
        plt.axis('off')
        plt.subplot(122)
        plt.imshow(output_image[:, :, ::-1])
        plt.title("Output Image")
        plt.axis('off')

    # Otherwise.
    else:

        # Return the output image.
        return output_image
    
def draw(frame, canvas, current_gesture, hands_tips_positions, prev_coordinates, paint_color, brush_size=20, eraser_size=80):
    '''
    This function will draw, erase and clear a canvas based on different hand gestures.
    Args:
        frame:                A frame/image from the webcam feed.
        canvas:               A black image equal to the webcam feed size, to draw on.
        current_gesture:      The current gesture of the hand recognized using our gesture recognizer from a previous lesson.
        hands_tips_positions:  dictionary containing the Alandmarks of the tips of the fingers of a hand.
        prev_coordinates:     The hand brush x and y coordinates from the previous frame.
        paint_color:          The color to draw with, on the canvas.
        brush_size:           The size of the paint brush to draw with, on the canvas.
        eraser_size:          The size of the eraser to erase with, on the canvas.
    Returns:
        canvas: The black image with the intented drawings on it, in the paint color.
    '''
    # Get the hand brush previous x and y coordinates values (i.e. from the previous frame).
    prev_x, prev_y = prev_coordinates
    
    # Get the height and width of the frame of the webcam video.
    frame_height, frame_width, _ = frame.shape
    
     # Check if the current hand gesture is INDEX POINTING UP.
    if current_gesture == 'INDEX POINTING UP':

        # Write the current mode on the frame with the paint color.
        cv2.putText(img=frame, text='Paint Mode Enabled', org=(10, frame_height-30),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                    color=paint_color, thickness=2)

        # Get the x and y coordinates of tip of the index finger of the hand.
        x, y = hands_tips_positions['INDEX']

        # Check if x and y have valid values.
        # These will be none if the right hand was not detected in the frame.
        # This check will be necessary if you are checking gesture of a different hand and
        # want tips landmarks of the different one. But we are not doing that right now,
        # so if you want you can remove this check.
        if x and y:

            # Check if the previous x and y donot have valid values.
            if not(prev_x) and not(prev_y):

                # Set the previous x and y to the current x and y values.
                prev_x, prev_y = x, y

            # Draw a line on the canvas from previous x and y to the current x and y with the paint color 
            # and thickness equal to the brush_size.
            cv2.line(img=canvas, pt1=(prev_x, prev_y), pt2=(x, y), color=paint_color, thickness=brush_size)

            # Update the previous x and y to the current x and y values.
            prev_x, prev_y = x, y

        
    # Check if the current hand gesture is HIGH-FIVE.
    elif current_gesture == 'HIGH-FIVE':

        # Write the current mode on the frame.
        cv2.putText(img=frame, text='Erase Mode Enabled', org=(10, frame_height-30),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, 
                    color=paint_color, thickness=2)

        # Get the x and y coordinates of tip of the middle finger of the hand.
        x1, y = hands_tips_positions['MIDDLE'] 

        # Get the x coordinate of tip of the ring finger of the hand.
        x2, _ = hands_tips_positions['RING'] 

        # Check if the right hand was detected in the frame.
        if x1 and x2 and y:

            # Calculate the midpoint between tip x coordinate of the middle and ring finger
            x = (x1 + x2) // 2

            # Check if the previous x and y donot have valid values.
            if not(prev_x) and not(prev_y):

                # Set the previous x and y to the current x and y values.
                prev_x, prev_y = x, y

            # Draw a circle on the frame at the current x and y coordinates, equal to the eraser size.
            # This is drawn just to represent an eraser on the current x and y values.
            cv2.circle(img=frame, center=(x, y), radius=int(eraser_size/2), color=(255,255,255), thickness=-1)

            # Draw a black line on the canvas from previous x and y to the current x and y.
            # This will erase the paint between previous x and y and the current x and y.
            cv2.line(img=canvas, pt1=(prev_x, prev_y), pt2=(x, y), color=(0,0,0), thickness=eraser_size)

            # Update the previous x and y to the current x and y values.
            prev_x, prev_y = x, y
    
    # Check if the current hand gesture is SPIDERMAN.
    elif current_gesture == 'SPIDERMAN':

        # Write 'Clear Everything' on the frame.
        cv2.putText(img=frame, text='Clear Everything', org=(10, frame_height-30), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1, color=paint_color, thickness=2)

        # Clear the canvas, by re-initializing it to a complete black image.
        canvas = np.zeros((frame_height, frame_width, 3), np.uint8)
    
    # Return the canvas along with the previous x and y coordinates.
    return canvas, (prev_x, prev_y)

def detectPoseLandmarks(image, pose, draw=True, display=True):
    '''
    This function performs pose landmarks detection on an image.
    Args:
        image:   The input image with a prominent person whose pose landmarks needs to be detected.
        pose:    The Mediapipe's pose landmarks detection function required to perform the pose detection.
        draw:    A boolean value that is if set to true the function draws the detected landmarks on the output image. 
        display: A boolean value that is if set to true the function displays the original input image, the segmentation mask, 
                 the resultant image, and the pose landmarks in 3D plot and returns nothing.
    Returns:
        output_image:   The input image with the detected pose landmarks drawn.
        pose_landmarks: An array containing the detected landmarks (x and y coordinates) converted into their original scale.
    '''
    
    # Retrieve the height and width of the input image.
    height, width, _ = image.shape
    
    # Create a copy of the input image.
    output_image = image.copy()
    
    # Convert the image from BGR into RGB format.
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Perform the Pose Landmarks Detection on the image.
    results = pose.process(imageRGB)
    
    # Initialize a list to store the pose landmarks.
    pose_landmarks = []
    
    # Initialize a variable to store the segmentation mask.
    segmentation_mask =  np.zeros(shape=(height, width), dtype=np.uint8)
    
    # Check if any landmarks are found.
    if results.pose_landmarks:
        
        # Get the segmentation mask of the person detected in the image.
        segmentation_mask = results.segmentation_mask
        
        # Convert the pose landmarks x and y coordinates into their original scale,
        # And store them into a numpy array.
        pose_landmarks = np.array([(landmark.x*width, landmark.y*height) if landmark.visibility>0.7 else (0, 0)\
                                   for landmark in results.pose_landmarks.landmark], dtype=np.int32)
        
        # Check if pose landmarks are specified to be drawn.
        if draw:
            
            # Draw Pose landmarks on the output image.
            mp_drawing.draw_landmarks(image=output_image, landmark_list=results.pose_landmarks,
                                      connections=mp_pose.POSE_CONNECTIONS, landmark_drawing_spec=mp_drawing_styles.
                                      get_default_pose_landmarks_style())
    
    # Check if the original input image, the segmentation mask, and the resultant image are specified to be displayed.
    if display:

        # Display the original input image, the segmentation mask, and the resultant image.
        plt.figure(figsize=[22,22])
        plt.subplot(131);plt.imshow(image[:,:,::-1]);plt.title("Original Image");plt.axis('off');
        plt.subplot(132);plt.imshow(segmentation_mask, cmap='gray');plt.title("Segmentation Mask");plt.axis('off');
        plt.subplot(133);plt.imshow(output_image[:,:,::-1]);plt.title("Output Image");plt.axis('off');


        # Also Plot the Pose landmarks in 3D.
        mp_drawing.plot_landmarks(results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)

    # Otherwise
    else:

        # Return the output image and landmarks detected.
        return output_image, pose_landmarks

def extractPoseKeypoints(image, pose):
    '''
    This function will extract the Pose Landmarks (after normalization) of a person in an image.
    Args:
        image: The input image of the person whose pose landmarks needs to be extracted.
        pose:  The Mediapipe's Pose landmarks detection function required to perform the landmarks detection.
    Returns:
        extracted_landmarks: An array containing the extracted normalized pose landmarks (x and y coordinates).
    '''
    
    # Retrieve the height and width of the image.
    image_height, image_width, _ = image.shape
    
    # Perform the Pose landmarks detection on the image.
    image, pose_landmarks = detectPoseLandmarks(image, pose, draw=True, display=False)
    
    # Initialize a list to store the extracted landmarks.
    extracted_landmarks = []
    
    # Check if pose landmarks are found. 
    if len(pose_landmarks) > 0:
            
        # Iterate over the found pose landmarks. 
        for landmark in pose_landmarks:
            
            # Normalize the landmarks and append them into the list.
            extracted_landmarks.append((landmark[0]/image_width, landmark[1]/image_height))
        
    # Convert the list into an array and flatten the array.
    extracted_landmarks = np.array(extracted_landmarks).flatten()
    
    # Return the image and the extracted normalized pose landmarks.
    return image, extracted_landmarks

def selectShape(frame, hand_tips_positions, DEFAULT_SHAPE_SIZE): 
    '''
    This function will select a shape utilizing a hand tips landmarks.
    Args:
        frame:               The current frame/image of a real-time webcam feed.
        hand_tips_positions: A dictionary containing the landmarks of the tips of the fingers of a hand.
        DEFAULT_SHAPE_SIZE:  The default size of each drawable shape.
    Returns:
        shape_selected: The shape to draw, selected by the user using middle finger tip.
    '''
    
    # Get the height and width of the frame of the webcam video.
    frame_height, frame_width, _ = frame.shape
    
    # Initialize a variable to store the selected shape.
    shape_selected = None
    
    # Read the shapes selection tab image and resize its width equal to the frame width.
    shapes_selector_tab = cv2.imread('media/overlays/shapes_selector.png')
    shapes_selector_height, shapes_selector_width, _ = shapes_selector_tab.shape
    shapes_selector_tab = cv2.resize(shapes_selector_tab, (frame_width, int((frame_width/shapes_selector_width)*shapes_selector_height)))
    
    # Overlay the shape selection tab image on the top of the frame.
    frame[0:shapes_selector_tab.shape[0], 0:shapes_selector_tab.shape[1]] = shapes_selector_tab
    
    # Get the x and y coordinates of tip of the MIDDLE finger of the hand. 
    x, y = hand_tips_positions['MIDDLE']
    
    # Check if the MIDDLE finger tip is over the shape selection tab image.
    if y <= shapes_selector_tab.shape[0]:
        
        # Check if the MIDDLE finger tip is over the Circle ROI.
        if x>(int(frame_width//11.5)-DEFAULT_SHAPE_SIZE//2) and \
        x<(int(frame_width//11.5)+DEFAULT_SHAPE_SIZE//2):
            
            # Update the selected shape variable to 'Circle'.
            shape_selected='Circle'
        
        # Check if the MIDDLE finger tip is over the Polygon ROI.
        elif x>(int(frame_width//4.35)-DEFAULT_SHAPE_SIZE/2) and \
        x<(int(frame_width//4.35)+DEFAULT_SHAPE_SIZE//2):
            
            # Update the selected shape variable to 'Polygon'.
            shape_selected='Polygon'
        
        # Check if the MIDDLE finger tip is over the Rectangle ROI.
        elif x>(int(frame_width//2.37)-DEFAULT_SHAPE_SIZE//2) and \
        x<(int(frame_width//2.37)+DEFAULT_SHAPE_SIZE//2):
            
            # Update the selected shape variable to 'Rectangle'.
            shape_selected='Rectangle'
        
        # Check if the MIDDLE finger tip is over the Square ROI.
        elif x>(int(frame_width//1.64)-DEFAULT_SHAPE_SIZE//2) and \
        x<(int(frame_width//1.64)+DEFAULT_SHAPE_SIZE//2):
            
            # Update the selected shape variable to 'Square'.
            shape_selected='Square'
        
        # Check if the MIDDLE finger tip is over the Triangle ROI.
        elif x>(int(frame_width//1.095)-DEFAULT_SHAPE_SIZE//2) and \
        x<(int(frame_width//1.095)+DEFAULT_SHAPE_SIZE//2):
            
            # Update the selected shape variable to 'Triangle'.
            shape_selected='Triangle'
        
        # Check if the MIDDLE finger tip is over the Right Triangle ROI.
        elif x>(int(frame_width//1.292)-DEFAULT_SHAPE_SIZE//2) and \
        x<(int(frame_width//1.292)+DEFAULT_SHAPE_SIZE//2):
            
            # Update the selected shape variable to 'Right Triangle'.
            shape_selected='Right Triangle'
    
    # Return the selected shape.
    return shape_selected

def drawShapes(frame, canvas, shape_selected, hand_gesture, hand_tips_positions, paint_color, shape_size):
    '''
    This function will draw a selected shape on a frame or canvas based on hand gestures.
    Args:
        frame:               The current frame/image of a real-time webcam feed.
        canvas:              A black image equal to the webcam feed size, to draw on.
        shape_selected:      The shape to draw, selected by the user using middle finger tip.
        hand_gesture:        The current hand gesture recognized in the frame.
        hand_tips_positions: A dictionary containing the landmarks of the tips of the fingers of a hand.
        paint_color:         The color to draw shapes with, on the canvas.
        shape_size:          The size of which, the selected shape is to be draw.
    Returns:
        frame:          The frame with the selected shape drawn and current active mode written.
        canvas:         The black image with the selected shapes drawn on it, in the paint color.
        shape_selected: The name of the selected shape.
    '''
    # Get the height and width of the frame of the webcam video.
    frame_height, frame_width, _ = frame.shape
    
    # Get the x and y coordinates of tip of the MIDDLE finger of the hand. 
    x, y = hand_tips_positions['MIDDLE']
    
    # Check if the current hand gesture is 'VICTORY'. 
    if hand_gesture == 'VICTORY':
        
        # Write 'Shape Selection Mode Enabled' on the frame.
        cv2.putText(frame, 'Shape Selection Mode Enabled', (10, frame_height-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, paint_color, 2)
        
        # Store the frame reference in the image variable.
        # Now if we draw on the image (variable), the drawing will be made on the frame.
        image = frame
            
    # Check if the current hand gesture is 'INDEX MIDDLE THUMB POINTING UP'. 
    elif hand_gesture == 'INDEX MIDDLE THUMB POINTING UP':
        
        # Write 'Shape Placement Mode Enabled' on the frame.
        cv2.putText(frame, 'Shape Placement Mode Enabled', (10, frame_height-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, paint_color, 2)
        
        # Store the canvas reference in the image variable.
        # Now if we draw on the image (variable), the drawing will be made on the canvas.
        image = canvas
           
    # Check if the current hand gesture is 'SPIDERMAN'.
    elif hand_gesture == 'SPIDERMAN':

        # Write 'Clear Everything' on the frame.
        cv2.putText(img=frame, text='Clear Everything', org=(10, frame_height-30), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1, color=paint_color, thickness=2)

        # Clear the canvas, by re-initializing it to a complete black image.
        canvas = np.zeros((frame_height, frame_width, 3), np.uint8)
    
    # Check if the current hand gesture is 'VICTORY' or 'INDEX MIDDLE THUMB POINTING UP'.
    if hand_gesture == 'VICTORY' or hand_gesture == 'INDEX MIDDLE THUMB POINTING UP':
        
        # Check if the selected shape is 'Circle'.
        if shape_selected == 'Circle':
            
            # Get the radius of the circle to draw.
            circle_radius = shape_size//2
            
            # Draw the circle on the image that will be either frame or canvas based on current gesture.
            image = cv2.circle(img=image, center=(x, y), radius=circle_radius,
                               color=paint_color, thickness=-1)
        
        # Check if the selected shape is 'Polygon'.
        elif shape_selected == 'Polygon':
            
            # Get the radius of the polygon to draw.
            polygon_radius = shape_size//2
            
            # Initialize a list to store the polygon contour (boundary) points.
            polygon_pts=[]
            
            # Iterate over 6 times. 
            # As we are drawing a polygon (hexagon) with 6 sides.
            for i in range(6):
                
                # Get the x and y coordinates of a polygon corner.
                poly_x = x + polygon_radius* math.cos(i * 2 * math.pi / 6)
                poly_y = y + polygon_radius * math.sin(i * 2 * math.pi / 6)
                
                # Append the x and y coordinates into the list.
                polygon_pts.append((poly_x, poly_y))
            
            # Draw the polygon on the image that will be either frame or canvas based on current gesture.
            image = cv2.fillPoly(image, pts = [np.array(polygon_pts, np.int32)],
                                 color=paint_color)
        
        # Check if the selected shape is 'Rectangle'.
        elif shape_selected == 'Rectangle':
            
            # Get the rectangle height and width.
            rec_width = shape_size*2
            rec_height = shape_size
            
            # Draw the rectangle on the image that will be either frame or canvas based on current gesture.
            image = cv2.rectangle(image, pt1=(x-rec_width//2,y-rec_height//2),
                                  pt2=(x+rec_width//2,y+rec_height//2),
                                  color=paint_color, thickness=-1)
        
        # Check if the selected shape is 'Square'.
        elif shape_selected == 'Square':
            
            # Draw the square on the image that will be either frame or canvas based on current gesture.
            image = cv2.rectangle(image, pt1=(x-shape_size//2, y-shape_size//2),
                                  pt2=(x+shape_size//2, y+shape_size//2),
                                  color=paint_color, thickness=-1)
        
        # Check if the selected shape is 'Triangle'.
        elif shape_selected == 'Triangle':
            
            # Get the x and y coordinates of the triangle corners.
            triangle_pts= [(x, y-shape_size//2),
                           (x-shape_size//2, y+shape_size//2),
                           (x+shape_size//2, y+shape_size//2)]
            
            # Draw the triangle on the image that will be either frame or canvas based on current gesture.
            image = cv2.drawContours(image=image, contours=[np.array(triangle_pts, np.int32)], contourIdx=0, 
                                     color=paint_color, thickness=-1)
        
        # Check if the selected shape is 'Right Triangle'.
        elif shape_selected == 'Right Triangle':
            
            # Get the x and y coordinates of the right triangle corners.
            triangle_pts= [(x-shape_size//2, y-shape_size//2),
                           (x-shape_size//2, y+shape_size//2),
                           (x+shape_size//2, y+shape_size//2)]
            
            # Draw the right triangle on the image that will be either frame or canvas based on current gesture.
            image = cv2.drawContours(image=image, contours=[np.array(triangle_pts, np.int32)], contourIdx=0, 
                                     color=paint_color, thickness=-1)
    
    # Check if the current hand gesture is 'INDEX MIDDLE THUMB POINTING UP'. 
    if hand_gesture == 'INDEX MIDDLE THUMB POINTING UP':
        
        # Update the selected shape variable to 'None'.
        shape_selected=None
    
    # Return the frame, canvas, and the selected shape.
    return frame, canvas, shape_selected

