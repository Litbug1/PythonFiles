import cv2
import numpy as np
import pyautogui

# Set the sensitivity of the movement detection
sensitivity = 10

# Set the minimum area of a moving object to be detected
min_area = 1000

while True:
    # Get the current frame
    frame = pyautogui.screenshot()
    frame = np.array(frame)

    # Get the width and height of the screen
    screen_width, screen_height = frame.shape[1], frame.shape[0]

    # Set the background model to be used for background subtraction
    bg_model = cv2.createBackgroundSubtractorMOG2()

    # Set the previous frame to a black image
    prev_frame = np.zeros((screen_height, screen_width, 3), np.uint8)

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply the background model to the current frame
    fg_mask = bg_model.apply(gray)

    # Dilate the foreground mask to fill in holes
    kernel = np.ones((5,5), np.uint8)
    fg_mask = cv2.dilate(fg_mask, kernel, iterations=1)

    
       # Resize the frame to make it smaller
    frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

    # Show the current frame
    cv2.imshow("Frame", frame)

    # Check if the 'q' key has been pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video capture and destroy the windows
cv2.destroyAllWindows()
