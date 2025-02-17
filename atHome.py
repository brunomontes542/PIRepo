import cv2 as cv
import numpy as np
from picamera2 import Picamera2

# Initialize the PiCamera2
picam2 = Picamera2()

# Start the camera
picam2.start()

# Get the camera resolution (use the default resolution or set one)
frame_width = 640
frame_height = 480

# Video writer setup
fourcc = cv.VideoWriter_fourcc(*'mp4v')
out = cv.VideoWriter('output.mp4', fourcc, 20.0, (frame_width, frame_height))

lower_orange_value = np.array([5, 150, 180])  # Darker neon orange (loosened S & V)
upper_orange_value = np.array([20, 255, 255])  # Brighter neon orange (increased S & V)

# Main loop
while True:
    # Capture a frame from PiCamera2
    frame = picam2.capture_array("main")

    # Resize the frame if needed (to match desired resolution)
    frame = cv.resize(frame, (frame_width, frame_height))
    frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)

    # Convert image to HSV 
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # Create the mask for the orange color
    mask = cv.inRange(hsv, lower_orange_value, upper_orange_value)

    # Apply morphological transformations to reduce noise
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    # Find contours in the mask
    contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv.contourArea(contour) > 200:  # Filter out small contours
            x, y, w, h = cv.boundingRect(contour)
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw bounding box
            cv.putText(frame, "orange ball", (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Show the live feed with bounding boxes
    cv.imshow("live feed", frame)

    # Write the frame to the output video file
    out.write(frame)

    # Break loop if 'q' is pressed
    if cv.waitKey(1) == ord('q'):
        break

# Release resources
picam2.stop()
out.release()
cv.destroyAllWindows()
