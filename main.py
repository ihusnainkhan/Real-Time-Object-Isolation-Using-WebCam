import cv2
from image_operations import (
    convert_grayscale,
    apply_canny,
    apply_threshold,
    apply_bitwise_and
)

# Open webcam
cap = cv2.VideoCapture(0)

# Check if webcam opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

# print("Press 's' to capture frame and process")
# print("Press 'q' to quit")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    # Show live webcam feed
    cv2.imshow("Live Webcam", frame)

    key = cv2.waitKey(10) & 0xFF

    # Capture frame and apply operations
    if key == ord('s'):
        
        # Step 1: Grayscale
        gray = convert_grayscale(frame)

        # Step 2: Canny Edge Detection
        edges = apply_canny(gray)

        # Step 3: Thresholding
        mask = apply_threshold(gray)

        # Step 4: Bitwise AND (Object Isolation)
        result = apply_bitwise_and(frame, mask)

        # Show results
        cv2.imshow("Captured Frame", frame)
        cv2.imshow("Grayscale", gray)
        cv2.imshow("Edges", edges)
        cv2.imshow("Threshold Mask", mask)
        cv2.imshow("Final Isolated Object", result)

    # Quit program
    elif key == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
