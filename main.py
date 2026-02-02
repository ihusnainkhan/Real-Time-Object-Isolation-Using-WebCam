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

print("Press 's' to capture frame and process")
print("Press 'q' to quit")

captured_frame = None
gray_frame = None

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    # Show live webcam feed
    cv2.imshow("Live Webcam", frame)
    
    key = cv2.waitKey(1) & 0xFF
    

    if key == ord('s'):
        
        captured_frame = frame.copy()
        
        cap.release()
        cv2.destroyAllWindows()
        
        gray_frame = convert_grayscale(captured_frame)
        print("Frame captured & webcam closed.")
        break
    
    elif key == ord('q'):
        
        cap.release()
        cv2.destroyAllWindows()
        exit()
        
if captured_frame is None:
    print("No frame captured.")
    exit()
    
print("\nChoose an operation:")
print("1 - Canny Edge Detection")
print("2 - Thresholding")
print("3 - Bitwise AND (Object Isolation)")

choice = input("Enter your choice (1/2/3): ").strip()

output = None
window_name = ""

if choice == "1":
    output = apply_canny(gray_frame)
    window_name = "Canny Edge Detection"
    
elif choice == "2":
    output = apply_threshold(gray_frame)
    window_name = "Thresholding"
    
elif choice == "3":
    mask = apply_threshold(gray_frame)
    output = apply_bitwise_and(captured_frame, mask)
    window_name = "Bitwise AND Result"
    
else:
    print("Invalid choice")
    exit()
    
if output is None:
    print("Error: No output to display")
    exit()

cv2.imshow(window_name, output)
cv2.waitKey(0)
cv2.destroyAllWindows()
