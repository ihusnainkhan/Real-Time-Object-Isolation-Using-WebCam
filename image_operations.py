import cv2

def convert_grayscale(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return gray

def apply_canny(gray_image, low_threshold=50, high_threshold=150):
    edges = cv2.Canny(gray_image, low_threshold, high_threshold)
    return edges

def apply_threshold(gray_image, thresh_value =127):
    ret, thresh_image = cv2.threshold(gray_image, thresh_value, 255, cv2.THRESH_BINARY)
    return thresh_image

def apply_bitwise_and(original_frame, mask):
    mask = mask.astype('uint8')
    bitwise_and = cv2.bitwise_and(original_frame, original_frame, mask=mask)
    return bitwise_and
