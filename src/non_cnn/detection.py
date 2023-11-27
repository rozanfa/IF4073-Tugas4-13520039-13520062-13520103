import cv2


def get_sub_images(image, padding=0.1):
    # Segment image
    image = image.copy()

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    blur = cv2.GaussianBlur(hsv, (5, 5), 0)
    mask = cv2.threshold(blur[:, :, 2], 0, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C + cv2.THRESH_OTSU)[1]

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)

    # Find contours
    cnts, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cnts_inv, _ = cv2.findContours(~opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cnts = cnts + cnts_inv

    min_area = 2000
    cropped_images = []
    bbox = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area > min_area:
            x, y, w, h = cv2.boundingRect(c)
            bbox.append([x, y, w, h])
            x = max(0, x - int(padding * w))
            y = max(0, y - int(padding * h))
            w = min(image.shape[1], w + 2 * int(padding * w))
            h = min(image.shape[0], h + 2 * int(padding * h))
            ROI = image[y:y+h, x:x+w]
            cropped_images.append(ROI)

    return cropped_images, bbox
