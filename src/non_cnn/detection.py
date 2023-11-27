import cv2


def get_sub_images(image, padding=0.1):
    # Segment image
    image = image.copy()

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    blur = cv2.GaussianBlur(hsv, (5, 5), 0)
    mask = cv2.threshold(
        blur[:, :, 2], 0, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C + cv2.THRESH_OTSU
    )[1]

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
            ROI = image[y : y + h, x : x + w]
            cropped_images.append(ROI)

    # Create hierarchy tree
    hierarchy = build_bvh(bbox)

    return cropped_images, bbox, hierarchy


def build_bvh(bboxes):
    """Build a bounding volume hierarchy and return a list of parent indices."""
    nodes = [-1] * len(bboxes)  # Initialize all nodes with -1 (no parent)

    for i, bbox in enumerate(bboxes):
        for j, other in enumerate(bboxes):
            if i != j and intersects(bbox, other):
                # Check if bbox[i] is within bbox[j] to consider j as a parent
                # The parent is the largest bounding box that entirely contains bbox[i]
                if is_within(bbox, other):
                    if nodes[i] == -1 or is_within(bboxes[nodes[i]], other):
                        nodes[i] = j

    return nodes


def is_within(inner, outer):
    """Check if the first bbox is completely within the second bbox."""
    xi, yi, wi, hi = inner
    xo, yo, wo, ho = outer
    return xi >= xo and yi >= yo and xi + wi <= xo + wo and yi + hi <= yo + ho


def intersects(bbox1, bbox2):
    """Check if two bounding boxes intersect."""
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    return not (x1 + w1 <= x2 or x1 >= x2 + w2 or y1 + h1 <= y2 or y1 >= y2 + h2)
