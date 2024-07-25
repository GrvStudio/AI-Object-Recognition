import cv2

def draw_border(img, pt1, pt2, color, thickness, line_length_x, line_length_y, padding):
    x1, y1 = pt1
    x2, y2 = pt2

    cv2.rectangle(img, (x1 - padding, y1 - padding), (x2 + padding, y2 + padding), color, thickness)

    cv2.line(img, (x1 - padding, y1 - padding), (x1 - padding + line_length_x, y1 - padding), color, thickness)
    cv2.line(img, (x1 - padding, y1 - padding), (x1 - padding, y1 - padding + line_length_y), color, thickness)

    cv2.line(img, (x2 + padding, y1 - padding), (x2 + padding - line_length_x, y1 - padding), color, thickness)
    cv2.line(img, (x2 + padding, y1 - padding), (x2 + padding, y1 - padding + line_length_y), color, thickness)

    cv2.line(img, (x1 - padding, y2 + padding), (x1 - padding + line_length_x, y2 + padding), color, thickness)
    cv2.line(img, (x1 - padding, y2 + padding), (x1 - padding, y2 + padding - line_length_y), color, thickness)

    cv2.line(img, (x2 + padding, y2 + padding), (x2 + padding - line_length_x, y2 + padding), color, thickness)
    cv2.line(img, (x2 + padding, y2 + padding), (x2 + padding, y2 + padding - line_length_y), color, thickness)

    return img