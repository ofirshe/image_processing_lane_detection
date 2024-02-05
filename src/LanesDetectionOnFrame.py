import cv2
import numpy as np
import matplotlib.pyplot as plt


class LanesDetectionOnFrame:
    CENTER_LINE_X = 417
    OFFSET = 5

    @staticmethod
    def preprocess_image(image):
        height, width, _ = image.shape

        # Convert to HLS color space and threshold the saturation channel
        hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
        saturation_threshold = 120
        sat_binary = cv2.inRange(hls, (0, saturation_threshold, 0), (255, 255, 255))

        # Create a mask to filter out non-lane regions
        mask = np.zeros_like(sat_binary)
        roi_vertices = np.array(
            [[(50, height), (width // 2 - 50, height // 2 + 50), (width // 2 + 50, height // 2 + 50),
              (width - 50, height)]], dtype=np.int32)
        cv2.fillPoly(mask, roi_vertices, 255)
        masked_image = cv2.bitwise_and(sat_binary, mask)

        # Apply morphological operations to reduce noise
        kernel = np.ones((5, 5), np.uint8)
        morph_image = cv2.morphologyEx(masked_image, cv2.MORPH_CLOSE, kernel)

        return morph_image

    @staticmethod
    def separate_lines(lines):
        left_lines = []
        right_lines = []

        for line in lines:
            x1, y1, x2, y2 = line[0]
            slope = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else 999  # Avoid division by zero

            # Threshold for considering a line as part of the left or right lane
            if abs(slope) < 0.3:  # Ignore lines with steep slopes (near vertical)
                continue

            if slope < 0:
                left_lines.append(line)
            else:
                right_lines.append(line)

        return left_lines, right_lines


    @staticmethod
    def find_x_on_line(line, y3):
        x1, y1, x2, y2 = line
        slope = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else 999  # Avoid division by zero
        y_intercept = y1 - slope * x1

        x3 = int((y3 - y_intercept) / slope) if slope != 0 else x1
        return x3

    @staticmethod
    def find_most_populous_line(image, lines, y_bottom):
        most_populous_line = None
        max_points = 0

        for line in lines:
            x1, y1, x2, y2 = line[0]
            points_on_line = np.sum(cv2.line(np.zeros_like(image), (x1, y1), (x2, y2), (0, 0, 255), 1))

            if points_on_line > max_points:
                most_populous_line = line
                max_points = points_on_line

        # if most_populous_line is not None:
        #     x3 = LanesDetectionOnFrame.find_x_on_line(most_populous_line, y_bottom)
        #     x1, y1, x2, y2 = most_populous_line[0]
        #     if y1 > y2:
        #         most_populous_line = (x3, y_bottom, x2, y2)
        #     else:
        #         most_populous_line = (x3, y_bottom, x1, y1)

        return most_populous_line

    @staticmethod
    def extend_line(line, y_bottom):
        x3 = LanesDetectionOnFrame.find_x_on_line(line, y_bottom)
        x1, y1, x2, y2 = line
        if y1 > y2:
            extended_line = (x3, y_bottom, x2, y2)
        else:
            extended_line = (x3, y_bottom, x1, y1)

        return extended_line

    @staticmethod
    def find_lane_lines(image):
        processed_image = LanesDetectionOnFrame.preprocess_image(image)
        # Edge Detection
        edges = cv2.Canny(processed_image, 50, 150)
        height, width = edges.shape

        roi_left_vertices = [(230, height),
                             (LanesDetectionOnFrame.CENTER_LINE_X - LanesDetectionOnFrame.OFFSET, height),
                             (LanesDetectionOnFrame.CENTER_LINE_X - LanesDetectionOnFrame.OFFSET, 360)]

        mask_left = np.zeros_like(edges)
        cv2.fillPoly(mask_left, [np.array(roi_left_vertices)], 255)

        masked_edges_left = cv2.bitwise_and(edges, mask_left)
        plt.imshow(masked_edges_left)
        plt.show()

        # Hough Transform for right mask

        roi_right_vertices = [(LanesDetectionOnFrame.CENTER_LINE_X + LanesDetectionOnFrame.OFFSET, height),
                              (700, height), (LanesDetectionOnFrame.CENTER_LINE_X + LanesDetectionOnFrame.OFFSET, 360)]

        mask_right = np.zeros_like(edges)
        cv2.fillPoly(mask_right, [np.array(roi_right_vertices)], 255)

        masked_edges_right = cv2.bitwise_and(edges, mask_right)
        plt.imshow(masked_edges_right)
        plt.show()

        # Hough Transform for left mask
        lines_right = cv2.HoughLinesP(masked_edges_right, 6, np.pi / 180, threshold=50, minLineLength=60,
                                      maxLineGap=200)
        lines_left = cv2.HoughLinesP(masked_edges_left, 6, np.pi / 90, threshold=50, minLineLength=60, maxLineGap=200)

        roi_vertices = [(230, height), (700, height), (height - 80, 360)]

        # mask = np.zeros_like(edges)

        # Average Lines and Extend to the Bottom
        y_bottom = height
        most_populous_left_line = None
        most_populous_right_line = None
        if lines_left is not None:
            most_populous_left_line = LanesDetectionOnFrame.find_most_populous_line(image, lines_left, y_bottom)
            most_populous_left_line = LanesDetectionOnFrame.extend_line(most_populous_left_line[0], y_bottom)

        if lines_right is not None:
            most_populous_right_line = LanesDetectionOnFrame.find_most_populous_line(image, lines_right, y_bottom)
            most_populous_right_line = LanesDetectionOnFrame.extend_line(most_populous_right_line[0], y_bottom)

        # Draw Lane Lines on Original Image
        if most_populous_left_line is not None:
            x1, y1, x2, y2 = most_populous_left_line
            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 8)

        if most_populous_right_line is not None:
            x1, y1, x2, y2 = most_populous_right_line
            cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 8)

        # Display the result
        plt.imshow(image)
        plt.show()

        return image
