import cv2
import numpy as np


class DistanceDetectionOnFrame:
    def __init__(self, center_line_x, y_depth_search, right_lane_x, left_lane_x, offset, medium_warning_distance,
                 high_warning_distance, debug=False):
        # Variables to calibrate the frame
        self.center_line_x = center_line_x
        self.y_depth_search = y_depth_search
        self.right_lane_x = right_lane_x
        self.left_lane_x = left_lane_x
        self.offset = offset
        self.debug = debug

        self.high_warning_distance = high_warning_distance
        self.medium_warning_distance = medium_warning_distance

    @staticmethod
    def filter_black(frame):
        hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)

        # Define range of black color in HLS
        lower_black = np.array([0, 0, 0])  # Lower range of black (adjust if necessary)
        upper_black = np.array([180, 60, 60])  # Upper range of black (adjust if necessary)

        # Threshold the HLS image to get only black colors
        mask = cv2.inRange(hls, lower_black, upper_black)
        return mask

    @staticmethod
    def calculate_mask(roi, edge_image):
        mask = np.zeros_like(edge_image)
        cv2.fillPoly(mask, [np.array(roi)], 255)
        mask_edge = cv2.bitwise_and(edge_image, mask)
        return mask_edge

    def detect_vehicles(self, frame, center_line_x, y_depth_search, right_lane_x, left_lane_x, offset):
        min_width = 40
        min_height = 40
        max_width = 200
        max_height = 200

        # Convert frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = gray_frame.shape
        mask = self.filter_black(frame)
        only_dark_result = cv2.bitwise_and(frame, frame, mask=mask)

        # Convert the result to binary
        binary = cv2.cvtColor(only_dark_result, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(binary, 1, 255, cv2.THRESH_BINARY)

        roi_center_vertices = [(left_lane_x, height),
                               (right_lane_x, height),
                               (center_line_x + offset, y_depth_search),
                               (center_line_x - offset, y_depth_search)]

        binary = self.calculate_mask(roi_center_vertices, binary)

        # Find contours in the foreground mask
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Initialize list to store bounding boxes of detected vehicles
        vehicle_boxes = []

        # Iterate through contours
        for contour in contours:
            # Calculate the area of the contour
            area = cv2.contourArea(contour)

            # If the contour area is too small, ignore it (noise)
            # if area < min_area_threshold:
            #     continue

            # Get the bounding box of the contour
            x, y, w, h = cv2.boundingRect(contour)

            # Filter out bounding boxes that are too small or too large to be vehicles
            if max_width > w > min_width and max_height > h > min_height:
                vehicle_boxes.append((x, y, x + w, y + h))

        return vehicle_boxes

    def draw_roi_lines_on_image(self, image, roi_points):
        if self.debug:
            overlay = image.copy()
            cv2.polylines(overlay, [np.array(roi_points)], isClosed=True, color=(0, 255, 255), thickness=4)
            alpha = 0.2
            cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

    def annotate_frame(self, frame, annotated_frame):
        vehicles = self.detect_vehicles(frame, self.center_line_x, self.y_depth_search, self.right_lane_x,
                                        self.left_lane_x,
                                        self.offset)
        # 3 channel frame
        height = frame.shape[0]
        roi_center_vertices = [(self.left_lane_x, height),
                               (self.right_lane_x, height),
                               (self.center_line_x + self.offset, self.y_depth_search),
                               (self.center_line_x - self.offset, self.y_depth_search)]

        self.draw_roi_lines_on_image(annotated_frame, roi_center_vertices)

        green = (0, 255, 0)
        red = (0, 0, 255)
        yellow = (0, 255, 255)
        color = green

        for (x1, y1, x2, y2) in vehicles:
            y_max = np.max([y1, y2])
            if y_max > self.medium_warning_distance:
                color = yellow
            if y_max > self.high_warning_distance:
                color = red
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

        return annotated_frame
