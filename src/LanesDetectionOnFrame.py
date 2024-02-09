import cv2
import numpy as np
import matplotlib.pyplot as plt


class LanesDetectionOnFrame:
    LEFT_ANNOTATION = "Left <-"
    RIGHT_ANNOTATION = "Right ->"

    def __init__(self, center_line_x, y_depth_search, right_lane_x, left_lane_x, offset, frame_rate_per_second,
                 saturation_threshold, rho, theta, min_line_length, blind_detection_offset=1, debug=False):
        # Variables to calibrate the frame
        self.center_line_x = center_line_x
        self.y_depth_search = y_depth_search
        self.right_lane_x = right_lane_x
        self.left_lane_x = left_lane_x
        self.offset = offset
        self.frame_rate_per_second = frame_rate_per_second
        self.saturation_threshold = saturation_threshold
        self.blind_detection_offset = blind_detection_offset

        # hough transform parameters
        self.rho = rho
        self.theta = theta
        self.min_line_length = min_line_length

        # Variables to keep track of the state of the road
        self.crossing_roads = False
        self.counter_frame = 0
        self.crossing_roads_lines = []
        self.cross_direction = None
        self.debug = debug

        self.high_warning_distance = 400
        self.medium_warning_distance = 390

        self.colors = []

    def preprocess_image(self, image):
        height, width, _ = image.shape

        # Convert to HLS color space and threshold the saturation channel
        hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
        sat_binary = cv2.inRange(hls, (0, self.saturation_threshold, 0), (255, 255, 255))

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
    def slope(x1, y1, x2, y2):
        return (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else 999  # Avoid division by zero

    @staticmethod
    def find_x_on_line(line, y3):
        x1, y1, x2, y2 = line
        slope = LanesDetectionOnFrame.slope(x1, y1, x2, y2)
        y_intercept = y1 - slope * x1

        x3 = int((y3 - y_intercept) / slope) if slope != 0 else x1
        return x3

    @staticmethod
    def find_most_populous_line(image, lines, orientation=None):
        most_populous_line = None
        max_points = 0

        for line in lines:
            x1, y1, x2, y2 = line[0]

            slope = LanesDetectionOnFrame.slope(x1, y1, x2, y2)

            if abs(slope) < 0.4:  # Ignore lines with steep slopes (near vertical)
                continue

            if orientation == "right" and slope < 0:  # Ignore lines with negative slopes
                continue
            elif orientation == "left" and slope > 0:  # Ignore lines with positive slopes
                continue
            elif orientation == "center" and abs(slope) < 0.8:  # In the center we expect almost vertical lines
                continue

            points_on_line = np.sum(cv2.line(np.zeros_like(image), (x1, y1), (x2, y2), (0, 0, 255), 1))

            if points_on_line > max_points:
                most_populous_line = line
                max_points = points_on_line

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
    def calculate_mask(roi, edge_image):
        mask = np.zeros_like(edge_image)
        cv2.fillPoly(mask, [np.array(roi)], 255)
        mask_edge = cv2.bitwise_and(edge_image, mask)
        return mask_edge

    def detect_middle_line(self, image, edges):
        height, width = edges.shape
        y_bottom = height

        roi_middle_vertices = [(self.center_line_x - 70, height),
                               (self.center_line_x - 50, self.y_depth_search),
                               (self.center_line_x + 50, self.y_depth_search),
                               (self.center_line_x + 70, height)
                               ]

        self.draw_roi_lines_on_image(image, roi_middle_vertices)
        masked_edges_middle = self.calculate_mask(roi_middle_vertices, edges)
        lines_middle = cv2.HoughLinesP(masked_edges_middle, 1, np.pi / 180, threshold=30, minLineLength=30,
                                       maxLineGap=200)
        # plt.imshow(masked_edges_middle)
        # plt.show()

        if lines_middle is not None:
            most_populous_middle_line = self.find_most_populous_line(image, lines_middle, orientation="center")
            if most_populous_middle_line is not None:
                most_populous_middle_line = self.extend_line(most_populous_middle_line[0], y_bottom)
                self.draw_annotation_lane(image, most_populous_middle_line)
                return most_populous_middle_line

    def draw_roi_lines_on_image(self, image, roi_points):
        if self.debug:
            overlay = image.copy()
            cv2.polylines(overlay, [np.array(roi_points)], isClosed=True, color=(0, 0, 255), thickness=4)
            alpha = 0.2
            cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

    def calculate_line_x_distance_from_center(self, line):
        x1, y1, x2, y2 = line
        return abs(x1 - self.center_line_x)

    def check_if_changing_lanes(self, image, edges, right_lane_detected, left_lane_detected):
        threshold = 70
        # no side road lanes detected, checking for lane in the center
        if right_lane_detected is None and left_lane_detected is None:
            middle_lane_detected = self.detect_middle_line(image, edges)
            if middle_lane_detected is not None:
                self.crossing_roads_lines.append(middle_lane_detected)
                return self.calculate_line_x_distance_from_center(middle_lane_detected) < threshold

        # only left lane detected
        if right_lane_detected is None and left_lane_detected is not None:
            if self.calculate_line_x_distance_from_center(left_lane_detected) < threshold:
                self.crossing_roads_lines.append(left_lane_detected)
                return True
        # only right lane detected
        elif right_lane_detected is not None and left_lane_detected is None:
            if self.calculate_line_x_distance_from_center(right_lane_detected) < threshold:
                self.crossing_roads_lines.append(right_lane_detected)
                return True

        return False

    @staticmethod
    def draw_annotation_lane(image, lane):
        x1, y1, x2, y2 = lane
        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 8)

    def draw_annotation_on_image(self, image, lanes_detected):
        for lane in lanes_detected:
            self.draw_annotation_lane(image, lane)

        if self.cross_direction:
            cv2.putText(image, f"Changing lane to {self.cross_direction}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255), 2, cv2.LINE_AA)

    def find_lane_lines(self, image):
        lanes_detected = []
        processed_image = self.preprocess_image(image)
        # Edge Detection
        edges = cv2.Canny(processed_image, 50, 150)
        height, width = edges.shape
        y_bottom = height
        most_populous_left_line = None
        most_populous_right_line = None

        # Detecting and removing vertical lines
        # roi_center_vertices = [(self.left_lane_x-40, height),
        #                        (self.right_lane_x+40, height),
        #                        (self.center_line_x + self.offset+40, self.y_depth_search),
        #                        (self.center_line_x - self.offset-40, self.y_depth_search)]
        roi_center_vertices = [(self.left_lane_x - 25, height),
                               (self.right_lane_x + 25, height),
                               (self.center_line_x + self.offset + 25, self.y_depth_search),
                               (self.center_line_x - self.offset - 25, self.y_depth_search)]
        self.draw_roi_lines_on_image(image, roi_center_vertices)

        masked_edges_center = self.calculate_mask(roi_center_vertices, edges)
        center = cv2.HoughLinesP(masked_edges_center, 1, np.pi / 180, threshold=20, minLineLength=15, maxLineGap=10)

        mask = np.zeros_like(edges)

        vertical_lines_y = []

        if center is not None:
            for line in center:
                x1, y1, x2, y2 = line[0]
                slope = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else np.inf
                if abs(slope) < 0.2:  # Adjust the threshold as needed
                    cv2.line(mask, (x1, y1), (x2, y2), 255, 2)
                    cv2.line(image, (x1, y1), (x2, y2), (0, 255, 255), 5)
                    vertical_lines_y.append(np.min([y1, y2]))

            mask = cv2.bitwise_not(mask)
            cv2.erode(mask, (100, 100), mask, iterations=5)

            edges = cv2.bitwise_and(edges, edges, mask=mask)

        roi_left_vertices = [(self.left_lane_x, height),
                             (self.center_line_x - self.blind_detection_offset, height),
                             (self.center_line_x - self.blind_detection_offset, self.y_depth_search),
                             (self.center_line_x - self.blind_detection_offset - self.offset, self.y_depth_search)]

        roi_right_vertices = [(self.right_lane_x, height),
                              (self.center_line_x + self.blind_detection_offset, height),
                              (self.center_line_x + self.blind_detection_offset, self.y_depth_search),
                              (self.center_line_x + self.blind_detection_offset + self.offset, self.y_depth_search)]

        self.draw_roi_lines_on_image(image, roi_left_vertices)
        self.draw_roi_lines_on_image(image, roi_right_vertices)

        masked_edges_left = self.calculate_mask(roi_left_vertices, edges)

        masked_edges_right = self.calculate_mask(roi_right_vertices, edges)

        lines_right = cv2.HoughLinesP(masked_edges_right, self.rho, self.theta, threshold=40,
                                      minLineLength=self.min_line_length, maxLineGap=200)
        lines_left = cv2.HoughLinesP(masked_edges_left, self.rho, self.theta, threshold=40,
                                     minLineLength=self.min_line_length, maxLineGap=200)

        if lines_left is not None:
            most_populous_left_line = self.find_most_populous_line(image, lines_left, orientation="left")
            if most_populous_left_line is not None:
                most_populous_left_line = self.extend_line(most_populous_left_line[0], y_bottom)
                lanes_detected.append(most_populous_left_line)

        if lines_right is not None:
            most_populous_right_line = self.find_most_populous_line(image, lines_right, orientation="right")
            if most_populous_right_line is not None:
                most_populous_right_line = self.extend_line(most_populous_right_line[0], y_bottom)
                lanes_detected.append(most_populous_right_line)

        # possible scenario of changing lanes
        if most_populous_right_line is None or most_populous_left_line is None:
            if self.check_if_changing_lanes(image, edges, most_populous_right_line, most_populous_left_line):
                self.crossing_roads = True
                self.counter_frame = 1
        elif most_populous_right_line is not None and most_populous_left_line is not None:
            self.crossing_roads = False
            self.crossing_roads_lines = []

        if len(self.crossing_roads_lines) == 10:
            x1, y1, x2, y2 = self.crossing_roads_lines[0]
            x3, y3, x4, y4 = self.crossing_roads_lines[9]
            if x1 < x3:
                self.cross_direction = self.LEFT_ANNOTATION
            else:
                self.cross_direction = self.RIGHT_ANNOTATION

        # if lane crossing is no longer detected, linger the annotation for a one more second and then reset
        if not self.crossing_roads and self.counter_frame > 0:
            self.counter_frame += 1

        if self.counter_frame == self.frame_rate_per_second:
            self.counter_frame = 0
            self.cross_direction = None

        self.draw_annotation_on_image(image, lanes_detected)

        # Define the border width
        border_width = 10

        # green = [0, 255, 0]
        # red = [0, 0, 255]
        # yellow = [0, 255, 255]
        #
        # color = green
        #
        # if len(vertical_lines_y) > 0:
        #     print(np.max(vertical_lines_y))
        #     if self.high_warning_distance > np.max(vertical_lines_y) > self.medium_warning_distance:
        #         color = yellow
        #     elif np.max(vertical_lines_y) > self.high_warning_distance:
        #         color = red
        # if len(self.colors) > 0:
        #     if self.colors[-1] == red or self.colors[-1] == yellow:
        #         if color == green:
        #             print("Changing color to green")
        #
        #
        # image = self.add_borders(image, color)
        # self.colors.append(color)

        return image

    @staticmethod
    def add_borders(image, color):
        border_width = 10
        # Create a copy of the original image
        bordered_image = np.copy(image)

        # Set the border color for the top and bottom sides
        bordered_image[:border_width, :] = color
        bordered_image[-border_width:, :] = color

        # Set the border color for the left and right sides
        bordered_image[:, :border_width] = color
        bordered_image[:, -border_width:] = color
        return bordered_image
