import cv2
import numpy as np
import matplotlib.pyplot as plt


class LanesDetectionOnFrame:
    LEFT_ANNOTATION = "Left <-"
    RIGHT_ANNOTATION = "Right ->"

    def __init__(self, center_line_x, offset, frame_rate_per_second):
        self.center_lane_x = 410
        self.y_depth_search = 386
        self.right_lane_x = 670
        self.left_lane_x = 180
        self.offset = offset
        self.frame_rate_per_second = frame_rate_per_second
        self.crossing_roads = False
        self.counter_frame = 0
        self.crossing_roads_lines = []
        self.cross_direction = None

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
    def find_most_populous_line(image, lines, orientation="right"):
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
        plt.imshow(mask)
        plt.show()
        mask_edge = cv2.bitwise_and(edge_image, mask)
        plt.imshow(mask_edge)
        plt.show()
        return mask_edge

    def find_lane_lines(self, image):
        processed_image = self.preprocess_image(image)
        # Edge Detection
        edges = cv2.Canny(processed_image, 50, 150)
        height, width = edges.shape
        y_bottom = height
        most_populous_left_line = None
        most_populous_right_line = None

        roi_left_vertices = [(self.left_lane_x, height),
                             (self.center_lane_x - 1, height),
                             (self.center_lane_x - 1, self.y_depth_search),
                             (self.center_lane_x - 1 - self.offset, self.y_depth_search)]

        roi_right_vertices = [(self.right_lane_x, height),
                              (self.center_lane_x + 1, height),
                              (self.center_lane_x + 1, self.y_depth_search),
                              (self.center_lane_x + 1 + self.offset, self.y_depth_search)]

        masked_edges_left = self.calculate_mask(roi_left_vertices, edges)

        masked_edges_right = self.calculate_mask(roi_right_vertices, edges)

        lines_right = cv2.HoughLinesP(masked_edges_right, 6, np.pi / 90, threshold=25, minLineLength=25, maxLineGap=200)
        lines_left = cv2.HoughLinesP(masked_edges_left, 6, np.pi / 90, threshold=25, minLineLength=25, maxLineGap=200)

        if lines_left is None:
            print("left line not found")
        if lines_right is None:
            print("right line not found")

        if lines_left is not None:
            self.crossing_roads = False
            self.crossing_roads_lines = []
            most_populous_left_line = self.find_most_populous_line(image, lines_left, orientation="left")
            if most_populous_left_line is not None:
                most_populous_left_line = self.extend_line(most_populous_left_line[0], y_bottom)

        if lines_right is not None:
            self.crossing_roads = False
            self.crossing_roads_lines = []
            most_populous_right_line = self.find_most_populous_line(image, lines_right, orientation="right")
            if most_populous_right_line is not None:
                most_populous_right_line = self.extend_line(most_populous_right_line[0], y_bottom)

        # possible change lane scenario
        # if lines_right is None and lines_left is None:
        #     roi_middle_vertices = [(350, height),
        #                            (370, 390),
        #                            (520, 390),
        #                            (550, height)
        #                            ]
        #
        #     masked_edges_middle = self.calculate_mask(roi_middle_vertices, edges)
        #
        #     lines_middle = cv2.HoughLinesP(masked_edges_middle, 6, np.pi / 180, threshold=30, minLineLength=30,
        #                                    maxLineGap=200)
        #     if lines_middle is not None:
        #         most_populous_middle_line = self.find_most_populous_line(image, lines_middle)
        #         if most_populous_middle_line is not None:
        #             self.crossing_roads = True
        #             self.counter_frame = 0
        #             most_populous_middle_line = self.extend_line(most_populous_middle_line[0],
        #                                                          y_bottom)
        #             self.crossing_roads_lines.append(most_populous_middle_line)
        #             x1, y1, x2, y2 = most_populous_middle_line
        #             cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 8)
        #
        # if len(self.crossing_roads_lines) == 5:
        #     x1, y1, x2, y2 = self.crossing_roads_lines[0]
        #     x3, y3, x4, y4 = self.crossing_roads_lines[4]
        #     if x1 < x3:
        #         self.cross_direction = self.LEFT_ANNOTATION
        #     else:
        #         self.cross_direction = self.RIGHT_ANNOTATION
        #
        # if self.cross_direction or 0 < self.counter_frame < self.frame_rate_per_second:
        #     cv2.putText(image, f"Changing lane to {self.cross_direction}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
        #                 (0, 0, 255), 2,
        #                 cv2.LINE_AA)
        #     self.counter_frame += 1
        #
        # if self.counter_frame == self.frame_rate_per_second:
        #     self.counter_frame = 0
        #     self.cross_direction = None

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
