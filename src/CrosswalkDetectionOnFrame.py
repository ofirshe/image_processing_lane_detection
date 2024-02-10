import cv2
import numpy as np

class CrosswalkDetectionOnFrame:
    def __init__(self, subsequent_frames, debug=False):
        # Variables to calibrate the frame
        self.subsequent_frames = subsequent_frames
        self.subsequent_frames_counter = subsequent_frames
        self.debug = debug
        self.caution_text = "Watch out, there are crosswalks!"

    def annotate_frame(self, frame, annotated_frame):
        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise and improve edge detection
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Apply Canny edge detection to find edges in the image
        edges = cv2.Canny(blurred, 50, 150)

        height, width = frame.shape[:2]
        roi_vertices = [(33, 400), (148, 350), (700, 360), (800, 420)]
        mask_color = 255
        mask = np.zeros_like(edges)
        cv2.fillPoly(mask, np.array([roi_vertices], dtype=np.int32), mask_color)
        masked_edges = cv2.bitwise_and(edges, mask)

        # Use contours to find potential crosswalk areas
        contours, _ = cv2.findContours(masked_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        font = cv2.FONT_HERSHEY_SIMPLEX
        crosswalks = []
        possible_contours = []

        for idx, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            # Get the bounding box of the contour
            x, y, w, h = cv2.boundingRect(contour)

            if (area > 3 and (w >= 20) and w <= 110 and h >= 20 and h <= 30):
                # Draw contours within the ROI
                crosswalks.append((x, y, w, h))
                possible_contours.append(contour)

        # Check if we have at least 3 crosswalks
        if len(crosswalks) >= 3:
            # Sort crosswalks by x-axis
            crosswalks.sort(key=lambda cw: cw[0])

            # Check differences between consecutive crosswalks
            differences = [crosswalks[i + 1][0] - crosswalks[i][0] for i in range(len(crosswalks) - 1)]

            # Check if differences fall within the specified range
            if all(40 <= diff <= 70 for diff in differences):
                # Add caution message on the image
                cv2.putText(annotated_frame, self.caution_text, (10, 30),
                            font, 1.5, (0, 0, 255), 2, cv2.LINE_AA)

                for contour in possible_contours:
                    cv2.drawContours(annotated_frame, [contour], -1, (0, 0, 255), 2)

                self.subsequent_frames_counter -= 1
        
        if self.subsequent_frames_counter < 20:
            if self.subsequent_frames_counter == 0:
                self.subsequent_frames_counter = 20
            else:
                cv2.putText(annotated_frame, self.caution_text, (10, 30),
                            font, 1.5, (0, 0, 255), 2, cv2.LINE_AA)
                self.subsequent_frames_counter -= 1

        return annotated_frame