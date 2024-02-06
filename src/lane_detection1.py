import cv2
import numpy as np

from src.LanesDetectionOnFrame import LanesDetectionOnFrame

# Global variables for input and output directories
input_directory = '../input_videos/'
output_directory = '../annotated_videos/'

# Read the video file
# video_filename = 'change_lane_to_right.mp4'

video_filename = 'cropped_change_lines_low_quality.mp4'
video_path = input_directory + video_filename

cap = cv2.VideoCapture(video_path)

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
output_filename = f"video1.mp4"
output_path = output_directory + output_filename
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# frame calibration
center_line_x = 425
y_depth_search = 386
right_lane_x = 670
left_lane_x = 190
offset = 80

lanesDetectionOnFrame = LanesDetectionOnFrame(center_line_x, y_depth_search, right_lane_x, left_lane_x, offset,
                                              frame_rate_per_second=fps, debug=False)

while cap.isOpened():
    # Step 1: Read Frame from capture
    ret, frame = cap.read()
    if not ret:
        break

    line_image = lanesDetectionOnFrame.find_lane_lines(frame)

    # Step 8: Drawing the Lines
    final_image = cv2.addWeighted(frame, 0.2, line_image, 1, 0)

    # Step 9: Overlaying the Lines on the Original Image
    cv2.imshow('Annotated Video', final_image)
    out.write(final_image)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
