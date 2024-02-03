import cv2
import numpy as np

# Global variables for input and output directories
input_directory = 'input_videos/'
output_directory = 'annotated_videos/'

# Read the video file
video_filename = 'cropped_change_lines_low_quality.mp4'
video_path = input_directory + video_filename

cap = cv2.VideoCapture(video_path)

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
output_filename = 'annotated_video.mp4'
output_path = output_directory + output_filename
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

while cap.isOpened():
    # Step 1: Read Frame from capture
    ret, frame = cap.read()
    if not ret:
        break

    # Step 2: Reading the Image
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    _, binary_threshold = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)

    # Step 4: Apply Gaussian Blur to the binary threshold image
    blur = cv2.GaussianBlur(binary_threshold, (5, 5), 0)

    # Step 4: Gaussian Blur
    edges = cv2.Canny(blur, 50, 150)

    roi_vertices = [(230, height), (700, height), (410, 360)]
    mask_color = 255
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, np.array([roi_vertices], dtype=np.int32), mask_color)
    masked_edges = cv2.bitwise_and(edges, mask)

    # Step 6: Region of Interest
    lines = cv2.HoughLinesP(masked_edges, rho=6, theta=np.pi/30, threshold=130, minLineLength=40, maxLineGap=50)

    # Step 7: Hough Transform
    line_image = np.zeros_like(frame)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 5)

    # Step 8: Drawing the Lines
    final_image = cv2.addWeighted(frame, 0.8, line_image, 1, 0)

    # Step 9: Overlaying the Lines on the Original Image
    cv2.imshow('Annotated Video', final_image)
    out.write(final_image)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()