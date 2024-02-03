import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Importing Libraries
image = cv2.imread('last_frame_image.jpg')
lane_image = image.copy()

# Step 2: Reading the Image
gray = cv2.cvtColor(lane_image, cv2.COLOR_BGR2GRAY)

# Step 3: Converting to Grayscale
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# Step 4: Gaussian Blur
edges = cv2.Canny(blur, 50, 150)

# Step 5: Canny Edge Detection
height, width = image.shape[:2]
roi_vertices = [(230, height), (700, height), (410, 360)]
mask_color = 255
mask = np.zeros_like(edges)
cv2.fillPoly(mask, np.array([roi_vertices], dtype=np.int32), mask_color)
masked_edges = cv2.bitwise_and(edges, mask)

# Step 6: Region of Interest
lines = cv2.HoughLinesP(masked_edges, rho=6, theta=np.pi/60, threshold=120, minLineLength=40, maxLineGap=25)

# Step 7: Hough Transform
line_image = np.zeros_like(image)
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 5)

# Step 8: Drawing the Lines
final_image = cv2.addWeighted(image, 0.8, line_image, 1, 0)

# Step 9: Overlaying the Lines on the Original Image
output_image_path = 'last_frame_image_annotated.jpg'
cv2.imwrite(output_image_path, final_image)

# Step 10: Display Image