import cv2
import numpy as np
import matplotlib.pyplot as plt

frame = cv2.imread('./result/frame_0105.png')

# Convert the frame to grayscale
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise and improve edge detection
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Apply Canny edge detection to find edges in the image
edges = cv2.Canny(blurred, 50, 150)

plt.imshow(edges)
plt.show()

height, width = frame.shape[:2]
roi_vertices = [(33, 400), (148, 350), (700, 360), (800, 420)]
mask_color = 255
mask = np.zeros_like(edges)
cv2.fillPoly(mask, np.array([roi_vertices], dtype=np.int32), mask_color)
masked_edges = cv2.bitwise_and(edges, mask)

plt.imshow(edges)
plt.show()
exit()

# Draw the detected lines on a copy of the original frame
frame_with_Contours = frame.copy()
# Use contours to find potential crosswalk areas
contours, _ = cv2.findContours(masked_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

font = cv2.FONT_HERSHEY_SIMPLEX
crosswalks = []

for idx, contour in enumerate(contours):
    area = cv2.contourArea(contour)
    # Get the bounding box of the contour
    x, y, w, h = cv2.boundingRect(contour)

    if (area > 3 and (w >= 20) and w <= 110 and h >= 20 and h <= 30):
        # Draw contours within the ROI
        cv2.drawContours(frame_with_Contours, [contour], -1, (0, 0, 255), 2)
        cv2.putText(frame_with_Contours, f"{idx}", (x + w // 2, y + h // 2),
                    font, 0.5, (255, 0, 0), 2, cv2.LINE_AA)
        crosswalks.append((idx, x, y, w, h))

# Check if we have at least 3 crosswalks
if len(crosswalks) >= 3:
    # Sort crosswalks by x-axis
    crosswalks.sort(key=lambda cw: cw[1])

    # Check differences between consecutive crosswalks
    differences = [crosswalks[i + 1][1] - crosswalks[i][1] for i in range(len(crosswalks) - 1)]

    # Check if differences fall within the specified range
    if all(40 <= diff <= 70 for diff in differences):
        # Add caution message on the image
        caution_text = "Watch out, there are crosswalks!"
        cv2.putText(frame_with_Contours, caution_text, (10, 30),
                    font, 1, (0, 0, 255), 2, cv2.LINE_AA)
        # Draw a red border around the caution message
        cv2.rectangle(frame_with_Contours, (10, 10), (550, 50), (0, 0, 255), 2)

plt.imshow(frame_with_Contours)
plt.show()