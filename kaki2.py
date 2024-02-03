import cv2

# Read the video file

input_directory = 'input_videos/'
video_filename = 'cropped_change_lines_low_quality.mp4'
video_path = input_directory + video_filename
cap = cv2.VideoCapture(video_path)

# Check if the video opened successfully
if not cap.isOpened():
    print(f"Error opening video file. Check if the file '{video_path}' exists and is a valid video file.")
    exit()

last_frame = None
# Read frames until the last frame is reached
while True:
    ret, frame = cap.read()

    # Break the loop if the frame is not read successfully (end of video)
    if not ret:
        break
    last_frame = frame

# Save the last frame as an image
output_image_path = 'last_frame_image.jpg'
cv2.imwrite(output_image_path, last_frame)

# Release the video capture object
cap.release()

print(f"The last frame has been saved as {output_image_path}")
