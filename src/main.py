import cv2
import numpy as np

from src.LanesDetectionOnFrame import LanesDetectionOnFrame


def record_movie(video_input_path, video_output_path, detection_func):
    cap = cv2.VideoCapture(video_input_path)

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        # Step 1: Read Frame from capture
        ret, frame = cap.read()
        if not ret:
            break

        frame_annotations = detection_func(frame)

        final_image = cv2.addWeighted(frame, 0.2, frame_annotations, 1, 0)

        cv2.imshow('Annotated Video', final_image)
        out.write(final_image)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    input_directory = '../input_videos/'
    output_directory = '../annotated_videos/'

    # 1. lanes changing
    video_input_filename = 'change_lanes.mp4'
    video_output_filename = f"{video_input_filename.split('.')[0]}_annotated.mp4"
    vid1_param_dict = {
        'center_line_x': 425,
        'y_depth_search': 386,
        'right_lane_x': 670,
        'left_lane_x': 190,
        'offset': 80,
        'frame_rate_per_second': 30,
        'saturation_threshold': 120,
        'rho': 6,
        'theta': np.pi / 90,
        'min_line_length': 25,
        'debug': True
    }

    vid1_lane_detection = LanesDetectionOnFrame(**vid1_param_dict)
    record_movie(input_directory + video_input_filename, output_directory + video_output_filename,
                 vid1_lane_detection.find_lane_lines)

    # 2. Night-Time Lane Detection
    video_input_filename = 'night_ride.mp4'
    video_output_filename = f"{video_input_filename.split('.')[0]}_annotated.mp4"
    vid2_param_dict = {
        'center_line_x': 434,
        'y_depth_search': 350,
        'right_lane_x': 706,
        'left_lane_x': 218,
        'offset': 80,
        'frame_rate_per_second': 30,
        'saturation_threshold': 170,
        'rho': 6,
        'theta': np.pi / 90,
        'min_line_length': 25,
        'debug': True
    }

    vid2_lane_detection = LanesDetectionOnFrame(**vid2_param_dict)
    record_movie(input_directory + video_input_filename, output_directory + video_output_filename,
                 vid2_lane_detection.find_lane_lines)

    # 3. distance approximation detection
    video_input_filename = 'distance_approximation.mp4'
    video_output_filename = f"{video_input_filename.split('.')[0]}_annotated.mp4"
    vid3_param_dict = {
        'center_line_x': 410,
        'y_depth_search': 380,
        'right_lane_x': 605,
        'left_lane_x': 218,
        'offset': 30,
        'frame_rate_per_second': 30,
        'saturation_threshold': 150,
        'rho': 6,
        'theta': np.pi / 60,
        'min_line_length': 50,
        'blind_detection_offset': 21,
        'debug': True
    }

    vid3_lane_detection = LanesDetectionOnFrame(**vid3_param_dict)
    record_movie(input_directory + video_input_filename, output_directory + video_output_filename,
                 vid3_lane_detection.find_lane_lines)
