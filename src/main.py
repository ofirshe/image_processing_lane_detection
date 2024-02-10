import cv2
import numpy as np

from DistanceDetectionOnFrame import DistanceDetectionOnFrame
from LanesDetectionOnFrame import LanesDetectionOnFrame
from CrosswalkDetectionOnFrame import CrosswalkDetectionOnFrame

def record_movie(video_input_path, video_output_path, lanes_detection_class, distance_detection_class=None, crosswalk_detection_class=None):
    cap = cv2.VideoCapture(video_input_path)

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_annotations = lanes_detection_class.find_lane_lines(frame.copy())

        if distance_detection_class is not None:
            frame_annotations = distance_detection_class.annotate_frame(frame=frame, annotated_frame=frame_annotations)
        
        if crosswalk_detection_class is not None:
            frame_annotations = crosswalk_detection_class.annotate_frame(frame=frame, annotated_frame=frame_annotations)

        final_image = cv2.addWeighted(frame, 0.2, frame_annotations, 1, 0)

        cv2.imshow('Annotated Video', final_image)
        out.write(final_image)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    input_directory = f'./input_videos/'
    output_directory = f'./annotated_videos/'

    # # 1. lanes changing
    # video_input_filename = 'change_lanes.mp4'
    # video_output_filename = f"{video_input_filename.split('.')[0]}_annotated.mp4"
    # vid1_param_dict = {
    #     'center_line_x': 425,
    #     'y_depth_search': 386,
    #     'right_lane_x': 670,
    #     'left_lane_x': 190,
    #     'offset': 80,
    #     'frame_rate_per_second': 30,
    #     'saturation_threshold': 120,
    #     'rho': 6,
    #     'theta': np.pi / 90,
    #     'min_line_length': 25,
    #     'debug': True
    # }

    # vid1_lane_detection = LanesDetectionOnFrame(**vid1_param_dict)
    # record_movie(input_directory + video_input_filename, output_directory + video_output_filename,
    #              vid1_lane_detection, distance_detection_class=None, crosswalk_detection_class=None)

    # # 2. Night-Time Lane Detection
    # video_input_filename = 'night_ride.mp4'
    # video_output_filename = f"{video_input_filename.split('.')[0]}_annotated.mp4"
    # vid2_param_dict = {
    #     'center_line_x': 434,
    #     'y_depth_search': 350,
    #     'right_lane_x': 706,
    #     'left_lane_x': 218,
    #     'offset': 80,
    #     'frame_rate_per_second': 30,
    #     'saturation_threshold': 170,
    #     'rho': 6,
    #     'theta': np.pi / 90,
    #     'min_line_length': 25,
    #     'debug': True
    # }

    # vid2_lane_detection = LanesDetectionOnFrame(**vid2_param_dict)
    # record_movie(input_directory + video_input_filename, output_directory + video_output_filename,
    #              vid2_lane_detection, distance_detection_class=None, crosswalk_detection_class=None)

    # # 3. Distance approximation detection
    # video_input_filename = 'distance_detection_2.mp4'
    # video_output_filename = f"{video_input_filename.split('.')[0]}_annotated.mp4"
    # vid3_lanes_detection_param_dict = {
    #     'center_line_x': 540,
    #     'y_depth_search': 377,
    #     'right_lane_x': 785,
    #     'left_lane_x': 300,
    #     'offset': 50,
    #     'frame_rate_per_second': 30,
    #     'saturation_threshold': 150,
    #     'rho': 6,
    #     'theta': np.pi / 60,
    #     'min_line_length': 50,
    #     'blind_detection_offset': 30,
    #     'debug': False
    # }

    # vid3_distance_detection_param_dict = {
    #     'center_line_x': 540,
    #     'y_depth_search': 340,
    #     'right_lane_x': 770,
    #     'left_lane_x': 300,
    #     'offset': 15,
    #     'high_warning_distance': 420,
    #     'medium_warning_distance': 300,
    #     'debug': False
    # }

    # vid3_lane_detection = LanesDetectionOnFrame(**vid3_lanes_detection_param_dict)
    # vid3_distance_detection = DistanceDetectionOnFrame(**vid3_distance_detection_param_dict)

    # record_movie(input_directory + video_input_filename, output_directory + video_output_filename,
    #              vid3_lane_detection, distance_detection_class=vid3_distance_detection, crosswalk_detection_class=None)

    # 4. Crosswalk Detection
    video_input_filename = 'crosswalks_detection.mp4'
    video_output_filename = f"{video_input_filename.split('.')[0]}_annotated.mp4"
    vid4_lanes_detection_param_dict = {
        'center_line_x': 530,
        'y_depth_search': 350,
        'right_lane_x': 800,
        'left_lane_x': 250,
        'offset': 30,
        'frame_rate_per_second': 30,
        'saturation_threshold': 120,
        'rho': 6,
        'theta': np.pi / 90,
        'min_line_length': 25,
        'debug': True
    }

    vid4_crosswalk_detection_param_dict = {
        'subsequent_frames': 20,
        'debug': False
    }

    vid4_lane_detection = LanesDetectionOnFrame(**vid4_lanes_detection_param_dict)
    vid4_crosswalk_detection = CrosswalkDetectionOnFrame(**vid4_crosswalk_detection_param_dict)

    record_movie(input_directory + video_input_filename, output_directory + video_output_filename,
                 vid4_lane_detection, distance_detection_class=None, crosswalk_detection_class=vid4_crosswalk_detection)

