from calibration_json.json_parser import camera_parameters
from cv2 import aruco
from cv2 import VideoCapture
from cv2.aruco import ArucoDetector
from cv2.typing import MatLike
from typing import Any, Dict, List, Sequence

import cv2
import numpy as np
import logging

log = logging.getLogger(__name__)

def get_aruco_info(frame_dict : Dict[int, MatLike],
                   arucoDetector:ArucoDetector) -> Dict[int, Dict[str, MatLike | Sequence[MatLike]]]:
    
    ID_0 = np.array([0])
    markerInfo = {0:{},1:{},2:{},3:{}}
                    
    for cam_number, frame in frame_dict.items():
        if frame is not None:
            corners, ids, _ = arucoDetector.detectMarkers(frame)
            filtered_corners, filtered_ids = filter_corners_ids(corners, ids, criteria=ID_0)
            log.info(f"Detected marker ID: {filtered_ids} with corners: {filtered_corners} from CAM {cam_number}")
            frame_marker = aruco.drawDetectedMarkers(frame.copy(), corners, ids)
            markerInfo[cam_number] = {  'corners': filtered_corners,
                                        'ids': filtered_ids, 
                                        'frame_marker': frame_marker } 
                      
                        
    return markerInfo


def get_centre(corners : MatLike) -> MatLike: 
    sum_x = 0
    sum_y = 0
    for corner in corners:       
        for points in corner[0]:
            x, y = points
            sum_x += x
            sum_y += y
    
    centre_x = 0.25 * sum_x
    centre_y = 0.25 * sum_y
    centre = np.array([centre_x, centre_y], dtype=int)
    
    return centre

def filter_corners_ids(corners : MatLike, ids : MatLike, criteria : np.ndarray) -> tuple[MatLike, MatLike]:
    filtered_corners = []
    filtered_ids = []
    if ids is not None:
        for id in ids:
            if id == criteria:
                idxs = np.where(id == criteria)
                filtered_ids.append(id)
                filtered_corners.append(corners[idxs[0][0]])

    filtered_corners = np.array(filtered_corners)
    filtered_ids = np.array(filtered_ids)  

    return filtered_corners, filtered_ids
    

def retrieve_cams_parameters() -> Dict[int, Dict[str, List | np.ndarray]]:
    
    path_str = './calibration_json/'
    file_extension = '.json'
    filenames = ['0', '1', '2', '3']
    files = []

    for filename in filenames:
        files.append(path_str + filename + file_extension)

    cams_params = {}

    for cam_number, file in enumerate(files):
        K, R, T, res, dis = camera_parameters(file)

        proj_matrix = np.eye(3)
        proj_matrix = np.hstack([proj_matrix, np.array([np.zeros(3)]).T])

        tf_matrix = np.hstack([R,T])
        tf_matrix = np.vstack([tf_matrix, np.array([[0,0,0,1]])])

        tf_inv = np.linalg.inv(tf_matrix)

        cams_params[cam_number] = { 'K':K,
                                    'R':R,
                                    'T':T,
                                    'PRJ_M':proj_matrix,
                                    'TF':tf_matrix,
                                    'TF_INV':tf_inv,
                                    'res':res,
                                    'dis':dis }
        
    

    return cams_params



def capture_videos() -> Dict[int, VideoCapture]:

    path_str = './videos/'
    file_extension = '.mp4'
    filenames = ['cam0', 'cam1', 'cam2', 'cam3']
    files = []
    video_capture = {}

    for filename in filenames:
        files.append(path_str + filename + file_extension)

    for cam_number, file in enumerate(files):
        video_capture[cam_number] = cv2.VideoCapture(file)

    return video_capture



