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
            frame_marker = aruco.drawDetectedMarkers(frame.copy(), filtered_corners, filtered_ids)
            markerInfo[cam_number] = {  'corners': filtered_corners,
                                        'ids': filtered_ids, 
                                        'frame_marker': frame_marker } 
                                             
    return markerInfo


def get_center(corners : MatLike) -> MatLike: 
    sum_x = 0
    sum_y = 0
    
    if len(corners) > 0:
        corners = corners.reshape(-1, 2)
        sum_x = np.sum(corners[:, 0])
        sum_y = np.sum(corners[:, 1])

        centre_x = 0.25 * sum_x
        centre_y = 0.25 * sum_y

        center_float = np.array([centre_x, centre_y], dtype=np.float16)
        center_int = np.array([centre_x, centre_y], dtype=np.int64)
        
        center = {'i':center_int, 'f':center_float}
        return center
    else:
        center_float = np.array([0.0, 0.0], dtype=np.float16)
        center_int = np.array([0, 0], dtype=np.int64)

        center = {'i':center_int, 'f':center_float}
        return center
  
def filter_corners_ids(corners : MatLike, ids : MatLike, criteria : np.ndarray) -> tuple[MatLike, MatLike]:
    if ids is None:
        return np.array([]), np.array([])

    filtered_corners = [corner for corner, id in zip(corners, ids) if id in criteria]
    filtered_ids = [id for id in ids if id in criteria]

    return np.array(filtered_corners, dtype=np.float64), np.array(filtered_ids, dtype=np.integer)
    

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

        PROJ_M = np.eye(3)
        PROJ_M = np.hstack([PROJ_M, np.array([np.zeros(3)]).T])

        TF_M = np.hstack([R,T])
        TF_M = np.vstack([TF_M, np.array([[0,0,0,1]])])

        TF_INV = np.linalg.inv(TF_M)

        GEN_PROJ = K @ PROJ_M @ TF_INV

        cams_params[cam_number] = { 'K':K,
                                    'R':R,
                                    'T':T,
                                    'PRJ_M':PROJ_M,
                                    'TF':TF_M,
                                    'TF_INV':TF_INV,
                                    'GEN_PROJ':GEN_PROJ,
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



