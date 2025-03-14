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

def remove_outliers_sync(position_estimate : Dict[str, List | np.ndarray], gaussian : bool = True, n_stds: int =2) -> None:
    
    x, y, z = np.array(position_estimate['x']), np.array(position_estimate['y']), np.array(position_estimate['z'])
    
    if gaussian:
        mean_x, std_x = np.mean(x), np.std(x)
        mean_y, std_y = np.mean(y), np.std(y)
        mean_z, std_z = np.mean(z), np.std(z)

        mask_x = (mean_x - n_stds * std_x < x) & (x < mean_x + n_stds * std_x)
        mask_y = (mean_y - n_stds * std_y < y) & (y < mean_y + n_stds * std_y)
        mask_z = (mean_z - n_stds * std_z < z) & (z < mean_z + n_stds * std_z)
    else:
        mask_x = (x > -2.0) & (x < 2.0)
        mask_y = (y > -1.0) & (x < 1.0)
        mask_z = (z >  0.0) & (z < 2.0)
        
    mask = mask_x & mask_y & mask_z

    position_estimate['x'] = x[mask].tolist()
    position_estimate['y'] = y[mask].tolist()
    position_estimate['z'] = z[mask].tolist()

def get_aruco_info(frame_dict : Dict[int, MatLike],
                   arucoDetector:ArucoDetector) -> Dict[int, Dict[str, MatLike | Sequence[MatLike]]]:
    
    ID_0 = np.array([0])

    markerInfo = {0:{},1:{},2:{},3:{}}
                    
    for cam_number, frame in frame_dict.items():
        if frame is not None:
            corners, ids, _ = arucoDetector.detectMarkers(frame)
            filtered_corners, filtered_ids = filter_corners_ids(corners, ids, criteria=ID_0)
            #log.info(f"Detected marker ID: {filtered_ids} with corners: {filtered_corners} from CAM {cam_number}")
            frame_marker = aruco.drawDetectedMarkers(frame.copy(), filtered_corners, filtered_ids)
            markerInfo[cam_number] = {  'corners': filtered_corners,
                                        'ids': filtered_ids, 
                                        'frame_marker': frame_marker } 
                      
                        
    return markerInfo


def get_center(corners : MatLike) -> MatLike: 
    sum_x = 0
    sum_y = 0

    if len(corners) > 0:
        for corner in corners:       
            for points in corner[0]:
                x, y = points
                sum_x += x
                sum_y += y
            
        centre_x = 0.25 * sum_x
        centre_y = 0.25 * sum_y

        center_float = np.array([centre_x, centre_y], dtype=np.float64)
        center_int = np.array([centre_x, centre_y], dtype=np.integer)
        
        center = {'i':center_int, 'f':center_float}
        return center
    else:
        return None
    

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



