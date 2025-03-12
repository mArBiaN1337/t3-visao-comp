from calibration_json.json_parser import camera_parameters
from cv2 import aruco
from cv2 import VideoCapture
from cv2.aruco import ArucoDetector
from cv2.typing import MatLike
from typing import Any, Dict, List, Sequence

import cv2
import numpy as np
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)

def get_aruco_info(img_list : List[MatLike],
                   arucoDetector:ArucoDetector) -> Dict[int, Dict[str, MatLike | Sequence[MatLike]]]:
    
    corners0, ids0, rejectedImgPoints0 = arucoDetector.detectMarkers(img_list[0])
    corners1, ids1, rejectedImgPoints1 = arucoDetector.detectMarkers(img_list[1])
    corners2, ids2, rejectedImgPoints2 = arucoDetector.detectMarkers(img_list[2])
    corners3, ids3, rejectedImgPoints3 = arucoDetector.detectMarkers(img_list[3])

    frame_markers0 = aruco.drawDetectedMarkers(img_list[0].copy(), corners0, ids0)
    frame_markers1 = aruco.drawDetectedMarkers(img_list[1].copy(), corners1, ids1)
    frame_markers2 = aruco.drawDetectedMarkers(img_list[2].copy(), corners2, ids2)
    frame_markers3 = aruco.drawDetectedMarkers(img_list[3].copy(), corners3, ids3)

    #IDS e CORNERS podem ser tratados aqui [AVALIAR] (ID=0 e diferente de None) 
    aruco_info_dict = {
        0:{'frame_markers':frame_markers0, 'corners':corners0, 'ids':ids0},
        1:{'frame_markers':frame_markers1, 'corners':corners1, 'ids':ids1},
        2:{'frame_markers':frame_markers2, 'corners':corners2, 'ids':ids2},
        3:{'frame_markers':frame_markers3, 'corners':corners3, 'ids':ids3}
    }

    return aruco_info_dict

def retrieve_cams_parameters() -> Dict[int, Dict[str, List | np.ndarray]]:
    
    K0, R0, T0, res0, dis0 = camera_parameters('./calibration_json/0.json')
    K1, R1, T1, res1, dis1 = camera_parameters('./calibration_json/1.json')
    K2, R2, T2, res2, dis2 = camera_parameters('./calibration_json/2.json')
    K3, R3, T3, res3, dis3 = camera_parameters('./calibration_json/3.json')

    cams_params = {
        0: {'K': K0, 'R': R0, 'T': T0, 'res': res0, 'dis': dis0},
        1: {'K': K1, 'R': R1, 'T': T1, 'res': res1, 'dis': dis1},
        2: {'K': K2, 'R': R2, 'T': T2, 'res': res2, 'dis': dis2},
        3: {'K': K3, 'R': R3, 'T': T3, 'res': res3, 'dis': dis3}
    }

    return cams_params

def capture_videos() -> Dict[int, VideoCapture]:
    
    cap0 = cv2.VideoCapture('./videos/cam0.mp4')
    cap1 = cv2.VideoCapture('./videos/cam1.mp4')
    cap2 = cv2.VideoCapture('./videos/cam2.mp4')
    cap3 = cv2.VideoCapture('./videos/cam3.mp4')

    return {0:cap0, 1:cap1, 2:cap2, 3:cap3}



