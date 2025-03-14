from setup.setup_main import *
from cv2 import aruco
import cv2
import numpy as np
import matplotlib.pyplot as plt
import logging

np.set_printoptions(precision=2, suppress=True)

log = logging.getLogger(__name__)
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filemode='w',
    filename='log.log', 
    encoding='utf-8', 
    level=logging.DEBUG)

log.info('Starting the program')
print("Program started - Press Q to Terminate")


# Read the intrinsic and extrinsic parameters of each camera
'''
n - Camera number
Kn - Parametros intrinsecos
Rn - Matriz de Rotacao
Tn - Vetor de Translacao
resn - Resolucao da Camera
disn - Distorcao Radial
'''

#Get cam parameters from json files
cams_params = retrieve_cams_parameters()
log.info(f'{cams_params}')

#Load Aruco Videos Frame by Frame 
cams_videos = capture_videos()

#Setup Aruco Detector
parameters =  aruco.DetectorParameters()
dictionary = aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
arucoDetector = aruco.ArucoDetector(dictionary, parameters)

while True:

    frame_dict : Dict[int, MatLike] = {}

    for cam_number, cam_cap in cams_videos.items():
        ret, frame = cam_cap.read()
        if ret:
            frame_dict[cam_number] = frame
        else:
            log.debug('Error reading from CAM {} - frame: {} '.format(cam_number, frame))
            continue

    marker_info = get_aruco_info(frame_dict, arucoDetector)
    for cam_number, frame in frame_dict.items():
        if frame is not None:
           centre = get_centre(marker_info[cam_number]['corners'])
           circle = cv2.circle(frame, centre, 8, (0,255,0), cv2.FILLED)
           cv2.imshow(f"CAM{cam_number}", frame) 

    # Quit by clicking 'q'
    if cv2.waitKey(1) == ord('q'):
        break    


cv2.destroyAllWindows()
log.info('Program finished')

