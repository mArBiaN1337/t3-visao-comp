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
    level=logging.INFO)

log.info('Starting the program')
print("Program started - Press Q to Close the Windows")

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

center_recognition = ({0:{},1:{},2:{},3:{}})
frame_counter = 0

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
            cv2.imshow(f"CAM{cam_number}", marker_info[cam_number]['frame_marker'])
            
            if len(marker_info[cam_number]['ids']) > 0:
               center_recognition[cam_number] = True
            else:
               center_recognition[cam_number] = False

    num_recog : int = 0  
    for cam_number, recognized in center_recognition.items():
        if recognized == True:
            num_recog += 1
    
    proj_m_lst = []
    m_dot_lst = []
    for cam_number, recognized in center_recognition.items():
        if recognized == True:
            if num_recog >= 2:
                if len(marker_info[cam_number]) > 0:
                    center = get_center(marker_info[cam_number]['corners'])
                    
                    center_point = np.array([center['f']])
                    center_point = np.hstack([center_point, np.array([[1]])]).T
                    proj_m = cams_params[cam_number]['GEN_PROJ']

                    proj_m_lst.append(proj_m)
                    m_dot_lst.append(center_point)

                    B = np.zeros([3*num_recog, num_recog + 4])

                    for i in range(num_recog)

                    # fazer matriz B para resolver sistema via SVD
                    # B tem dimensão 3n X (4 + n), onde n é o número
                    # de reconhecimentos (num_recog)

    log.info(f'{B} {num_recog}')
    
    # Quit by clicking 'Q'
    if cv2.waitKey(1) == ord('q') or cv2.waitKey(1) == ord('Q'): 
        break   


cv2.destroyAllWindows()

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter([1,0,0],[0,1,0],[0,0,1])
ax.set_title('Position Estimate 3D - Calibrated Cameras')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()

log.info('Program finished')



