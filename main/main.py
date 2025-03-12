from calibration_json.json_parser import camera_parameters
from cv2 import aruco
import cv2
import numpy as np
import matplotlib.pyplot as plt
import logging


'''
TO-DO:
- [ ] MODULARIZAR O CODIGO COM FUNÇÕES PARA CADA ETAPA DO PROCESSO 
- [ ] GARANTIR QUE APENAS OS IDS=0 SEJAM DETECTADOS
- [ ] GARANTIR QUE NUMERO DE CORNERS = NUMERO DE IDS NO METODO drawDetectedMarkers
'''

logger = logging.getLogger(__name__)
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filemode='w',
    filename='log.txt', 
    encoding='utf-8', 
    level=logging.DEBUG)

logger.info('Starting the program')

# Read the intrinsic and extrinsic parameters of each camera
'''
n - Camera number
Kn - Parametros intrinsecos
Rn - Matriz de Rotacao
Tn - Vetor de Translacao
resn - Resolucao da Camera
disn - Distorcao Radial
'''
K0, R0, T0, res0, dis0 = camera_parameters('./calibration_json/0.json')
K1, R1, T1, res1, dis1 = camera_parameters('./calibration_json/1.json')
K2, R2, T2, res2, dis2 = camera_parameters('./calibration_json/2.json')
K3, R3, T3, res3, dis3 = camera_parameters('./calibration_json/3.json')

#Organize all in a dict 
cams = {
    0: {'K': K0, 'R': R0, 'T': T0, 'res': res0, 'dis': dis0},
    1: {'K': K1, 'R': R1, 'T': T1, 'res': res1, 'dis': dis1},
    2: {'K': K2, 'R': R2, 'T': T2, 'res': res2, 'dis': dis2},
    3: {'K': K3, 'R': R3, 'T': T3, 'res': res3, 'dis': dis3}
}

#Load Aruco Videos Frame by Frame 
cap0 = cv2.VideoCapture('./videos/cam0.mp4')
cap1 = cv2.VideoCapture('./videos/cam1.mp4')
cap2 = cv2.VideoCapture('./videos/cam2.mp4')
cap3 = cv2.VideoCapture('./videos/cam3.mp4')

#Capture Aruco ID=0
parameters =  aruco.DetectorParameters()
dictionary = aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
arucoDetector = aruco.ArucoDetector(dictionary, parameters)

while True:
    _, img0 = cap0.read()
    _, img1 = cap1.read()
    _, img2 = cap2.read()
    _, img3 = cap3.read()
    
    if img0 is None or img1 is None or img2 is None or img3 is None:
        if img0 is None:
            logger.debug('Error reading frame from camera 0')
        elif img1 is None:
            logger.debug('Error reading frame from camera 1')
        elif img2 is None:
            logger.debug('Error reading frame from camera 2')
        elif img3 is None:
            logger.debug('Error reading frame from camera 3')
        
        break

    corners0, ids0, rejectedImgPoints0 = arucoDetector.detectMarkers(img0)
    corners1, ids1, rejectedImgPoints1 = arucoDetector.detectMarkers(img1)
    corners2, ids2, rejectedImgPoints2 = arucoDetector.detectMarkers(img2)
    corners3, ids3, rejectedImgPoints3 = arucoDetector.detectMarkers(img3)
    
    #Select only not None Ids
    if ids0 is not None and ids1 is not None and ids2 is not None and ids3 is not None:
        ids0 = ids0.flatten()
        ids1 = ids1.flatten()
        ids2 = ids2.flatten()
        ids3 = ids3.flatten()

    
    logger.debug(f'Ids0: {ids0}')

    # frame_markers0 = aruco.drawDetectedMarkers(img0.copy(), corners0, ids0)
    # frame_markers1 = aruco.drawDetectedMarkers(img1.copy(), corners1, ids1)
    # frame_markers2 = aruco.drawDetectedMarkers(img2.copy(), corners2, ids2)
    # frame_markers3 = aruco.drawDetectedMarkers(img3.copy(), corners3, ids3)

    # cv2.imshow('cam0', frame_markers0)
    # cv2.imshow('cam1', frame_markers1)
    # cv2.imshow('cam2', frame_markers2)
    # cv2.imshow('cam3', frame_markers3)

    # Quit by clicking 'q'
    if cv2.waitKey(1) == ord('q'):
        break


cv2.destroyAllWindows()
logger.info('Program finished')

