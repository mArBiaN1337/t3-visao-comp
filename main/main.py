from setup.setup_main import *
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

log = logging.getLogger(__name__)
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filemode='w',
    filename='log.txt', 
    encoding='utf-8', 
    level=logging.DEBUG)

log.info('Starting the program')

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

#Load Aruco Videos Frame by Frame 
cams_videos = capture_videos()

#Setup Aruco Detector
parameters =  aruco.DetectorParameters()
dictionary = aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
arucoDetector = aruco.ArucoDetector(dictionary, parameters)

while True:
    _, img0 = cams_videos[0].read()
    _, img1 = cams_videos[1].read()
    _, img2 = cams_videos[2].read()
    _, img3 = cams_videos[3].read()

    img_list = [img0, img1, img2, img3]
    imgs_read = all(img is not None for img in img_list)
    log.debug(imgs_read)
    if imgs_read == True:

        aruco_info_dict = get_aruco_info(img_list, arucoDetector)

        cv2.imshow('cam0', aruco_info_dict[0]['frame_markers'])
        cv2.imshow('cam1', aruco_info_dict[1]['frame_markers'])
        cv2.imshow('cam2', aruco_info_dict[2]['frame_markers'])
        cv2.imshow('cam3', aruco_info_dict[3]['frame_markers'])

    # Quit by clicking 'q'
    if cv2.waitKey(1) == ord('q'):
        break    


cv2.destroyAllWindows()
log.info('Program finished')

