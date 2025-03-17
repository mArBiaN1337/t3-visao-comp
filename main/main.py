from setup.setup_main import *
from cv2 import aruco
import cv2
import numpy as np
import matplotlib.pyplot as plt
import logging

'''
TO-DO
 ILUSTRAÇÃO DA TRAJETÓRIA ESTÁ BOA, MAS COM ALGUNS PONTOS QUE REPRESENTAM ALGUNS PROBLEMAS:
    - RUÍDO?
    - PROBLEMA NO FILTRO DE IDS OU DE CORNERS?

[] - ESTUDAR MANEIRA DE FILTRAR ISSO
[] - FECHAR AS JANELAS QUANDO O VIDEO ACABAR, SEM O USO DO INPUT Q
'''

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

WINDOW_SIZE = 950, 500

WINDOW_OFFSETS = {  0:(0,0),
                    1:(WINDOW_SIZE[0],0),
                    2:(0,WINDOW_SIZE[1]),
                    3:(WINDOW_SIZE[0],WINDOW_SIZE[1]) }

DISPLAY = True

#Get cam parameters from json files
cams_params = retrieve_cams_parameters()

#Load Aruco Videos Frame by Frame 
cams_videos = capture_videos()

#Setup Aruco Detector
parameters =  aruco.DetectorParameters()
dictionary = aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
arucoDetector = aruco.ArucoDetector(dictionary, parameters)

center_recognition = {0:{},1:{},2:{},3:{}}
position_estimate = {'x':[], 'y':[], 'z':[]}

try:   
    print("Or Press 'Ctrl+C'.")
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
                if DISPLAY:
                    window_name = f"CAM{cam_number}"
                    cv2.imshow(window_name, marker_info[cam_number]['frame_marker'])
                    cv2.moveWindow(window_name, WINDOW_OFFSETS[cam_number][0], WINDOW_OFFSETS[cam_number][1])
                    cv2.resizeWindow(window_name, WINDOW_SIZE[0], WINDOW_SIZE[1])
                    
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

        b_matrix = np.zeros([3*num_recog, 4 + num_recog])
        for i in range(num_recog):
            if len(proj_m_lst) > 0 and len(m_dot_lst) > 0:
                proj_m = proj_m_lst[i]
                m_dot = m_dot_lst[i]

                b_matrix[3*i : 3*(i+1), 0:4] = proj_m
                b_matrix[3*i : 3*(i+1), 4+i] = m_dot[:,0]

        U, S, Vt = np.linalg.svd(b_matrix)
        Vt = Vt.T
        pos_est = Vt[0:4,-1]

        if pos_est[-1] != np.float64(0):
            pos_est = pos_est / pos_est[-1]
            x, y, z = pos_est[0], pos_est[1], pos_est[2]

            if z > np.float64(0):
                position_estimate['x'].append(x)
                position_estimate['y'].append(y)
                position_estimate['z'].append(z)

        if cv2.waitKey(1) == ord('q') or cv2.waitKey(1) == ord('Q'):
            break
        

except KeyboardInterrupt: pass
finally:
    cv2.destroyAllWindows()


try:
    print("Press 'Ctrl+C' to quit 3D plot")
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    for i in range(len(position_estimate.get('x'))):
        log.info("x:{:.2f} y:{:.2f} z:{:.2f}".format(
            position_estimate.get('x')[i],
            position_estimate.get('y')[i],
            position_estimate.get('z')[i],
        ))

    ax.scatter(position_estimate['x'],position_estimate['y'],position_estimate['z'],c='g')
    ax.set_title('Position Estimate 3D - Calibrated Cameras')
    ax.axes.set_xlim([-2,2])
    ax.axes.set_ylim([-1,1])
    ax.axes.set_zlim([0,1])
    ax.grid(visible=True)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

    log.info(f"ALTURA MEDIA DO OBJETO: {np.mean(position_estimate['z']) * 100 : .2f} cm")

except KeyboardInterrupt: pass

finally: plt.close('all')

log.info('Program finished')



