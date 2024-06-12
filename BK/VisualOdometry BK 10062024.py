# region Imports
# from asyncio.windows_events import NULL
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use backend sem interface gráfica
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import pandas as pd
import os
import logging
from datetime import datetime
from tqdm import tqdm
import ManualPoints
import math
# from picamera2 import Picamera2
# endregion


pointsTestImage0 = ManualPoints.pointsTestImage0
pointsTestImage1 = ManualPoints.pointsTestImage1

def ConfigDataLogger():
    dataTimeNow = datetime.now()
    # Configuração básica do log
    OutputFolderDataLogger = f'DataLogger/DataLogger_{dataTimeNow.strftime("%m.%Y")}/{dataTimeNow.strftime("%d")}'

    os.makedirs(OutputFolderDataLogger, exist_ok=True)

    logging.basicConfig(filename=f'{OutputFolderDataLogger}/dataLogger_{dataTimeNow.strftime("%H")}h{dataTimeNow.strftime("%M")}m{dataTimeNow.strftime("%S")}s.txt',
                            level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    return logging.getLogger('dataLogger')
        
class Camera:
    def __init__(self, dataLogger):
        # Editable parameters
        self.liveON = False
        self.numFramesToLoad = 1500        
        self.idCamera = 0
        # end

        # self.dataLogger = dataLogger
        self.filePath = f'Recursos/00/image_{self.idCamera}'
        # self.filePath = "FotosPICamera"
        # self.filePath = f'Recursos/Teste'
        self.idFrame = 0
        self.framesStored = []
        self.framesLoaded = []
        self.intrinsicParameters = []
        self.projectionMatrix = []
        # self.picam2 = Picamera2()
        self.webCapture = cv2.VideoCapture(0)
        self.recaptureFrame = False

    def CalibrationFile(self):
        # Define o caminho para o arquivo de calibração
        # file = "CalibrationCam/calib.txt"
        file = "CalibrationCam/ParameteresCamera.txt"
        
        try:
            with open(file) as fileCalib:
                for line in fileCalib:
                    if line.startswith(f"P{self.idCamera}:"):
                        # Extrai os números da linha, ignorando o identificador "P0:"
                        elementos = np.fromstring(line[3:], sep=' ', dtype=np.float64)
                        # Reorganiza os elementos para formar a matriz de projeção 3x4
                        self.projectionMatrix = np.reshape(elementos, (3, 4))
                        # Extrai os parâmetros intrínsecos (as três primeiras colunas da matriz de projeção)
                        self.intrinsicParameters = self.projectionMatrix[:, :3]
                        print("Parâmetros intrínsecos:")
                        print(self.intrinsicParameters)
                        break  # Encerra o loop após processar a linha desejada

        except FileNotFoundError:
            print(f"Arquivo não encontrado: {file}")
            return

        # Assumindo que # self.dataLogger.info é um método válido para registrar informação
        # self.dataLogger.info(f'\nParâmetros Intrínsecos:\n{self.intrinsicParameters}')        
        # dataLogger.info(f'\n distortioncoefficient \n {distortioncoefficient}')

    def SetupFrames(self):
        if self.liveON is False:
            framePath = [os.path.join(self.filePath, file) for file in sorted(os.listdir(self.filePath))][:self.numFramesToLoad]
            self.framesStored = [cv2.imread(path) for path in framePath][:self.numFramesToLoad]
            return print( '\n Frames Loaded \n')
        if self.liveON is True:
            self.LiveCam()
    
    def LoadFrames(self):
        if self.liveON is False:
            if(len(self.framesLoaded) == 0):
                self.framesLoaded.append(self.framesStored[self.idFrame])
            else:
                if( (self.idFrame + 1) < len(self.framesStored) ):
                    self.framesLoaded.append(self.framesStored[self.idFrame + 1])
            self.idFrame = len(self.framesLoaded) - 1
            self.PrintFrame()
            return 

        if self.liveON is True:
            # Capture new frame
            self.LiveCam()
            self.idFrame = len(self.framesLoaded) - 1
            return print( '\n Live ON \n')
        
    def LoadVideo(self):
        videoPath = "Recursos/KittiVideo.mp4"
        # dataLogger.info(f'Video: {videoPath}')
        #CapturedVideo = cv2.VideoCapture(0) # Live
        CapturedVideo = cv2.VideoCapture(videoPath)
        CapturedVideo.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        # Verificar se o video foi carregado corretamente
        if not CapturedVideo.isOpened():
            print("Erro ao abrir o vídeo.")
            return -1
    
    def CreateVideoWithDataSetFrames(self, numFrames):
        if os.path.exists("Recursos/KittiVideo.mp4") is False:
            numFramesVideo = numFrames
            imagesDir = "Recursos/00/image_2"
            listImages = sorted(os.listdir(imagesDir))
            listImages = listImages[:numFramesVideo]
            # Ler Primeira imagem para obter as dimensões
            firstImage = cv2.imread(os.path.join(imagesDir, listImages[0]))
            height, width, _ = firstImage.shape
            fps = 30
            # Criar o objeto VideoWriter
            videoWrite = cv2.VideoWriter("Recursos/KittiVideo.mp4", cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, (width, height))
            # Iterar sobre todas as imagnes e gravar no videoWriter
            for listImage in listImages:
                image = cv2.imread(os.path.join(imagesDir, listImage))
                videoWrite.write(image)
                print(listImage)
            # Libertar Recursos
            videoWrite.release()
            self.log.info('\n Kitti video is ready \n')
        CapturedVideo = self.LoadVideo()

    def LiveCam(self):
        # Live
        ret, frameCaptured = self.webCapture.read()
        if not ret:
            print("Erro: Não foi possível capturar o frame.")
            return
        cv2.imshow('Live Frame', frameCaptured)
        cv2.waitKey(1)
        if(self.recaptureFrame == True):
            self.recaptureFrame = False
            self.framesLoaded[self.idFrame] =  frameCaptured 
            
        else:
            self.framesLoaded.append(frameCaptured) 

    def PrintFrame(self):
        cv2.imshow('Frame', self.framesLoaded[self.idFrame])
        cv2.waitKey(1)

    def PrintCustomFrame(self, frame):
        cv2.imshow('Custom Frame', frame)
        cv2.waitKey(1)

class GroundTruth:    
    def __init__(self, dataLogger):
        with open('Recursos\\data_odometry_poses\\dataset\\poses\\00.txt', 'r') as file:
            self.posesReaded = np.loadtxt(file, delimiter=' ', dtype=float)
            return 
        
    def GetPose(self, dataLogger, idFrame):
        self.poses = (np.array(self.posesReaded[idFrame]).reshape((3, 4)))
        dataLogger.info(f'\n Ground Truth idFrame({idFrame}) : \n {self.poses}')
        return self.poses

class VisualOdometry (Camera):
    def __init__(self, dataLogger):
        super().__init__(dataLogger)
        self.CalibrationFile()
        self.ResetCorners = 5
        # self.dataLogger = dataLogger
        self.featuresDetected = []
        self.featuresTracked = []
        self.prevFeaturesTracked = []
        self.essencialMatrix = []
        self.rotationMatrix = []
        self.translationMatrix = []
        if(self.liveON == True):
            self.SetupFrames()
            self.mask = np.zeros_like(self.framesLoaded[0])
            self.framesLoaded = []
        if(self.liveON == False):
            self.SetupFrames()
            self.mask = np.zeros_like(self.framesStored[0])
        self.idFramePreviuos = 0
        self.idFrameTracked = 0

        # Cria o objeto FAST com parâmetros específicos
        self.fastDetector = cv2.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True, type=2 )
         
    def FrameProcess(self):
        # frame = cv2.resize(frame, (FrameWidth, FrameHeight))
        # Convert Frame RGB on Gray scale
        frameGray = cv2.cvtColor(self.framesLoaded[self.idFrame], cv2.COLOR_BGR2GRAY)        
        
        frameFiltered = frameGray
        # frameFiltered = self.BandPassFilter(frameGray)
        # tamanho_kernel = (7, 7)
        # desvio_padrao = 12  # Valor maior para mais desfoque
        # frameFiltered = cv2.GaussianBlur(frameGray, tamanho_kernel, desvio_padrao)

        self.PrintCustomFrame(frameFiltered)
        return frameFiltered

    def DetectingFutures(self):
        # Parâmetros do detector de cantos Shi-Tomasi
        ShiTomasiParams = dict(maxCorners=500, # Number maximum to detect conrners
                               qualityLevel=0.6,
                               minDistance=5,
                               blockSize=5)
         
        if (self.framesLoaded[self.idFrame] != 0 ):
            keypointsDetected = cv2.goodFeaturesToTrack(self.FrameProcess(self.framesLoaded[self.idFrame]), mask=None, **ShiTomasiParams, useHarrisDetector=True, k=0.04)
            # keypointsDetected = keypointsDetected.astype(np.int32)
            self.featuresDetected.append(keypointsDetected)

            idfeaturesTracked = len(self.featuresDetected) - 1

        if self.idFrame == 0:
            self.featuresTracked.append(self.featuresDetected[idfeaturesTracked])
        else:
            self.featuresTracked[self.idFrame - 1] = self.featuresDetected[idfeaturesTracked]

        # self.dataLogger.info(f'\n featuresDetected ({idfeaturesTracked}) \n {self.featuresDetected[idfeaturesTracked]}')
            
    def DetectingFeaturesFASTMethod(self):
        # Checks if the current frame is loaded
        if (self.framesLoaded[self.idFrame] != None):
            # Converts the image to grayscale
            frameProcessed = self.FrameProcess()
            # Finds the points of interest using the FAST detector
            keypoints = self.fastDetector.detect(frameProcessed, None)
            # Keeps only the points with a better response
            keypoints = [ kp for kp in keypoints if kp.response > 50 ]
            # Converts the keypoints to a numpy array
            keypoints = np.array([kp.pt for kp in keypoints], dtype=np.float32)
            # Adds the detected keypoints to the feature list
            self.featuresDetected.append(keypoints)
            # Determines the index of the detected features for this frame
            idfeaturesDetected = len(self.featuresDetected) - 1
            # If this is the first frame, initializes featuresTracked with the detected features
            # For other frames, updates featuresTracked with the new detected features
            if self.idFrame == 0:
                self.featuresTracked.append(self.featuresDetected[idfeaturesDetected])
            else:
                # Adjustment: This updates the value for the current frame
                self.featuresTracked[self.idFrame - 1] = self.featuresDetected[idfeaturesDetected]
            # Log of the detected points
            self.dataLogger.info(f'\n featuresDetected ({idfeaturesDetected}) \n {self.featuresDetected[idfeaturesDetected]}')

    def BandPassFilter(self, frame):
        # Transformada de Fourier
        f_transform = np.fft.fft2(frame)  # Aplica a transformada de Fourier na imagem
        f_shift = np.fft.fftshift(f_transform)  # Shift para o centro do espectro

        # Criar filtro passa-banda
        rows, cols = frame.shape
        crow, ccol = rows // 2, cols // 2  # Encontrar o centro da imagem
        mask = np.zeros((rows, cols), np.uint8)  # Criar uma matriz de zeros do tamanho da imagem
        mask[crow-30:crow+30, ccol-30:ccol+30] = 1  # Definir uma região de passa-banda no centro da imagem

        # Aplicar filtro
        f_shift = f_shift * mask  # Multiplicar a transformada pela máscara de filtro

        # Transformada inversa de Fourier
        f_ishift = np.fft.ifftshift(f_shift)  # Desfazer o shift
        image_filtered = np.fft.ifft2(f_ishift)  # Aplicar a transformada inversa
        image_filtered = np.abs(image_filtered)  # Obter a magnitude

        # Converter de volta para o tipo uint8
        return np.uint8(image_filtered)

    def TrackingFutures(self):
        # Parameters for Lucas-Kanade optical flow
        LucasKanadeParams = dict(winSize=(21, 21),  # Slightly larger window to capture more context
                                 maxLevel=3,  # Considers more levels in the pyramid to handle larger movements
                                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.01))  # Stricter criteria for accuracy             
        
        if ( (self.idFrame == self.idFramePreviuos + 5) or (len(self.featuresTracked[self.idFrame - 1]) < 10)):
            self.idFramePreviuos = self.idFrame
            self.DetectingFeaturesFASTMethod()
            while (len(self.featuresTracked[self.idFrame - 1]) < 5):
                    self.recaptureFrame = True
                    self.LoadFrames()
                    self.idFrame = self.idFramePreviuos
                    self.DetectingFeaturesFASTMethod()
            # self.DetectingFutures()
            self.mask = np.zeros_like(self.framesLoaded[0])
            # print("\n Features Tracked = Features Detected")
        
        # Optical Flow only with previously detected features
        opticalFlow, status, _err = cv2.calcOpticalFlowPyrLK(self.framesLoaded[self.idFrame - 1], self.framesLoaded[self.idFrame],
                                                             self.featuresTracked[self.idFrame - 1], None, **LucasKanadeParams)

        # Removes points that did not match in the current featuresTracked
        self.featuresTracked[self.idFrame - 1] = self.featuresTracked[self.idFrame - 1][status[:, 0] == 1]
        # Save only new corners that have matched
        self.featuresTracked.append(opticalFlow[status[:, 0] == 1]) 

        # self.DrawFeaturesTracked()
        self.FramesOverlapping()  
        
        # self.dataLogger.info(f'\n featuresTracked ({self.idFrame}) \n {self.featuresTracked[self.idFrameTracked]}')
        return True

    def MatrixEssencial(self):   
        # Calculates the essential matrix using the tracked features
        self.essentialMatrix, mask = cv2.findEssentialMat(self.featuresTracked[self.idFrame], self.featuresTracked[self.idFrame - 1], self.intrinsicParameters, method=cv2.RANSAC, prob=0.99, threshold=0.1, maxIters=100)
        
        if ((self.essentialMatrix != None) and (len(self.essentialMatrix) == 3)):
            self.DecomposeEssentialMatrix()  # Decomposes the essential matrix to extract rotation and translation

            F = self.FundamentalMatrix(self.essencialMatrix, self.intrinsicParameters)

            # Extrai os pontos de features para passar para a função de desenho
            # Você pode precisar ajustar como os pontos são extraídos de suas estruturas de dados
            points1 = np.int32(self.featuresTracked[self.idFrame - 1])
            points2 = np.int32(self.featuresTracked[self.idFrame])

            # Desenha linhas epipolares nas imagens
            img1EpipolarLines, img2EpipolarLines = self.DrawEpipolarLines(self.framesLoaded[self.idFrame -1], self.framesLoaded[self.idFrame], points1, points2, F)

            # Exibe as imagens com linhas epipolares
            cv2.imshow("Image 1 with Epipolar Lines", img1EpipolarLines)
            cv2.imshow("Image 2 with Epipolar Lines", img2EpipolarLines)

            # self.dataLogger.info(f'\n matrizEssencial \n {self.essencialMatrix}')
            # self.dataLogger.info(f'\n rotation \n {self.rotationMatrix}')
            # self.dataLogger.info(f'\n essencialMatrixTranslation \n {self.translationMatrix}')
    
    def DecomposeEssentialMatrix(self):
        # Retrieves the rotation and translation matrices from the essential matrix
        _, self.rotationMatrix, self.translationMatrix, _ = cv2.recoverPose(self.essentialMatrix, self.featuresTracked[self.idFrame - 1], self.featuresTracked[self.idFrame], self.intrinsicParameters)

    def FundamentalMatrix(self, E, K):
        """ Calcula a matriz fundamental a partir da matriz essencial e dos parâmetros intrínsecos da câmera.

        Args:
        E (np.array): Matriz essencial.
        K (np.array): Matriz dos parâmetros intrínsecos da câmera.

        Returns:
        F (np.array): Matriz fundamental.
        """
        K_inv = np.linalg.inv(K)
        F = K_inv.T @ E @ K_inv
        return F

    def DrawEpipolarLines(self, img1, img2, points1, points2, F):
        """Desenha linhas epipolares e pontos correspondentes entre duas imagens baseadas na matriz fundamental.

        Args:
        img1, img2 (np.array): Imagens nas quais as linhas epipolares serão desenhadas.
        points1, points2 (np.array): Pontos correspondentes nas imagens.
        F (np.array): Matriz fundamental.

        Returns:
        img1, img2 (np.array): Imagens com linhas e pontos desenhados.
        """
        # Linhas na primeira imagem
        lines1 = cv2.computeCorrespondEpilines(points2.reshape(-1, 1, 2), 2, F)
        lines1 = lines1.reshape(-1, 3)
        img1_color = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR) if img1.ndim == 2 else img1.copy()

        for r, pt in zip(lines1, points1):
            color = tuple(np.random.randint(0, 255, 3).tolist())
            x0, y0 = map(int, [0, -r[2]/r[1] ])
            x1, y1 = map(int, [img1.shape[1], -(r[2]+r[0]*img1.shape[1])/r[1]])
            img1_color = cv2.line(img1_color, (x0, y0), (x1, y1), color, 1)
            img1_color = cv2.circle(img1_color, tuple(pt), 5, color, -1)

        # Linhas na segunda imagem
        lines2 = cv2.computeCorrespondEpilines(points1.reshape(-1, 1, 2), 1, F)
        lines2 = lines2.reshape(-1, 3)
        img2_color = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR) if img2.ndim == 2 else img2.copy()

        for r, pt in zip(lines2, points2):
            color = tuple(np.random.randint(0, 255, 3).tolist())
            x0, y0 = map(int, [0, -r[2]/r[1] ])
            x1, y1 = map(int, [img2.shape[1], -(r[2]+r[0]*img2.shape[1])/r[1]])
            img2_color = cv2.line(img2_color, (x0, y0), (x1, y1), color, 1)
            img2_color = cv2.circle(img2_color, tuple(pt), 5, color, -1)

        return img1_color, img2_color

    def DrawFeaturesTracked(self):
        # Desenhar as linhas que representam o fluxo optico nos quadros
        idFrameTracked = self.idFrame - 1
        # self.mask = np.zeros_like(self.framesLoaded[0])
        for i, (new, old) in enumerate(zip(self.featuresTracked[idFrameTracked], self.featuresTracked[idFrameTracked - 1])):
            a, b = new.ravel()
            c, d = old.ravel()
            a, b, c, d = int(a), int(b), int(c), int(d)  # Convert to integers
            self.mask = cv2.line(self.mask, (a, b), (c, d), (0, 255, 0), 1)
            circle = cv2.circle(self.framesLoaded[idFrameTracked], (a, b), 2, (0, 0, 255), 2)
            frameMask = cv2.add(circle, self.mask)
        cv2.imshow('Matchs On Frame', frameMask)
        cv2.waitKey(1)
        return True
    
    def FramesOverlapping(self):
        # Converte ambas as imagens para escala de cinza
        grayFramesOld = cv2.cvtColor(self.framesLoaded[self.idFrame - 1], cv2.COLOR_BGR2GRAY)
        grayFramesNew = cv2.cvtColor(self.framesLoaded[self.idFrame], cv2.COLOR_BGR2GRAY)

        # Cria o anaglifo combinando os canais
        anaglyph = cv2.merge((grayFramesNew, grayFramesOld, grayFramesOld))

        # Mostra a imagem anaglifa
        cv2.imshow('Anaglyph', anaglyph)

class Plots:
    def __init__(self, dataLogger):
        # self.dataLogger = dataLogger
        self.numPlots = 0
        # Corrigir 

        self.xValuesGroundTruth = []
        self.yValuesGroundTruth = []
        self.zValuesGroundTruth = []
        self.xValuesTrajectory = []
        self.yValuesTrajectory = []
        self.zValuesTrajectory = []
        self.errorX = []
        self.errorY = []
        self.errorZ = []
        self.errorIDs = []
        
        # Opening a file for appending the poinys
        self.fileOutput = open("Resultados/OutputTrajectory.txt", "w")
        self.fileOutput.close()

        # self.fig, self.ax = plt.subplots()
        self.fig3d = plt.figure()
        self.ax3d = self.fig3d.add_subplot(111, projection='3d')

        self.fig2d = plt.figure()
        self.ax2d = self.fig2d.add_subplot(111)

        self.fig1d = plt.figure()
        self.error = self.fig1d.add_subplot(111)

        self.ax3d.set_xlabel('X')
        self.ax3d.set_ylabel('Y')
        self.ax3d.set_zlabel('Z')
        self.ax3d.set_title('3D Camera Trajectory')
        self.ax3d.grid()

        self.ax2d.set_xlabel('X')
        self.ax2d.set_ylabel('Z')
        self.ax2d.set_title('2D Camera Trajectory')

        self.error.set_xlabel('Frame')
        self.error.set_ylabel('Error')
        self.error.set_title('Error Between Grand troth and tranjectory')

    def PrintPlots(self):

        self.ax2d.plot(self.xValuesGroundTruth, self.zValuesGroundTruth, color = 'green', label='GroundTruth')
        self.ax2d.scatter(self.xValuesGroundTruth, self.zValuesGroundTruth, color='red', marker='x')

        self.ax2d.plot(self.xValuesTrajectory, self.zValuesTrajectory, color = 'blue', label='Trajectory')
        self.ax2d.scatter(self.xValuesTrajectory, self.zValuesTrajectory, color='red', marker='o')

        self.error.plot(self.errorIDs, self.errorX, color = 'blue', label='errorX')
        self.error.plot(self.errorIDs, self.errorY, color = 'green', label='errorY')
        self.error.plot(self.errorIDs, self.errorZ, color = 'red', label='errorZ')
        self.error.plot(self.errorIDs, np.zeros(len(self.errorIDs)), color = 'black')
        self.error.scatter(self.errorIDs, self.errorX, color = 'blue', marker='.')
        self.error.scatter(self.errorIDs, self.errorY, color = 'green', marker='.')
        self.error.scatter(self.errorIDs, self.errorZ, color = 'red', marker='.')

        self.ax3d.plot(self.xValuesGroundTruth, self.yValuesGroundTruth, self.zValuesGroundTruth, color = 'green', label='GroundTruth')
        self.ax3d.scatter(self.xValuesGroundTruth, self.yValuesGroundTruth, self.zValuesGroundTruth, color='red', marker='x')

        self.ax3d.plot(self.xValuesTrajectory, self.yValuesTrajectory, self.zValuesTrajectory, color = 'blue', label='Trajectory')
        self.ax3d.scatter(self.xValuesTrajectory, self.yValuesTrajectory, self.zValuesTrajectory, color='blue', marker='o')

        self.numPlots += 1
        self.ShowPlot()

    def ShowPlot(self):
        if (self.numPlots > 0):
            if (self.numPlots == 1):     
                self.ax2d.legend()
                self.ax3d.legend()
                self.error.legend()
            self.fig1d.savefig("Resultados/PlotError.pdf")
            self.fig2d.savefig("Resultados/Trajectory2D.pdf")
            self.fig3d.savefig("Resultados/Trajectory3D.pdf")            
            plt.show()

        else:
            print("No data to plot.")

    def AddPointsToAxis(self, trajectory, type):
        # Function to plot the trajectory

        if (type == 'GroundTruth'):
            self.xValuesGroundTruth.append(trajectory[0, 3])
            self.yValuesGroundTruth.append(trajectory[1, 3])
            self.zValuesGroundTruth.append(trajectory[2, 3])
            # print(f"GroundTruth : x: {trajectory[0, 3]}, y: {trajectory[1, 3]}, z: {trajectory[2, 3]}" )
            # self.dataLogger.info(f"GroundTruth : x: {trajectory[0, 3]}, y: {trajectory[1, 3]},  z: {trajectory[2, 3]}" )


        if (type == 'Trajectory'):
            # multiply trajectory by -1 for inverte for really trajecotry
            x = trajectory[0, 3] * (-1)
            y = trajectory[1, 3] * (1)
            z = trajectory[2, 3] * (-1)
            self.xValuesTrajectory.append(x)
            self.yValuesTrajectory.append(y)
            self.zValuesTrajectory.append(z)
            
            # Opening a file for appending the poinys
            self.fileOutput = open("Resultados/OutputTrajectory.txt", "a")
            self.fileOutput.write(f"{x} {y} {z}\n")
            self.fileOutput.close()

            # Log data
            # print(f"Trajectory : x: {trajectory[0, 3]}, y: {trajectory[1, 3]}, z: {trajectory[2, 3]}" )
            # self.dataLogger.info(f"Trajectory : x: {trajectory[0, 3]},  y: {trajectory[1, 3]},  z: {trajectory[2, 3]}")
   
class Trajectory (Plots):
    def __init__(self, dataLogger, vo_instance):
        super().__init__(dataLogger)
        self.vo = vo_instance
        # self.dataLogger = dataLogger
        # self.trajectory = []
        self.allPointsTrajectory = []
        self.trajectory = np.identity(4)        
        self.allPointsTrajectory.append(self.trajectory)
        self.typeTrajectory = 'Trajectory'
        self.typeGroundTruth = 'GroundTruth'
        # Criar um image em branco
        self.scaleInCM = 10 # In cm
        self.trackLength = self.scaleInCM * 100  # Length in cm
        self.trackWidth = self.scaleInCM * 50 # Wigth in cm
        
        self.imageTrajectory = np.ones((700, 1200, 3), dtype=np.uint8) * 255  # image branco
        cv2.rectangle(self.imageTrajectory, (100, 100) , (100 + self.trackLength, 100 + self.trackWidth), (0, 255, 0), 10)
        self.pos = np.zeros((3, 1), dtype=np.float32)
        self.rot = np.eye(3)

    def PrintTrajectory(self):        
        # Convert the camera positions to pixel coordinates on the image
        # centerX, centerZ = self.imageTrajectory.shape[1], self.imageTrajectory.shape[0]
        centerX, centerZ = int(self.imageTrajectory.shape[1] / 2), int(self.imageTrajectory.shape[0] / 2)
        self.errorIDs.append(self.vo.idFrame)
        colorGroundTruth = (255, 0, 0)
        colorTrajectory = (0, 0, 255)
        colorError = (125, 200, 0)

        textPositionGroundTruth = (10, 40)
        textPositionTrajectory = (10, 60)
        textPositionError = (10, 80)
       
        textPositionAxixZ = (10, centerZ)
        textPositionAxixX = (centerX  , self.imageTrajectory.shape[0] - 10)

        cv2.putText(self.imageTrajectory, 'Z', textPositionAxixZ, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(self.imageTrajectory, 'X', textPositionAxixX, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # region GroundTruth
        # Draw the trajectory on the image as a line
        cv2.line(self.imageTrajectory, (centerX + int(self.xValuesGroundTruth[self.vo.idFrame]), centerZ - int(self.zValuesGroundTruth[self.vo.idFrame]))
                                    , (centerX + int(self.xValuesGroundTruth[self.vo.idFrame - 1]), centerZ - int(self.zValuesGroundTruth[self.vo.idFrame - 1])), colorGroundTruth, 2)
        
        # Add text with X, Y, and Z coordinates at the current position
        textGroundTruth = (f"Ground Truth X: {self.xValuesGroundTruth[self.vo.idFrame]:.2f}, Y: {self.yValuesGroundTruth[self.vo.idFrame]:.2f}, Z: {self.zValuesGroundTruth[self.vo.idFrame]:.2f}")
        # textPositionGroundTruth = (centerX + int(self.xValuesGroundTruth[self.vo.idFrame - 1]), centerZ - int(self.zValuesGroundTruth[self.vo.idFrame - 1]) - 20)
        cv2.putText(self.imageTrajectory, textGroundTruth, textPositionGroundTruth, cv2.FONT_HERSHEY_SIMPLEX, 0.5, colorGroundTruth, 1)
        # endregion

        # region trajectory
        # Draw the current position as a red circle
        # cv2.circle(self.imageTrajectory, (centerX + int(self.xValuesTrajectory[self.vo.idFrame]), centerZ - int(self.zValuesGroundTruth[self.vo.idFrame])), 2, (0, 255, 0), -1)
        # Draw the trajectory on the image as a line
        cv2.line(self.imageTrajectory, (centerX + int(self.xValuesTrajectory[self.vo.idFrame]), centerZ - int(self.zValuesTrajectory[self.vo.idFrame]))
                                    , (centerX + int(self.xValuesTrajectory[self.vo.idFrame - 1]), centerZ - int(self.zValuesTrajectory[self.vo.idFrame - 1])), colorTrajectory, 2)
        
        # Add text with X, Y, and Z coordinates at the current position
        textValuesTrajectory = (f"Trajectory X: {self.xValuesTrajectory[self.vo.idFrame]:.2f}, Y: {self.yValuesTrajectory[self.vo.idFrame]:.2f}, Z: {self.zValuesTrajectory[self.vo.idFrame]:.2f}")
        # textPositionTrajectory = (centerX + int(self.xValuesTrajectory[self.vo.idFrame - 1]), centerZ - int(self.zValuesGroundTruth[self.vo.idFrame - 1]) - 20)
        cv2.putText(self.imageTrajectory, textValuesTrajectory, textPositionTrajectory, cv2.FONT_HERSHEY_SIMPLEX, 0.5, colorTrajectory, 1)
        # endregion

        # region error        
        # Add text with X, Y, and Z coordinates at the current position
        self.errorX.append (self.xValuesTrajectory[self.vo.idFrame] - self.xValuesGroundTruth[self.vo.idFrame] )
        self.errorY.append (self.yValuesTrajectory[self.vo.idFrame] - self.yValuesGroundTruth[self.vo.idFrame] )
        self.errorZ.append (self.zValuesTrajectory[self.vo.idFrame] - self.zValuesGroundTruth[self.vo.idFrame] )

        # Add error text with X, Y, and Z coordinates at the current position
        textValuesError = (f"Error X: {self.errorX[self.vo.idFrame]:.2f}, Y: {self.errorY[self.vo.idFrame]:.2f}, Z: {self.errorZ[self.vo.idFrame]:.2f}")
        cv2.putText(self.imageTrajectory, textValuesError, textPositionError, cv2.FONT_HERSHEY_SIMPLEX, 0.5, colorError, 1)
        # endregion

        cv2.imshow('Trajectory', self.imageTrajectory)

        # region clean
        textGroundTruth = f"Ground Truth X: {self.xValuesGroundTruth[self.vo.idFrame]:.2f}, Y: {self.yValuesGroundTruth[self.vo.idFrame]:.2f}, Z: {self.zValuesGroundTruth[self.vo.idFrame]:.2f}"
        cv2.putText(self.imageTrajectory, textGroundTruth, textPositionGroundTruth, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        textValuesTrajectory = f"Trajectory X: {self.xValuesTrajectory[self.vo.idFrame]:.2f}, Y: {self.yValuesTrajectory[self.vo.idFrame]:.2f}, Z: {self.zValuesTrajectory[self.vo.idFrame]:.2f}"
        cv2.putText(self.imageTrajectory, textValuesTrajectory, textPositionTrajectory, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Add error text with X, Y, and Z coordinates at the current position
        textValuesError = (f"Error X: {self.errorX[self.vo.idFrame]:.2f}, Y: {self.errorY[self.vo.idFrame]:.2f}, Z: {self.errorZ[self.vo.idFrame]:.2f}")
        cv2.putText(self.imageTrajectory, textValuesError, textPositionError, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        # endregion

    def GetTrajectory(self):
        # self.trajectory = np.array([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 100.0]])
        # A posição é dada por
        # 𝐶𝑛 = 𝐶𝑛−1𝑇𝑛
        # A posição e orientação da câmera no instante n é dada por
        # 𝐶𝑛 = 𝑅𝑛,𝑛−1𝐶𝑛−1 + 𝑇𝑛,𝑛−1
        
        # TkHomogeneous = np.eye(4)
        # TkHomogeneous[:3, :3] = self.vo.rotationMatrix
        # TkHomogeneous[:3, 3] = self.vo.translationMatrix.ravel()
        # self.trajectory = self.trajectory @ TkHomogeneous # Matrix Multiplication

        # self.trajectory[:3, 3] = self.vo.estimatedState[:3, 0].ravel()
        
        self.pos = self.pos + self.rot @ self.vo.translationMatrix
        self.rot = self.rot @ self.vo.rotationMatrix
        self.trajectory = cv2.hconcat([self.rot, self.pos])

        # self.dataLogger.info(f'\n trajectory \n {self.trajectory}')
        # print(self.trajectory)
        
        return self.trajectory

def mainTest():
    idCamera = 0

    dataLogger = ConfigDataLogger()

    # frames = Camera(dataLogger, idCamera)
    groundTruth = GroundTruth(dataLogger)
    vo = VisualOdometry(dataLogger, idCamera)
    trajectory = Trajectory(dataLogger, vo)

    # region TESTE
    # trajectory.AddPointsToAxis(trajectory.GetTrajectory(), trajectory.typeTrajectory)
    # trajectory.AddPointsToAxis(np.array([[0.0, 0.0, 0.0, 50.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 100.0]]), trajectory.typeTrajectory)
    # trajectory.AddPointsToAxis(np.array([[0.0, 0.0, 0.0, -50.0], [0.0, 0.0, 0.0, 10.0], [0.0, 0.0, 0.0, 150.0]]), trajectory.typeTrajectory)
    # trajectory.AddPointsToAxis(np.array([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 15.0], [0.0, 0.0, 0.0, 200.0]]), trajectory.typeTrajectory)
    # endregion 

    vo.LoadFrames()

    vo.featuresTracked.append(pointsTestImage0)
    vo.featuresTracked.append(pointsTestImage1)

    vo.MatrixEssencial()

    trajectory.AddPointsToAxis(groundTruth.GetPose(dataLogger, 0), trajectory.typeGroundTruth)
    trajectory.AddPointsToAxis(groundTruth.GetPose(dataLogger, 1), trajectory.typeGroundTruth)
    trajectory.AddPointsToAxis(trajectory.trajectory, trajectory.typeTrajectory)
    trajectory.AddPointsToAxis(trajectory.GetTrajectory(vo.idFrame), trajectory.typeTrajectory)
    trajectory.PrintTrajectory()
    
    vo.idFrame += 1
    # for i in range(vo.numFramesToLoad):
    #     trajectory.AddPointsToAxis(groundTruth.GetPose(dataLogger, vo.idFrame), trajectory.typeGroundTruth)        
    #     trajectory.PrintTrajectory()
    #     vo.idFrame += 1
    #     cv2.waitKey(1)
    trajectory.PrintPlots()
    # cv2.waitKey(0)
    
    return

def getAbsoluteScale(pose1, pose2):
    x1 = pose1[3]
    y1 = pose1[7]
    z1 = pose1[11]
    x2 = pose2[3]
    y2 = pose2[7]
    z2 = pose2[11]
    return math.sqrt((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2)

def main():
    try:
        idCamera = 0

        # instancias
        dataLogger = ConfigDataLogger()
        groundTruth = GroundTruth(dataLogger)
        vo = VisualOdometry(dataLogger)
        trajectory = Trajectory(dataLogger, vo)

        # Start
        # 1st frame
        trajectory.AddPointsToAxis(groundTruth.GetPose(dataLogger, vo.idFrame), trajectory.typeGroundTruth)
        trajectory.AddPointsToAxis(trajectory.trajectory, trajectory.typeTrajectory) 
        trajectory.PrintTrajectory()

        vo.LoadFrames()        
        vo.DetectingFeaturesFASTMethod()
        
        vo.LoadFrames()        

        if(vo.liveON == True):
            while(True):    
                vo.TrackingFutures()
                vo.MatrixEssencial()
                vo.PrintFrameMatches(vo.featuresTracked[vo.idFrame - 1], vo.featuresTracked[vo.idFrame])
                
                trajectory.AddPointsToAxis(groundTruth.GetPose(dataLogger, vo.idFrame), trajectory.typeGroundTruth) # rever idFrame
                trajectory.AddPointsToAxis(trajectory.GetTrajectory(), trajectory.typeTrajectory)
                trajectory.PrintTrajectory()
                          
                vo.LoadFrames()
                # vo.DrawFeaturesTracked()
        else:        
            for i in tqdm(range(len(vo.framesStored))):
                vo.TrackingFutures()
                vo.MatrixEssencial()
                
                # scale = getAbsoluteScale(groundTruth.GetPose(dataLogger, vo.idFrame-1), groundTruth.GetPose(dataLogger, vo.idFrame))
                trajectory.AddPointsToAxis(groundTruth.GetPose(dataLogger, vo.idFrame), trajectory.typeGroundTruth) # rever idFrame
                trajectory.AddPointsToAxis(trajectory.GetTrajectory(), trajectory.typeTrajectory)
                trajectory.PrintTrajectory()
                          
                vo.LoadFrames()
                # vo.DrawFeaturesTracked()
                cv2.waitKey(1)
            
    # except IndexError:
    except MemoryError:
        print("Erro: Fim de programa.")

    
    # matlab1(vo.idFrame, vo)
    trajectory.PrintPlots()
    
    cv2.waitKey(0)

    return 1

if __name__ == '__main__':
    main()
    # mainTest()