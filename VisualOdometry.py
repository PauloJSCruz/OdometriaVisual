# region Imports
from asyncio.windows_events import NULL
import cv2
import numpy as np
import matplotlib.pylab as plt
import pandas as pd
import os

import logging
from datetime import datetime
from tqdm import tqdm
# endregion

def ConfigDataLogger():
    dataTimeNow = datetime.now()
    # Configuração básica do log
    OutputFolderDataLogger = f'DataLogger/DataLogger_{dataTimeNow.strftime("%m.%Y")}/{dataTimeNow.strftime("%d")}'

    os.makedirs(OutputFolderDataLogger, exist_ok=True)

    logging.basicConfig(filename=f'{OutputFolderDataLogger}/dataLogger_{dataTimeNow.strftime("%H")}h{dataTimeNow.strftime("%M")}m{dataTimeNow.strftime("%S")}s.txt',
                            level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    return logging.getLogger('dataLogger')
        
class Camera:
    def __init__(self, dataLogger, IndexCamera):
        self.idCamera = IndexCamera
        self.dataLogger = dataLogger
        self.filePath = f'Recursos/00/image_{self.idCamera}'        
        self.idFrame = 0
        self.liveON = False
        self.framesLoaded = []
        self.loadNewFrame = []
        self.intrinsicParameters = []

    def CalibrationFile(self):
        projectionMatrix = []
        # Save in one list all the files name in this directory
        pathFiles = os.listdir("CalibrationCam")
        file = os.path.join("CalibrationCam", pathFiles[0])

        with open(file) as fileCalib:
            for line in fileCalib: 
                elementos = np.fromstring(line, dtype=np.float64, sep=' ')
                matriz = np.reshape(elementos, (3,4))
                projectionMatrix.append(matriz)

            for i, matriz in enumerate(projectionMatrix):
                print(f"Matriz P{i}:")
                print(projectionMatrix)

            # [: -> todas as linhas da matriz , :3 -> primeiras três colunas da matriz]
            self.intrinsicParameters = projectionMatrix[self.idCamera][:, :3]
            print(self.intrinsicParameters)

        self.dataLogger.info(f'\n intrinsicParameters \n {self.intrinsicParameters}')
        # dataLogger.info(f'\n distortioncoefficient \n {distortioncoefficient}')

    def LoadFrames(self, numFrames):
        if self.liveON is False:
            framePath = [os.path.join(self.filePath, file) for file in sorted(os.listdir(self.filePath))][:numFrames]
            self.framesLoaded = [cv2.imread(path) for path in framePath][:numFrames]

        if self.liveON is True:
            # Capture video
            # Load Frame
            # Append new frame
            return print( '\n Live ON \n')
        
        
    def LoadVideo(self):
        videoPath = "Recursos/KittiVideo.mp4"

        # dataLogger.info(f'Video: {videoPath}')

        #CapturedVideo = cv2.VideoCapture(0) # Live
        CapturedVideo = cv2.VideoCapture(videoPath)    
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
        self.frame = cv2.VideoCapture(0) 

    def PrintFrame(self):
            return 1
        
    def PrintFrameMatches(self, prevFeaturesTracked, featuresTracked):
        # Definir a largura e altura desejadas do frame
        # FrameWidth = 1680
        # FrameHeight = 700
        combined_frame = cv2.hconcat((self.framesLoaded[self.idFrame - 1], self.framesLoaded[self.idFrame]))
        # Desenhar uma linha entre os pontos nos quadros atual e anterior
        if featuresTracked is not None:
            for i in range(len(featuresTracked)):
                prev_pt = tuple(map(int, prevFeaturesTracked[i].ravel()))
                current_pt = tuple(map(int, prevFeaturesTracked[i].ravel()))

                # Gerar uma cor RGB aleatoria
                color = tuple(np.random.randint(0, 255, 3).tolist())
                # Desenhar a linha com a cor aleatoria no quadro combinado
                cv2.line(combined_frame, prev_pt, (current_pt[0] + self.framesLoaded[self.idFrame - 1].shape[1], current_pt[1]), color, 1)
                # Desenhar circulos sem preenchimento nas extremidades da linha
                cv2.circle(combined_frame, prev_pt, 1, color, -1)
                cv2.circle(combined_frame, (current_pt[0] + self.framesLoaded[self.idFrame - 1].shape[1], current_pt[1]), 1, color, -1)

                # Rodar e Redimensionar o frame para a largura e altura desejadas
                # combined_frame = cv2.resize(combined_frame, (FrameWidth, FrameHeight))
            cv2.imshow('Combined Frames', combined_frame)

class GroundTruth:    
    def __init__(self, dataLogger):
        self.posesReaded = pd.read_csv( f'Recursos/data_odometry_poses/dataset/poses/00.txt' , delimiter= ' ' , header= None ) 
        # print ( 'Tamanho do dataframe de pose:' , poses.shape) 
        print(f'posesReaded: {self.posesReaded.head()}\n')
        self.poses = (np.array(self.posesReaded.iloc[0]).reshape((3, 4)))
        
    def GetPose(self, dataLogger, idFrame):
        self.poses = (np.array(self.posesReaded.iloc[idFrame]).reshape((3, 4)))
        dataLogger.info(f'\n Ground Truth idFrame({idFrame}) : \n {self.poses}')
        return self.poses

class VisualOdometry (Camera):
    def __init__(self, dataLogger, indexCamera, numFramesToLoad):
        super().__init__(dataLogger, indexCamera)
        self.LoadFrames(numFramesToLoad)
        self.CalibrationFile()
        self.ResetCorners = 5
        self.idFrame = 0
        self.dataLogger = dataLogger
        self.featuresDetected = []
        self.featuresTracked = []
        self.prevFeaturesTracked = []
        self.essencialMatrix = []
        self.rotationMatrix = []
        self.translationMatrix = []
        self.mask = np.zeros_like(self.framesLoaded[0])
        # Analized needed
        self.firstFrame = []
        self.featuresTrackedReset = False
        self.idFrameTracked = 0
        
    
    def FrameProcess(self, frame):
        # Convert Frame RGB on Gray scale
        frameProcessedGray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        # Applying the canny algorithm
        frameProcessedCanny = cv2.Canny(frameProcessedGray, 250, 255, None, 3)
        # cv2.imshow('Combined Frames', frameProcessedCanny)
        return frameProcessedCanny

    def DetectingFutures(self):
        # region Parâmetros do detector de cantos Shi-Tomasi
        ShiTomasiParams = dict(maxCorners=1000,
                               qualityLevel=0.2,
                               minDistance=50,
                               blockSize=7)
        # ShiTomasiParams = dict(maxCorners=2000,
        #                        qualityLevel=0.2,
        #                        minDistance=25,
        #                        blockSize=7)
        # endregion

        # # Load number of iterations
        # self.idFrame = len(self.featuresTracked)
         
        if self.framesLoaded[self.idFrame] is not NULL:
            keypointsDetected = cv2.goodFeaturesToTrack(self.FrameProcess(self.framesLoaded[self.idFrame]), mask=None, **ShiTomasiParams, 
                                                                useHarrisDetector=True, k=0.04)
            # keypointsDetected = keypointsDetected.astype(np.int32)
            self.featuresDetected.append(keypointsDetected)

            idfeaturesDetected = len(self.featuresDetected) - 1

        if self.idFrame == 0:
            self.featuresTracked.append(self.featuresDetected[idfeaturesDetected])
        else:
            self.featuresTracked[self.idFrame - 1] = self.featuresDetected[idfeaturesDetected]

        self.dataLogger.info(f'\n featuresDetected ({idfeaturesDetected}) \n {self.featuresDetected[idfeaturesDetected]}')
            
    def TrackingFutures(self):
        # region Parameters for lucas kanade optical flow
        LucasKanadeParams = dict(winSize=(25, 25),
                                 maxLevel=2,
                                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.01))
        # endregion        

        self.idFrameTracked = self.idFrame - 1        

        # region Optical Flow with previous and currently features detected (Original form)
        # Calcular o fluxo ótico usando o método Lucas-Kanade
        # opticalFlow, status, _err = cv2.calcOpticalFlowPyrLK( self.framesLoaded[self.idFrame - 1], self.framesLoaded[self.idFrame],
        #                                                        self.featuresDetected[self.idFrame - 1], self.featuresDetected[self.idFrame], **LucasKanadeParams)
        # endregion
        
        if( self.idFrame > 1 ):
            if ( len(self.featuresTracked[self.idFrameTracked]) < 10 ):
                self.DetectingFutures()
                self.featuresTrackedReset = True
                self.mask = np.zeros_like(self.framesLoaded[0])
                print("\n Features Tracked = Features Detected")
            else:
                self.featuresTrackedReset = False

        # Optical Flow only with previous features detected
        opticalFlow, status, _err = cv2.calcOpticalFlowPyrLK( self.framesLoaded[self.idFrame - 1], self.framesLoaded[self.idFrame],
                                                               self.featuresTracked[self.idFrame - 1] , None, **LucasKanadeParams)

        # Remove os pontos que não tiveram correspondência no featuresTracked atual
        # teste = self.featuresTracked[self.idFrame - 1].copy()
        self.featuresTracked[self.idFrame - 1] = self.featuresTracked[self.idFrame - 1][status[:, 0] == 1]

        # Save only newCorners that have done match
        self.featuresTracked.append(opticalFlow[status[:, 0] == 1])       
        
        self.dataLogger.info(f'\n featuresTracked ({self.idFrame}) \n {self.featuresTracked[self.idFrameTracked]}')
        return True

    def MatrixEssencial(self):
        
        self.essencialMatrix, mask = cv2.findEssentialMat(self.featuresTracked[self.idFrame - 1], self.featuresTracked[self.idFrame], self.intrinsicParameters, method = cv2.RANSAC, prob = 0.999, threshold = 1.0)

        self.DecomporMatrizEssencial()

        self.dataLogger.info(f'\n matrizEssencial \n {self.essencialMatrix}')
        self.dataLogger.info(f'\n rotation \n {self.rotationMatrix}')
        self.dataLogger.info(f'\n essencialMatrixTranslation \n {self.translationMatrix}')
    
    def DecomporMatrizEssencial(self):
        # Recupera as matrizes de rotação e translação da matriz essencial
        _, self.rotationMatrix, self.translationMatrix, _ = cv2.recoverPose( self.essencialMatrix, self.featuresTracked[self.idFrame - 1], self.featuresTracked[self.idFrame], self.intrinsicParameters)

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
        return True

class Plots:
    def __init__(self, dataLogger):
        self.dataLogger = dataLogger
        self.xValuesGroundTruth = []
        self.yValuesGroundTruth = []
        self.zValuesGroundTruth = []

        self.xValuesTrajectory = []
        self.yValuesTrajectory = []
        self.zValuesTrajectory = []

        # self.fig, self.ax = plt.subplots()
        self.fig3d = plt.figure()
        self.ax3d = self.fig3d.add_subplot(111, projection='3d')

        self.fig2d = plt.figure()
        self.ax2d = self.fig2d.add_subplot(111)

        self.ax3d.set_xlabel('X')
        self.ax3d.set_ylabel('Y')
        self.ax3d.set_zlabel('Z')
        self.ax3d.set_title('3D Camera Trajectory')
        self.ax3d.grid()

        self.ax2d.set_xlabel('X')
        self.ax2d.set_ylabel('Z')
        self.ax2d.set_title('2D Camera Trajectory')
        self.numPlots = 0

        self.numPlots = 0

    def PrintPlot(self):

        self.ax2d.plot(self.xValuesGroundTruth, self.zValuesGroundTruth, color = 'green', label='GroundTruth')
        self.ax2d.scatter(self.xValuesGroundTruth, self.zValuesGroundTruth, color='red', marker='x')

        self.ax2d.plot(self.xValuesTrajectory, self.zValuesTrajectory, color = 'blue', label='Trajectory')
        self.ax2d.scatter(self.xValuesTrajectory, self.zValuesTrajectory, color='red', marker='o')

        self.ax3d.plot(self.xValuesGroundTruth, self.yValuesGroundTruth, self.zValuesGroundTruth, color = 'green', label='GroundTruth')
        self.ax3d.scatter(self.xValuesGroundTruth, self.yValuesGroundTruth, self.zValuesGroundTruth, color='red', marker='x')

        self.ax3d.plot(self.xValuesTrajectory, self.yValuesTrajectory, self.zValuesTrajectory, color = 'blue', label='Trajectory')
        self.ax3d.scatter(self.xValuesTrajectory, self.yValuesTrajectory, self.zValuesTrajectory, color='red', marker='o')

        self.numPlots += 1
        self.ShowPlot()

    # def PrintPlot(self):
    #     for i in range(len(self.xValues)):
    #         plt.plot(self.xValues[i], self.yValues[i], 'b-')
    #         plt.scatter(self.xValues[i], self.yValues[i], color='red', label='Camera Positions')
    #         plt.show()

    def ShowPlot(self):
        if self.numPlots > 0:
            if self.numPlots is 1:     
                self.ax2d.legend()
            plt.show()
        else:
            print("No data to plot.")

    def AddPointsPlot(self, trajectory, type):
        # Function to plot the trajectory

        if type is 'GroundTruth':
            self.xValuesGroundTruth.append(trajectory[0, 3])
            self.yValuesGroundTruth.append(trajectory[1, 3])
            self.zValuesGroundTruth.append(trajectory[2, 3])

        if type is 'Trajectory':
            self.xValuesTrajectory.append(trajectory[0, 3])
            self.yValuesTrajectory.append(trajectory[1, 3])
            self.zValuesTrajectory.append(trajectory[2, 3])

class Trajectory ():
    def __init__(self, dataLogger, vo_instance):
        self.vo_instance = vo_instance
        self.dataLogger = dataLogger
        # self.initialPoint = np.array([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
        self.trajectory = []
        self.trajectory = np.identity(4)
        self.typeTrajectory = 'Trajectory'
        self.typeGroundTruth = 'GroundTruth'

    def PrintTrajectory(self):
        print('Nathing')

    def GetTrajectory(self, idFrame):
        # self.trajectory = np.array([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 100.0]])
        # A posição é dada por
        # 𝐶𝑛 = 𝐶𝑛−1𝑇𝑛
        # A posição e orientação da câmera no instante n é dada por
        # 𝐶𝑛 = 𝑅𝑛,𝑛−1𝐶𝑛−1 + 𝑇𝑛,𝑛−1

        Tk_homogeneous = np.eye(4)
        Tk_homogeneous[:3, :3] = self.vo_instance.rotationMatrix
        Tk_homogeneous[:3, 3] = self.vo_instance.translationMatrix.ravel()
        self.trajectory = self.trajectory @ Tk_homogeneous

        # for i in range(1, len(self.featuresTracked)):
        #     # Extract the rotation and translation matrices from visual odometry
        #     rotation_matrix = self.rotationMatrix[i - 1]
        #     translation_matrix = self.translationMatrix[i - 1]

        #     # Create a 4x4 transformation matrix
        #     transformation_matrix = np.identity(4)
        #     transformation_matrix[:3, :3] = rotation_matrix
        #     transformation_matrix[:3, 3] = translation_matrix

        #     # Update the transformation matrix based on the previous frame
        #     trajectory.append(np.dot(trajectory[-1], transformation_matrix))

        
        return self.trajectory

def main():
    try:
        idCamera = 0
        numFramesToLoad = 2000

        dataLogger = ConfigDataLogger()

        # frames = Camera(dataLogger, idCamera)
        groundTruth = GroundTruth(dataLogger)
        vo = VisualOdometry(dataLogger, idCamera, numFramesToLoad)
        trajectory = Trajectory(dataLogger, vo)
        plots = Plots(dataLogger)

        # region TESTE
        # plots.AddPointsPlot(trajectory.GetTrajectory(), trajectory.typeTrajectory)
        # plots.AddPointsPlot(np.array([[0.0, 0.0, 0.0, 50.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 100.0]]), trajectory.typeTrajectory)
        # plots.AddPointsPlot(np.array([[0.0, 0.0, 0.0, -50.0], [0.0, 0.0, 0.0, 10.0], [0.0, 0.0, 0.0, 150.0]]), trajectory.typeTrajectory)
        # plots.AddPointsPlot(np.array([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 15.0], [0.0, 0.0, 0.0, 200.0]]), trajectory.typeTrajectory)
        # endregion      

        # 1st
        plots.AddPointsPlot(groundTruth.poses, trajectory.typeGroundTruth)
        # plots.AddPointsPlot(trajectory.initialPoint, trajectory.typeTrajectory)
        vo.DetectingFutures()
        vo.idFrame += 1
        
        # 2nd
        plots.AddPointsPlot(groundTruth.GetPose(dataLogger, 1), trajectory.typeGroundTruth)
        # vo.DetectingFutures()
        vo.TrackingFutures()
        vo.PrintFrameMatches(vo.featuresTracked[vo.idFrame - 1], vo.featuresTracked[vo.idFrame])
        vo.MatrixEssencial()
        vo.rotationMatrix
        vo.translationMatrix
        plots.AddPointsPlot(trajectory.GetTrajectory(vo.idFrame - 1), trajectory.typeTrajectory)
        # Load number of iterations
        vo.idFrame = len(vo.featuresTracked)    
        
        vo.DrawFeaturesTracked()
        
        numFeaturesDetected = len(vo.featuresDetected)
        for i in tqdm(range(len(vo.framesLoaded)- numFeaturesDetected)): 
            vo.TrackingFutures()
            vo.PrintFrameMatches(vo.featuresTracked[vo.idFrame - 1], vo.featuresTracked[vo.idFrame])
            
            plots.AddPointsPlot(groundTruth.GetPose(dataLogger, i), trajectory.typeGroundTruth) 
            plots.AddPointsPlot(trajectory.GetTrajectory(vo.idFrame - 1), trajectory.typeTrajectory)
            
            # Load number of iterations            
            vo.idFrame += 1
            vo.DrawFeaturesTracked()
            cv2.waitKey(1)
    except IndexError:
        print("Erro: Fim de programa.")

    plots.PrintPlot()

    cv2.waitKey(0)

    return 1

if __name__ == '__main__':
    main()