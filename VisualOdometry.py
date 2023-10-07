# region Imports
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
        self.filePath = 'Recursos/00/image_2'

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
            intrinsicParameters = projectionMatrix[self.idCamera][:, :3]
            print(intrinsicParameters)

        self.dataLogger.info(f'\n intrinsicParameters \n {intrinsicParameters}')
        # dataLogger.info(f'\n distortioncoefficient \n {distortioncoefficient}')

    def LoadFrames(self, numFrames):
        framePath = [os.path.join(self.filePath, file) for file in sorted(os.listdir(self.filePath))][:numFrames]
        # self.framesLoaded =  [cv2.imread(path) for path in framePath][:numFrames]
        return [cv2.imread(path) for path in framePath][:numFrames]
        
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

    def Live(self):
        self.frame = cv2.VideoCapture(0) # Live

class GroundTruth:
    
    def __init__(self, dataLogger):
        self.posesReaded = pd.read_csv( 'Recursos/data_odometry_poses/dataset/poses/00.txt' , delimiter= ' ' , header= None ) 
        # print ( 'Tamanho do dataframe de pose:' , poses.shape) 
        print(f'posesReaded: {self.posesReaded.head()}\n')
        self.poses = (np.array(self.posesReaded.iloc[0]).reshape((3, 4)))
        self.type = 'GroundTruth'
        
    def GetPose(self, dataLogger, idFrame):
        self.poses = (np.array(self.posesReaded.iloc[idFrame]).reshape((3, 4)))
        dataLogger.info(f'\n Ground Truth idFrame({idFrame}) : \n {self.poses}')
        return self.poses
 
class VisualOdometry:
    def __init__(self, dataLogger, idCamera):
        self.cam = Camera(dataLogger, idCamera)
        self.ResetCorners = 5
        self.dataLogger = dataLogger
        self.intrinsicParameters = self.cam.CalibrationFile()   
    

    def DetectingFutures(self):
        return True
    
    def TrackingFutures(self):
        return True

    def MatrixEssencial(self, prevFeatures, newFeatures):

        matrizEssencial, mask = cv2.findEssentialMat(prevFeatures, newFeatures, self.intrinsicParameters, method=cv2.RANSAC, prob=0.999, threshold=1.0)


        essencialMatrixRotation, essencialMatrixTranslation = self.DecomporMatrizEssencial( matrizEssencial, prevFeatures, newFeatures, self.intrinsicParameters)

        self.dataLogger.info(f'\n matrizEssencial \n {matrizEssencial}')
        self.dataLogger.info(f'\n rotation \n {essencialMatrixRotation}')
        self.dataLogger.info(f'\n essencialMatrixTranslation \n {essencialMatrixTranslation}')

        return essencialMatrixRotation, essencialMatrixTranslation
    
    def DecomporMatrizEssencial(essentialMatrix, prevFeatures, newFeatures, cameraMatrix):
        # Recupera as matrizes de rotação e translação da matriz essencial
        _, breakDownRotation, breakDownTranslation, _ = cv2.recoverPose( essentialMatrix, prevFeatures, newFeatures, cameraMatrix)
        return breakDownRotation, breakDownTranslation

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
        self.ax2d.set_ylabel('Y')
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

        self.ax3d.plot(self.xValuesTrajectory, self.yValuesTrajectory, color = 'blue', label='Trajectory')
        self.ax3d.scatter(self.xValuesTrajectory, self.yValuesTrajectory, color='red', marker='o')

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

class Trajectory:
    def __init__(self, dataLogger):
        self.dataLogger = dataLogger
        self.initialPoint = np.array([[0.0, 0.0, 0.0, -0.05], [0.0, 0.0, 0.0, 0.-0.05], [0.0, 0.0, 0.0, 0.05]])
        self.trajectory = []        
        self.trajectory.append(self.initialPoint)
        self.type = 'Trajectory'

    def PrintTrajectory(self):
        print('Nathing')

    def GetTrajectory(self):
        self.trajectory = np.array([[0.0, 0.0, 0.0, -0.05], [0.0, 0.0, 0.0, 0.10], [0.0, 0.0, 0.0, 0.2]])
        return self.trajectory

def main():
    idCamera = 2
    numFrames = 2000

    dataLogger = ConfigDataLogger()

    camera = Camera(dataLogger, idCamera)
    groundTruth = GroundTruth(dataLogger)
    vo = VisualOdometry(dataLogger, idCamera)
    trajectory = Trajectory(dataLogger)
    plots = Plots(dataLogger)

    # 1st
    # plots.AddPointsPlot(groundTruth.poses, groundTruth.type)
    plots.AddPointsPlot(trajectory.initialPoint, trajectory.type)

    # 2nd
    # plots.AddPointsPlot(groundTruth.GetPose(dataLogger, 1), groundTruth.type)
    plots.AddPointsPlot(trajectory.GetTrajectory(), trajectory.type)

    for i in tqdm(range(numFrames)):
        plots.AddPointsPlot(groundTruth.GetPose(dataLogger, i), groundTruth.type)

    
    plots.PrintPlot()
   
    framesLoaded = Camera.LoadFrames(camera, numFrames)

    
    return 1

if __name__ == '__main__':
    main()