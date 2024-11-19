# region Imports
import cv2
import numpy as np
import matplotlib.pylab as plt
import os
import logging
import sys
from datetime import datetime
from tqdm import tqdm
import math
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
    def __init__(self, dataLogger):
        # Editable parameters
        self.liveON = False
        self.numFramesToLoad = 1000      
        self.idCamera = 0
        # end
        self.dataLogger = dataLogger
        self.idFrame = 0
        self.idStored = 0
        self.framesStored = []
        self.framesLoaded = []
        self.intrinsicParameters = []
        self.projectionMatrix = []
        # self.picam2 = Picamera2()
        self.webCapture = cv2.VideoCapture(0)
        self.recaptureFrame = False
        self.prevTime = datetime.now()
        self.totalFPS = 0.0
        self.instaFPS = 0.0

    def CalibrationFile(self):
        # Define o caminho para o arquivo de calibração
        file = "CalibrationCam/calib.txt"
        try:
            with open(file) as fileCalib:
                for line in fileCalib:
                    if (line.startswith(f"P{self.idCamera}:")):
                        # Extrai os números da linha, ignorando o identificador "P0:"
                        elementos = np.fromstring(line[3:], sep=' ', dtype=np.float64)
                        # Reorganiza os elementos para formar a matriz de projeção 3x4
                        matrizProjecao = np.reshape(elementos, (3, 4))
                        # Extrai os parâmetros intrínsecos (as três primeiras colunas da matriz de projeção)
                        self.intrinsicParameters = matrizProjecao[:, :3]
                        print(f"Parâmetros intrínsecos camara {self.idCamera}:")
                        print(self.intrinsicParameters)
                        break  # Encerra o loop após processar a linha desejada

        except FileNotFoundError:
            print(f"Arquivo não encontrado: {file}")
            return
        
        # self.dataLogger.info(f'\nParâmetros Intrínsecos:\n{self.intrinsicParameters}')

    def LoadFrames(self):
        self.filePath = f'Recursos/00/image_{self.idCamera}'
        if (self.liveON == False):
            if ( len(self.framesStored) == 0):
                framePath = [os.path.join(self.filePath, file) for file in sorted(os.listdir(self.filePath))][:self.numFramesToLoad]
                self.framesStored = [cv2.imread(path) for path in framePath][:self.numFramesToLoad]
                self.frameHeight, self.frameWidth = self.framesStored[0].shape[:2]
                return print( '\n Frames Loaded \n')
            else:
                self.PrintFrame(self.framesStored[self.idStored])
                if self.idCamera == 2 or self.idCamera == 3:
                    self.framesStored[self.idStored] = cv2.cvtColor(self.framesStored[self.idStored], cv2.COLOR_BGR2GRAY)                
                self.framesLoaded.append(self.framesStored[self.idStored])
                self.idFrame = len(self.framesLoaded) - 1
                self.idStored += 1                
                return 

        if (self.liveON == True):
            # Capture new frame
            self.LiveCam()
            self.idFrame = len(self.framesLoaded) - 1
            return print( '\n Live ON \n')

    def PrintFrame(self, frame):
        currentTime = datetime.now()
        time = (currentTime - self.prevTime).total_seconds()
        if time > 0:
            self.instaFPS = round(1 / time, 2) 
            self.totalFPS += self.instaFPS  
            self.prevTime = currentTime
    
    def GetPose(self, dataLogger, idFrame):
        self.poses = np.array(self.posesReaded[idFrame])
        self.poses = self.poses.reshape((3, 4))
        dataLogger.info(f'\n Ground Truth idFrame({idFrame}) : \n {self.poses}')
        return self.poses

class GroundTruth:    
    def __init__(self, dataLogger):
        with open('Recursos/data_odometry_poses/dataset/poses/00.txt', 'r') as file:
            self.posesReaded = np.loadtxt(file, delimiter=' ', dtype=float)
            return
    
    def GetPose(self, dataLogger, idFrame):
        self.poses = np.array(self.posesReaded[idFrame])
        self.poses = self.poses.reshape((3, 4))
        dataLogger.info(f'\n Ground Truth idFrame({idFrame}) : \n {self.poses}')
        return self.poses

class VisualOdometry (Camera):
    def __init__(self, dataLogger):
        super().__init__(dataLogger)
        self.ResetCorners = 5
        # self.dataLogger = dataLogger
        self.featuresDetected = []
        self.featuresTracked = []
        self.prevFeaturesTracked = []
        self.essencialMatrix = []
        self.rotationMatrix = []
        self.translationMatrix = []
        self.idFramePreviuos = 0
        self.mask = []
        self.points3D = None


        # Cria o objeto FAST com parâmetros específicos
        self.fastDetector = cv2.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True, type=2 )

    def FrameProcess(self):      
        frameFiltered = self.framesLoaded[self.idFrame].copy()
        # frameFiltered = self.BandPassFilter(frameGray)
        # tamanho_kernel = (9, 9)
        # desvio_padrao = 3  # Valor maior para mais desfoque
        # frameFiltered = cv2.GaussianBlur(frameFiltered, tamanho_kernel, desvio_padrao)

        # self.PrintCustomFrame("Frame filtred", frameFiltered)
        return frameFiltered
    
    def DetectingFeaturesFASTMethod(self):
        # Checks if the current frame is loaded
        if (self.framesLoaded[self.idFrame] is not None):
            
            # Converts the image to grayscale
            frameProcessed = self.FrameProcess()
            
            # Finds the points of interest using the FAST detector
            keypoints = self.fastDetector.detect(frameProcessed, None)

            # Keeps only the points with a better response
            # dif = len(keypoints)
            response = 20
            keypointsgood = [ kp for kp in keypoints if kp.response > response ]
            while(len(keypointsgood) < 25):
                keypointsgood = [ kp for kp in keypoints if kp.response > response ]
                response -= 5
            keypoints = keypointsgood

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
            # self.dataLogger.info(f'\n featuresDetected ({idfeaturesDetected}) \n {self.featuresDetected[idfeaturesDetected]}')

    def TrackingFutures(self):
        # Parameters for Lucas-Kanade optical flow
        LucasKanadeParams = dict(winSize=(15, 15),  # Slightly larger window to capture more context
                                 maxLevel=4,  # Considers more levels in the pyramid to handle larger movements
                                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.01))  # Stricter criteria for accuracy             
        
        if ( (self.idFrame == self.idFramePreviuos + 5) or (len(self.featuresTracked[self.idFrame - 1]) < 20)):
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
        newFeatures = opticalFlow[status[:, 0] == 1]
        # Save only new corners that have matched
        self.featuresTracked.append(opticalFlow[status[:, 0] == 1]) 

        # self.DrawFeaturesMatched()
        # self.FramesOverlapping(self.DrawFeaturesTracked(newFeatures, self.featuresTracked[self.idFrame - 1]))  
        
        self.dataLogger.info(f'\n featuresTracked ({self.idFrame}) \n {self.featuresTracked[self.idFrame]}')
        return True

    def CalculateEssentialMatrix(self):
        # Calculates the essential matrix using the tracked features
        self.essentialMatrix, mask = cv2.findEssentialMat(self.featuresTracked[self.idFrame], 
                                                          self.featuresTracked[self.idFrame - 1],
                                                          self.intrinsicParameters, 
                                                          method=cv2.RANSAC, prob=0.99, threshold=0.1, maxIters=100)

        # self.essentialMatrix, mask = cv2.findEssentialMat(self.featuresTracked[self.idFrame], 
        #                                                   self.featuresTracked[self.idFrame - 1],
        #                                                   self.intrinsicParameters, 
        #                                                   method=cv2.RANSAC, prob=0.99, threshold=1, maxIters=100)
        
        if ((self.essentialMatrix is not None) and (len(self.essentialMatrix) == 3)):
            self.DecomposeEssentialMatrix()  # Decomposes the essential matrix to extract rotation and translation

    
    def DecomposeEssentialMatrix(self):
        # Retrieves the rotation and translation matrices from the essential matrix
        _, self.rotationMatrix, self.translationMatrix, _ = cv2.recoverPose(self.essentialMatrix, 
                                                                            self.featuresTracked[self.idFrame], 
                                                                            self.featuresTracked[self.idFrame - 1], 
                                                                            self.intrinsicParameters)

class Plots:
    def __init__(self, dataLogger):
        self.dataLogger = dataLogger
        self.numPlots = 0
        self.xValuesGroundTruth = []
        self.yValuesGroundTruth = []
        self.zValuesGroundTruth = []
        self.xValuesTrajectory = []
        self.yValuesTrajectory = []
        self.zValuesTrajectory = []
        self.errorX = []
        self.errorY = []
        self.errorZ = []
        self.MeanAbsoluteError = []
        self.RootMeanSquaredError = []
        self.MeanAbsoluteError.append(0.0)
        self.RootMeanSquaredError.append(0.0)
        self.errorIDs = []
        
        self.trajectoryPath = "Resultados/OutputTrajectory.txt"
        open(self.trajectoryPath, 'w')

        # self.fig, self.ax = plt.subplots()
        self.fig3d = plt.figure()
        self.ax3d = self.fig3d.add_subplot(111, projection='3d')

        self.fig2d = plt.figure()
        self.ax2d = self.fig2d.add_subplot(111)

        self.figErroAxes = plt.figure()
        self.errorAxes = self.figErroAxes.add_subplot(111)

        self.figError = plt.figure()
        self.errorMean  = self.figError.add_subplot(111)

        self.ax3d.set_xlabel('X')
        self.ax3d.set_ylabel('Y')
        self.ax3d.set_zlabel('Z')
        self.ax3d.set_title('3D Camera trajectory')
        self.ax3d.grid()

        self.ax2d.set_xlabel('X')
        self.ax2d.set_ylabel('Z')
        self.ax2d.set_title('2D Camera trajectory')

        self.errorAxes.set_xlabel('Frame Number')
        self.errorAxes.set_ylabel('Metros')
        self.errorAxes.set_title('Error Between Grand truth axes and trajectory axes')

        self.errorMean .set_xlabel('Frame Number')
        self.errorMean .set_ylabel('Metros')
        self.errorMean .set_title('Error Between Grand truth and trajectory')

    def PrintPlots(self):

        self.ax2d.plot(self.xValuesGroundTruth, self.zValuesGroundTruth, color = 'blue', label='GroundTruth')
        # self.ax2d.scatter(self.xValuesGroundTruth, self.zValuesGroundTruth, color='red', marker='x')

        self.ax2d.plot(self.xValuesTrajectory, self.zValuesTrajectory, color = 'red', label='Trajectory')
        # self.ax2d.scatter(self.xValuesTrajectory, self.zValuesTrajectory, color='red', marker='o')

        self.errorAxes.plot(self.errorIDs, self.errorX, color = 'blue', label='errorX')
        self.errorAxes.plot(self.errorIDs, self.errorY, color = 'green', label='errorY')
        self.errorAxes.plot(self.errorIDs, self.errorZ, color = 'red', label='errorZ')
        self.errorAxes.plot(self.errorIDs, np.zeros(len(self.errorIDs)), color = 'black')
        # self.errorAxes.scatter(self.errorIDs, self.errorX, color = 'blue', marker='.')
        # self.errorAxes.scatter(self.errorIDs, self.errorY, color = 'green', marker='.')
        # self.errorAxes.scatter(self.errorIDs, self.errorZ, color = 'red', marker='.')

        self.errorMean.plot(self.errorIDs, self.MeanAbsoluteError, color = 'blue', label='MeanAbsoluteError')
        self.errorMean.plot(self.errorIDs, self.RootMeanSquaredError, color = 'green', label='RootMeanSquaredError')
        self.errorMean.plot(self.errorIDs, np.zeros(len(self.errorIDs)), color = 'black')

        # self.errorMean.scatter(self.errorIDs, self.MeanAbsoluteError, color = 'red', marker='.')
        # self.errorMean.scatter(self.errorIDs, self.RootMeanSquaredError, color = 'red', marker='.')

        self.ax3d.plot(self.xValuesGroundTruth, self.yValuesGroundTruth, self.zValuesGroundTruth, color = 'blue', label='GroundTruth')
        # self.ax3d.scatter(self.xValuesGroundTruth, self.yValuesGroundTruth, self.zValuesGroundTruth, color='red', marker='x')

        self.ax3d.plot(self.xValuesTrajectory, self.yValuesTrajectory, self.zValuesTrajectory, color = 'red', label='Trajectory')
        # self.ax3d.scatter(self.xValuesTrajectory, self.yValuesTrajectory, self.zValuesTrajectory, color='blue', marker='o')

        self.numPlots += 1
        self.ShowPlot()

    def ShowPlot(self):
        dataTimeNow = datetime.now()
        if (self.numPlots > 0):
            if (self.numPlots == 1):     
                self.ax2d.legend()
                self.ax3d.legend()
                self.errorMean .legend()
                self.errorAxes.legend()
            self.fig2d.savefig(f"Resultados/Trajectory2D{dataTimeNow.strftime('%H')}h{dataTimeNow.strftime('%M')}m{dataTimeNow.strftime('%S')}s.pdf")
            self.fig3d.savefig(f"Resultados/Trajectory3D{dataTimeNow.strftime('%H')}h{dataTimeNow.strftime('%M')}m{dataTimeNow.strftime('%S')}s.pdf")
            self.figErroAxes.savefig(f"Resultados/PlotErrorAxes{dataTimeNow.strftime('%H')}h{dataTimeNow.strftime('%M')}m{dataTimeNow.strftime('%S')}s.pdf")
            self.figError.savefig(f"Resultados/PlotError{dataTimeNow.strftime('%H')}h{dataTimeNow.strftime('%M')}m{dataTimeNow.strftime('%S')}s.pdf")
                        
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
            x = trajectory[0, 0] * (1)
            y = trajectory[1, 0] * (1)
            z = trajectory[2, 0] * (1)
            self.xValuesTrajectory.append(x)
            self.yValuesTrajectory.append(y)
            self.zValuesTrajectory.append(z)
            
            # Opening a file for appending the poinys
            self.fileOutput = open(self.trajectoryPath, "a")
            self.fileOutput.write(f"{x} {y} {z}\n")
            self.fileOutput.close()
            
            # self.dataLogger.info(f"Trajectory : x: {trajectory[0, 3]},  y: {trajectory[1, 3]},  z: {trajectory[2, 3]}")
   
class Trajectory (Plots):
    def __init__(self, dataLogger, voInstance):
        super().__init__(dataLogger)
        self.vo = voInstance
        self.dataLogger = dataLogger
        self.typeTrajectory = 'Trajectory'
        self.typeGroundTruth = 'GroundTruth'
        # Criar um image em branco
        self.imageTrajectory = np.ones((1000, 1920, 3), dtype=np.uint8) * 255  # image branco        
        self.trajectoryPosition = np.zeros((3, 1), dtype=np.float32)
        self.trajectoryRotation = np.eye(3)
        self.trajectory = []
        self.trajectory.append(self.trajectoryPosition)

    def PrintTrajectory(self):        
        self.errorX.append (self.xValuesTrajectory[self.vo.idFrame] - self.xValuesGroundTruth[self.vo.idFrame] )
        self.errorY.append (self.yValuesTrajectory[self.vo.idFrame] - self.yValuesGroundTruth[self.vo.idFrame] )
        self.errorZ.append (self.zValuesTrajectory[self.vo.idFrame] - self.zValuesGroundTruth[self.vo.idFrame] )

    def GetTrajectory(self):
        # The position is given by
        # C_n = C_(n-1) * T_n
        # The camera's position and orientation at time n is given by
        # C_n = R_(n,n-1) * C_(n-1) + T_(n,n-1)

        # Calcular a escala usando os dados de ground truth
        if self.vo.idFrame > 0:
            prevGroundTruthPose = np.array([self.xValuesGroundTruth[self.vo.idFrame - 1], self.yValuesGroundTruth[self.vo.idFrame - 1], self.zValuesGroundTruth[self.vo.idFrame - 1]])
            currentGroundTruthPose = np.array([self.xValuesGroundTruth[self.vo.idFrame], self.yValuesGroundTruth[self.vo.idFrame], self.zValuesGroundTruth[self.vo.idFrame]])
            
            trueDistance = np.linalg.norm(currentGroundTruthPose - prevGroundTruthPose)
            estimatedDistance = np.linalg.norm(self.vo.translationMatrix)
            
            self.scaleFactor = trueDistance / estimatedDistance if estimatedDistance != 0 else 1.0
        
        # Ajuste a translação usando a escala
        self.vo.translationMatrix *= self.scaleFactor

        # Calcule a posição e rotação acumuladas
        self.trajectoryPosition = self.trajectoryPosition + self.trajectoryRotation @ self.vo.translationMatrix  # Update the position using translation matrix
        self.trajectoryRotation = self.trajectoryRotation @ self.vo.rotationMatrix  # Update the rotation using rotation matrix
        # trajectoryMatrix = np.hstack([self.trajectoryRotation, self.trajectoryPosition])  # Concatenates rotation and position to form the trajectory matrix
        
        self.trajectory.append(self.trajectoryPosition.copy())

        # Calcule erros
        groundTruthMatrix = currentGroundTruthPose.reshape((3, 1))
        mae = np.mean(np.abs(groundTruthMatrix - self.trajectoryPosition))
        mse = np.mean((groundTruthMatrix - self.trajectoryPosition) ** 2)
        rmse = math.sqrt(mse)

        self.MeanAbsoluteError.append(mae)
        self.RootMeanSquaredError.append(rmse)

        # Log da trajetória
        # self.dataLogger.info(f'\n trajectory \n {self.trajectory}')        
        return self.trajectory

def main():
    idCamera = 2
    numFramesToLoad = 4510
    liveON = False
    try:
         # Analisa os argumentos manualmente
        for i in range(1, len(sys.argv), 2):
            if sys.argv[i] == '-idCam':
                idCamera = int(sys.argv[i + 1])
            elif sys.argv[i] == '-numFrames':
                numFramesToLoad = int(sys.argv[i + 1])
            elif sys.argv[i] == '-live':
                liveON = bool(sys.argv[i + 1])
            elif sys.argv[i] == '-help':
                print("Flags:")
                print("-idCam: Id form camera")
                print("-numFrames: Number of frames that are load")
                print("-live: True or False if the frames are capture from robot camera")
                print("Struct: python meuprograma.py -id <idCamera> -numFrames <numFramesToLoad> -live <true>\n")
                sys.exit(1)
            else:
                print(f"Argumento desconhecido: {sys.argv[i]}")
                print("python meuprograma.py -id <idCamera> -numFrames <numFramesToLoad> -live <true>")
                sys.exit(1)
 
        # instancias
        dataLogger = ConfigDataLogger()
        groundTruth = GroundTruth(dataLogger)
        vo = VisualOdometry(dataLogger)
        trajectory = Trajectory(dataLogger, vo)

        if(idCamera != vo.idCamera):
            vo.idCamera = idCamera
        if(numFramesToLoad != vo.numFramesToLoad):
            vo.numFramesToLoad = numFramesToLoad
        if(liveON != vo.liveON):
            vo.liveON = liveON     

        if(vo.liveON == True):
            vo.LoadFrames()
            vo.mask = np.zeros_like(vo.framesLoaded[0])
            vo.framesLoaded = []
        if(vo.liveON == False):
            vo.LoadFrames()
            vo.mask = np.zeros_like(vo.framesStored[0])
        
        vo.CalibrationFile()

        # Start
        trajectory.AddPointsToAxis(groundTruth.GetPose(dataLogger, vo.idFrame), trajectory.typeGroundTruth)
        trajectory.AddPointsToAxis(trajectory.trajectory[vo.idFrame], trajectory.typeTrajectory) 
        trajectory.PrintTrajectory()

        vo.LoadFrames()        
        vo.DetectingFeaturesFASTMethod()
        
        vo.LoadFrames()   

        for i in tqdm(range(len(vo.framesStored))):
            vo.TrackingFutures()
            vo.CalculateEssentialMatrix()
            
            trajectory.AddPointsToAxis(groundTruth.GetPose(dataLogger, vo.idFrame), trajectory.typeGroundTruth) # rever idFrame
            trajectory.AddPointsToAxis(trajectory.GetTrajectory()[vo.idFrame], trajectory.typeTrajectory)
            trajectory.PrintTrajectory()
          
            if(vo.idStored ==  vo.numFramesToLoad):
                break
            vo.LoadFrames()
              
            cv2.waitKey(1)
            
    except IndexError:
        return
    # except MemoryError:
        # print("Erro: Index error \nFim de programa.")

    totalDistanceGroundTruth = 0.0
    totalDistanceTrajctory = 0.0
    distanceDifference = 0.0
    for i in range(1, len(trajectory.xValuesTrajectory)):
        totalDistanceTrajctory += math.sqrt( (trajectory.xValuesTrajectory[i] - trajectory.xValuesTrajectory[i - 1])**2 
                                   + (trajectory.yValuesTrajectory[i] - trajectory.yValuesTrajectory[i - 1])**2 
                                   + (trajectory.zValuesTrajectory[i] - trajectory.zValuesTrajectory[i - 1])**2 )
        totalDistanceGroundTruth += math.sqrt( (trajectory.xValuesGroundTruth[i] - trajectory.xValuesGroundTruth[i - 1])**2 
                                   + (trajectory.yValuesGroundTruth[i] - trajectory.yValuesGroundTruth[i - 1])**2 
                                   + (trajectory.zValuesGroundTruth[i] - trajectory.zValuesGroundTruth[i - 1])**2 )
        distanceDifference += (totalDistanceTrajctory - totalDistanceGroundTruth)
        
        
    print(f"Distance travelled: GrandTruth: {totalDistanceGroundTruth}, Trajecotry: {totalDistanceTrajctory}, Difference: {distanceDifference}")
    print(f"Erros mimimo x: {min(trajectory.errorX)}m, y: {min(trajectory.errorY)}m, z: {min(trajectory.errorZ)}m")
    print(f"Erros máximos x: {max(trajectory.errorX)}m, y: {max(trajectory.errorY)}m, z: {max(trajectory.errorZ)}m")
    
    averageFPS = round(vo.totalFPS / len(vo.framesStored), 2)
    print(f"fps médios: {averageFPS}")
    
    cv2.waitKey(0)

    return 1

if __name__ == '__main__':
    main()