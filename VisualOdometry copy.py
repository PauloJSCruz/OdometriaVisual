# region Imports
import cv2
import numpy as np
import matplotlib.pylab as plt
import pandas as pd
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
        self.frameHeight = 0 
        self.frameWidth = 0

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

        # Assumindo que # self.dataLogger.info é um método válido para registrar informação
        # self.dataLogger.info(f'\nParâmetros Intrínsecos:\n{self.intrinsicParameters}')        
        # dataLogger.info(f'\n distortioncoefficient \n {distortioncoefficient}')

    def LoadFrames(self):
        self.filePath = f'Recursos/00/image_{self.idCamera}'
        if (self.liveON == False):
            if ( len(self.framesStored) == 0):
                framePath = [os.path.join(self.filePath, file) for file in sorted(os.listdir(self.filePath))][:self.numFramesToLoad]
                self.framesStored = [cv2.imread(path) for path in framePath][:self.numFramesToLoad]
                self.frameHeight, self.frameWidth = self.framesStored[0].shape[:2]
                return print( '\n Frames Loaded \n')
            else:                
                self.framesLoaded.append(self.framesStored[self.idStored])
                self.idFrame = len(self.framesLoaded) - 1
                self.idStored += 1
                self.PrintFrame()
                return 

        if (self.liveON == True):
            # Capture new frame
            self.LiveCam()
            self.idFrame = len(self.framesLoaded) - 1
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
        if (os.path.exists("Recursos/KittiVideo.mp4") == False):
            numFramesVideo = numFrames
            imagesDir = "Recursos/00/image_2"
            listImages = sorted(os.listdir(imagesDir))
            listImages = listImages[:numFramesVideo]
            # Ler Primeira imagem para obter as dimensões
            firstImage = cv2.imread(os.path.join(imagesDir, listImages[0]))
            height, width, _ = firstImage.shape
            fps = 30
            # Criar o objeto VideoWriter
            videoWrite = cv2.VideoWriter("Recursos\\KittiVideo.mp4", cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, (width, height))
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
        self.framesLoaded.append( cv2.VideoCapture(0) )

    def PrintFrame(self):
        # Calcula o tempo decorrido entre o frame atual e o anterior
        currentTime = datetime.now()
        time = (currentTime - self.prevTime).total_seconds()
        if time > 0:
            fps = round(1 / time)
        self.prevTime = currentTime
        currentFrame = self.framesLoaded[self.idFrame].copy()
        cv2.putText (currentFrame, f'fps: {fps}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA )
        cv2.putText (currentFrame, f'Frame: {self.idFrame}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA )
        cv2.imshow('Frame', currentFrame)

    def PrintCustomFrame(self, text, frame):
        cv2.imshow(text, frame)

class GroundTruth:    
    def __init__(self, dataLogger):
        self.posesReaded = pd.read_csv( f'Recursos\\data_odometry_poses\\dataset\\poses\\00.txt' , delimiter= ' ' , header= None ) 
        # print ( 'Tamanho do dataframe de pose:' , poses.shape) 
        # print(f'posesReaded: {self.posesReaded.head()}\n')
        self.poses = (np.array(self.posesReaded.iloc[0]).reshape((3, 4)))
        
    def GetPose(self, dataLogger, idFrame):
        self.poses = (np.array(self.posesReaded.iloc[idFrame]).reshape((3, 4)))
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

        # Cria o objeto FAST com parâmetros específicos
        self.fastDetector = cv2.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True, type=2 )

    def FrameProcess(self):
        # Convert Frame RGB on Gray scale
        frameGray = cv2.cvtColor(self.framesLoaded[self.idFrame], cv2.COLOR_BGR2GRAY)        
        
        frameFiltered = frameGray
        # frameFiltered = self.BandPassFilter(frameGray)
        # tamanho_kernel = (7, 7)
        # desvio_padrao = 12  # Valor maior para mais desfoque
        # frameFiltered = cv2.GaussianBlur(frameGray, tamanho_kernel, desvio_padrao)

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
            # response = 150
            # keypointsgood = [ kp for kp in keypoints if kp.response > response ]
            # while(len(keypointsgood) < 25):
            #     keypointsgood = [ kp for kp in keypoints if kp.response > response ]
            #     response -= 5
            #     print(response)
            # keypoints = keypointsgood

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
        LucasKanadeParams = dict(winSize=(21, 21),  # Slightly larger window to capture more context
                                 maxLevel=3,  # Considers more levels in the pyramid to handle larger movements
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

        self.DrawFeaturesMatched()
        # self.FramesOverlapping(self.DrawFeaturesTracked(newFeatures, self.featuresTracked[self.idFrame - 1]))  
        
        self.dataLogger.info(f'\n featuresTracked ({self.idFrame}) \n {self.featuresTracked[self.idFrame]}')
        return True

    def DrawFeaturesMatched(self, numPoints = 5):
        if self.idFrame < 1:
            return

        newFrame = self.framesLoaded[self.idFrame]
        oldFrame = self.framesLoaded[self.idFrame - 1]
        cv2.putText (newFrame, f'Frame: {self.idFrame}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA )
        cv2.putText (oldFrame, f'Frame: {self.idFrame - 1}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA )

        # Cria uma imagem composta para visualização
        height1, width1 = self.frameHeight, self.frameWidth
        height2, width2 = self.frameHeight, self.frameWidth
        compositeImage = np.zeros((max(height1, height2), width1 + width2, 3), dtype=np.uint8)
        compositeImage[:height1, :width1] = oldFrame
        compositeImage[:height2, width1:width1 + width2] = newFrame

        # Seleciona um subconjunto aleatório dos pontos correspondentes
        numPoints = min(numPoints, len(self.featuresTracked[self.idFrame - 1]))
        indices = np.random.choice(len(self.featuresTracked[self.idFrame - 1]), numPoints)

        for i in indices:
            p1 = self.featuresTracked[self.idFrame - 1][i]
            p2 = self.featuresTracked[self.idFrame][i]
            pt1 = (int(p1[0]), int(p1[1]))
            pt2 = (int(p2[0] + width1), int(p2[1]))
            cv2.line(compositeImage, pt1, pt2, (0, 255, 0), 1)
            cv2.circle(compositeImage, pt1, 5, (255, 0, 255), -1)
            cv2.circle(compositeImage, pt2, 5, (0, 0, 255), -1)

        # Mostra a imagem com correspondências
        self.PrintCustomFrame("Featrues matched", self.resizeImage(compositeImage, 1920, 1080))

    # Função para redimensionar a imagem mantendo a proporção
    def resizeImage(self, image, screenWidth, screenHeight):
        
        # Calcular a razão de redimensionamento mantendo a proporção
        frameWidth = self.frameWidth * 2
        widthRatio = screenWidth / frameWidth
        heightRatio = screenHeight / self.frameHeight
        resizeRatio = min(widthRatio, heightRatio)

        # Calcular as novas dimensões
        newWidth = int(frameWidth * resizeRatio)
        newHeight = int(self.frameHeight * resizeRatio)

        # Redimensionar a imagem
        resizedImage = cv2.resize(image, (newWidth, newHeight), interpolation=cv2.INTER_AREA)
        
        return resizedImage

    def DrawFeaturesTracked(self, newFeatures, oldFeatures, numPoints = 5):
        verticalRange = 100
        # Filtre os pontos que estão dentro do intervalo vertical especificado
        verticalFiltered = [point for point in oldFeatures if abs(point[1] - (self.frameHeight / 2)) < verticalRange] 

        numPoints = min(numPoints, len(verticalFiltered))

        # Selecione os primeiros 'numPoints' pontos do intervalo filtrado
        selectedPoints = verticalFiltered[:numPoints]

        # Obtenha os índices dos pontos selecionados na lista original de oldFeatures
        indices = [np.where((oldFeatures == point).all(axis=1))[0][0] for point in selectedPoints]
        
        currentFrame = self.framesLoaded[self.idFrame].copy()

        for i in indices:
            p1 = self.featuresTracked[self.idFrame - 1][i]
            p2 = self.featuresTracked[self.idFrame][i]
            pt1 = (int(p1[0]), int(p1[1]))
            pt2 = (int(p2[0]), int(p2[1]))           
            self.mask = cv2.line(self.mask, pt1, pt2, (0, 255, 0), 1)
            circle = cv2.circle(currentFrame, pt1, 2, (0, 0, 255), 2)
            frameMask = cv2.add(circle, self.mask)
        return frameMask

    def CalculateEssentialMatrix(self):
        # Calculates the essential matrix using the tracked features
        self.essentialMatrix, mask = cv2.findEssentialMat(self.featuresTracked[self.idFrame], 
                                                          self.featuresTracked[self.idFrame - 1],
                                                          self.intrinsicParameters, 
                                                          method=cv2.RANSAC, prob=0.99, threshold=0.1, maxIters=100)
        

        # # Remove outliers
        # self.featuresTracked[self.idFrame - 1] = self.featuresTracked[self.idFrame - 1][mask.ravel() == 1]
        # self.featuresTracked[self.idFrame] = self.featuresTracked[self.idFrame][mask.ravel() == 1]

        # self.essentialMatrix, mask = cv2.findEssentialMat(self.featuresTracked[self.idFrame], 
        #                                                   self.featuresTracked[self.idFrame - 1],
        #                                                   self.intrinsicParameters, 
        #                                                   method=cv2.RANSAC, prob=0.99, threshold=1, maxIters=100)
        
        if ((self.essentialMatrix is not None) and (len(self.essentialMatrix) == 3)):
            self.DecomposeEssentialMatrix()  # Decomposes the essential matrix to extract rotation and translation

            F = self.FundamentalMatrix(self.essentialMatrix, self.intrinsicParameters)

            # # Extrai os pontos de features para passar para a função de desenho
            # # Você pode precisar ajustar como os pontos são extraídos de suas estruturas de dados
            # points1 = np.int32(self.featuresTracked[self.idFrame - 1])
            # points2 = np.int32(self.featuresTracked[self.idFrame])

            # # Desenha linhas epipolares nas imagens
            # img1EpipolarLines, img2EpipolarLines = self.DrawEpipolarLines(self.framesLoaded[self.idFrame -1], self.framesLoaded[self.idFrame], points1, points2, F)

            # # Exibe as imagens com linhas epipolares
            # cv2.imshow("Image 1 with Epipolar Lines", img1EpipolarLines)
            # cv2.imshow("Image 2 with Epipolar Lines", img2EpipolarLines)

            # self.dataLogger.info(f'\n essentialMatrix \n {self.essentialMatrix}')
            # self.dataLogger.info(f'\n rotation \n {self.rotationMatrix}')
            # self.dataLogger.info(f'\n essentialMatrixTranslation \n {self.translationMatrix}')
    
    def DecomposeEssentialMatrix(self):
        # Retrieves the rotation and translation matrices from the essential matrix
        _, self.rotationMatrix, self.translationMatrix, _ = cv2.recoverPose(self.essentialMatrix, 
                                                                            self.featuresTracked[self.idFrame], 
                                                                            self.featuresTracked[self.idFrame - 1], 
                                                                            self.intrinsicParameters)

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
    
    def FramesOverlapping(self, newFrame):    
        # Converte ambas as imagens para escala de cinza
        grayFrameOld = cv2.cvtColor(self.framesLoaded[self.idFrame - 1], cv2.COLOR_BGR2GRAY)

        # Cria o anaglifo usando o frame em escala de cinza e os canais de cores do frame atual
        # Este passo assume que você quer um efeito visual usando o vermelho do frame anterior e verde/azul do novo
        anaglyphFrame = cv2.merge((grayFrameOld, newFrame[:,:,1], newFrame[:,:,2]))

        # Add the number of detected corners to the image
        text = f"Number of corners: {len(self.featuresTracked[self.idFrame])}"
        cv2.putText(anaglyphFrame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        self.PrintCustomFrame("Frame with anaglyph", anaglyphFrame)

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
        self.errorIDs = []

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
            x = trajectory[0, 3] * (1)
            y = trajectory[1, 3] * (1)
            z = trajectory[2, 3] * (1)
            self.xValuesTrajectory.append(x)
            self.yValuesTrajectory.append(y)
            self.zValuesTrajectory.append(z)
            
            # Opening a file for appending the poinys
            self.fileOutput = open("Resultados/OutputTrajectory.txt", "a")
            self.fileOutput.write(f"{x} {y} {z}\n")
            self.fileOutput.close()
            
            # self.dataLogger.info(f"Trajectory : x: {trajectory[0, 3]},  y: {trajectory[1, 3]},  z: {trajectory[2, 3]}")
   
class Trajectory (Plots):
    def __init__(self, dataLogger, voInstance):
        super().__init__(dataLogger)
        self.vo = voInstance
        self.dataLogger = dataLogger
        self.allPointsTrajectory = []
        self.trajectory = np.identity(4)
        
        self.allPointsTrajectory.append(self.trajectory)
        self.typeTrajectory = 'Trajectory'
        self.typeGroundTruth = 'GroundTruth'
        # Criar um image em branco
        self.imageTrajectory = np.ones((1000, 1920, 3), dtype=np.uint8) * 255  # image branco        
        self.pos = np.zeros((3, 1), dtype=np.float32)
        self.rot = np.eye(3)

    def PrintTrajectory(self):        
        # Convert the camera positions to pixel coordinates on the image
        # centerX, centerZ = self.imageTrajectory.shape[1], self.imageTrajectory.shape[0]
        centerX, centerZ = int(self.imageTrajectory.shape[1] / 2), int(self.imageTrajectory.shape[0] / 2)
        centerZ = centerZ + 200
        self.errorIDs.append(self.vo.idFrame)
        colorGroundTruth = (255, 0, 0)
        colorTrajectory = (0, 0, 255)
        colorError = (125, 200, 0)

        textPositionGroundTruth = (10, 40)
        textPositionTrajectory = (10, 60)
        textPositionError = (10, 80)
       
        textPositionAxixZ = (10, centerZ)
        textPositionAxixX = (centerX  , self.imageTrajectory.shape[0] - 200)

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
        # The position is given by
        # C_n = C_(n-1) * T_n
        # The camera's position and orientation at time n is given by
        # C_n = R_(n,n-1) * C_(n-1) + T_(n,n-1)
        
        self.pos = self.pos + self.rot @ self.vo.translationMatrix  # Update the position using translation matrix
        self.rot = self.rot @ self.vo.rotationMatrix  # Update the rotation using rotation matrix
        self.trajectory = cv2.hconcat([self.rot, self.pos])  # Concatenates rotation and position to form the trajectory matrix

        # self.dataLogger.info(f'\n trajectory \n {self.trajectory}')        
        return self.trajectory

def main():
    try:
        # instancias
        dataLogger = ConfigDataLogger()
        groundTruth = GroundTruth(dataLogger)
        vo = VisualOdometry(dataLogger)
        trajectory = Trajectory(dataLogger, vo)

        # Analisa os argumentos manualmente
        for i in range(1, len(sys.argv), 2):
            if sys.argv[i] == '-id':
                vo.idCamera = int(sys.argv[i + 1])
            elif sys.argv[i] == '-numFrames':
                vo.numFramesToLoad = int(sys.argv[i + 1])
            elif sys.argv[i] == '-live':
                vo.liveON = bool(sys.argv[i + 1])
            elif sys.argv[i] == '-help':
                print("Flags:")
                print("-id: Id form camera")
                print("-numFrames: Number of frames that are load")
                print("-live: True or False if the frames are capture from robot camera")
                print("Struct: python meuprograma.py -id <idCamera> -numFrames <numFramesToLoad> -live <true>\n")
                sys.exit(1)
            else:
                print(f"Argumento desconhecido: {sys.argv[i]}")
                print("python meuprograma.py -id <idCamera> -numFrames <numFramesToLoad> -live <true>")
                sys.exit(1)

        if(vo.liveON == True):
            vo.LoadFrames()
            vo.mask = np.zeros_like(vo.framesLoaded[0])
            vo.framesLoaded = []
        if(vo.liveON == False):
            vo.LoadFrames()
            vo.mask = np.zeros_like(vo.framesStored[0])
        
        vo.CalibrationFile()

        # Start
        # 1st frame
        trajectory.AddPointsToAxis(groundTruth.GetPose(dataLogger, vo.idFrame), trajectory.typeGroundTruth)
        trajectory.AddPointsToAxis(trajectory.trajectory, trajectory.typeTrajectory) 
        trajectory.PrintTrajectory()

        fpsStart = datetime.now()
        vo.LoadFrames()        
        vo.DetectingFeaturesFASTMethod()
        
        vo.LoadFrames()   

        for i in tqdm(range(len(vo.framesStored))):
            vo.TrackingFutures()
            vo.CalculateEssentialMatrix()
            
            # scale = getAbsoluteScale(groundTruth.GetPose(dataLogger, vo.idFrame-1), groundTruth.GetPose(dataLogger, vo.idFrame))
            trajectory.AddPointsToAxis(groundTruth.GetPose(dataLogger, vo.idFrame), trajectory.typeGroundTruth) # rever idFrame
            trajectory.AddPointsToAxis(trajectory.GetTrajectory(), trajectory.typeTrajectory)
            trajectory.PrintTrajectory()
          
            if(vo.idStored ==  vo.numFramesToLoad):
                break
            vo.LoadFrames()
              
            # vo.DrawFeaturesTracked()
            cv2.waitKey(1)
            
    except IndexError:
    # except MemoryError:
        print("Erro: Fim de programa.")

    totalDistanceGroundTruth = 0
    totalDistanceTrajctory = 0
    for i in range(1, len(trajectory.xValuesTrajectory)):
        totalDistanceTrajctory += math.sqrt( (trajectory.xValuesTrajectory[i] - trajectory.xValuesTrajectory[i - 1])**2 
                                   + (trajectory.yValuesTrajectory[i] - trajectory.yValuesTrajectory[i - 1])**2 
                                   + (trajectory.zValuesTrajectory[i] - trajectory.zValuesTrajectory[i - 1])**2 )
    for i in range(1, len(trajectory.xValuesGroundTruth)):
        totalDistanceGroundTruth += math.sqrt( (trajectory.xValuesGroundTruth[i] - trajectory.xValuesGroundTruth[i - 1])**2 
                                   + (trajectory.yValuesGroundTruth[i] - trajectory.yValuesGroundTruth[i - 1])**2 
                                   + (trajectory.zValuesGroundTruth[i] - trajectory.zValuesGroundTruth[i - 1])**2 )
    print(f"Distance travelled: GrandTruth: {totalDistanceGroundTruth}, Trajecotry: {totalDistanceTrajctory}")
    print(f"Erros mimimo x: {min(trajectory.errorX)}m, y: {min(trajectory.errorY)}m, z: {min(trajectory.errorZ)}m")
    print(f"Erros máximos x: {max(trajectory.errorX)}m, y: {max(trajectory.errorY)}m, z: {max(trajectory.errorZ)}m")
    # matlab1(vo.idFrame, vo)
    trajectory.PrintPlots()
    
    cv2.waitKey(0)

    return 1

if __name__ == '__main__':
    main()