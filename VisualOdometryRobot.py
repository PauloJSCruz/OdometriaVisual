# region Imports
# from asyncio.windows_events import NULL
import cv2
import numpy as np
import matplotlib.pylab as plt
import pandas as pd
import os
import logging
from datetime import datetime
from tqdm import tqdm
import ManualPoints
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
    def __init__(self, dataLogger, IndexCamera):
        self.idCamera = IndexCamera
        self.numFramesToLoad = 0
        # self.dataLogger = dataLogger
        self.filePath = f'Recursos/00/image_{self.idCamera}'
        # self.filePath = f'Recursos/Teste'
        self.idFrame = 0
        self.liveON = False
        self.framesStored = []
        self.framesLoaded = []
        self.intrinsicParameters = []

    def CalibrationFile(self):
        # Define o caminho para o arquivo de calibração
        file = "CalibrationCam/calib.txt"

        try:
            with open(file) as fileCalib:
                for line in fileCalib:
                    if line.startswith(f"P{self.idCamera}:"):
                        # Extrai os números da linha, ignorando o identificador "P0:"
                        elementos = np.fromstring(line[3:], sep=' ', dtype=np.float64)
                        # Reorganiza os elementos para formar a matriz de projeção 3x4
                        matriz_projecao = np.reshape(elementos, (3, 4))
                        # Extrai os parâmetros intrínsecos (as três primeiras colunas da matriz de projeção)
                        self.intrinsicParameters = matriz_projecao[:, :3]
                        print("Parâmetros intrínsecos:")
                        print(self.intrinsicParameters)
                        break  # Encerra o loop após processar a linha desejada

        except FileNotFoundError:
            print(f"Arquivo não encontrado: {file}")
            return

        # Assumindo que # self.dataLogger.info é um método válido para registrar informação
        # self.dataLogger.info(f'\nParâmetros Intrínsecos:\n{self.intrinsicParameters}')        
        # dataLogger.info(f'\n distortioncoefficient \n {distortioncoefficient}')

    def LoadFrames(self):
        if self.liveON is False:
            if ( len(self.framesStored) == 0):
                framePath = [os.path.join(self.filePath, file) for file in sorted(os.listdir(self.filePath))][:self.numFramesToLoad]
                self.framesStored = [cv2.imread(path) for path in framePath][:self.numFramesToLoad]
                return print( '\n Frames Loaded \n')
            else:
                self.framesLoaded.append(self.framesStored[self.idFrame])
                self.idFrame = len(self.framesLoaded) - 1
                # self.PrintFrame()
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
        self.framesLoaded.append( cv2.VideoCapture(0) )

    def PrintFrame(self):
            cv2.imshow('Frame', self.framesLoaded[self.idFrame])

    def PrintCustomFrame(self, frame):
        cv2.imshow('Custom Frame', frame)

class GroundTruth:    
    def __init__(self, dataLogger):
        self.posesReaded = pd.read_csv( f'Recursos/data_odometry_poses/dataset/poses/00.txt' , delimiter= ' ' , header= None ) 
        # print ( 'Tamanho do dataframe de pose:' , poses.shape) 
        # print(f'posesReaded: {self.posesReaded.head()}\n')
        self.poses = (np.array(self.posesReaded.iloc[0]).reshape((3, 4)))
        
    def GetPose(self, dataLogger, idFrame):
        self.poses = (np.array(self.posesReaded.iloc[idFrame]).reshape((3, 4)))
        dataLogger.info(f'\n Ground Truth idFrame({idFrame}) : \n {self.poses}')
        return self.poses

class VisualOdometry (Camera):
    def __init__(self, dataLogger, indexCamera, numFramestoLoad):
        super().__init__(dataLogger, indexCamera)
        self.numFramesToLoad = numFramestoLoad
        self.LoadFrames()
        self.CalibrationFile()
        self.ResetCorners = 5
        # self.dataLogger = dataLogger
        self.featuresDetected = []
        self.featuresTracked = []
        self.prevFeaturesTracked = []
        self.essencialMatrix = []
        self.rotationMatrix = []
        self.translationMatrix = []
        self.mask = np.zeros_like(self.framesStored[0])
        # Analized needed
        self.firstFrame = []
        self.featuresTrackedReset = False
        self.idFrame = 0
        self.idFramePreviuos = 0
        self.idFrameTracked = 0

        # Cria o objeto FAST com parâmetros específicos
        self.fastDetector = cv2.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True, type=2 )
                 
    def DetectingFeaturesFastMethod(self):
        # Verifica se o frame atual está carregado
        if self.framesLoaded[self.idFrame] is not None:
            
            # Converte a imagem para escala de cinza
            frameGray = cv2.cvtColor(self.framesLoaded[self.idFrame], cv2.COLOR_BGR2GRAY)
            # self.PrintCustomFrame(frameGray)
            
            # Encontra os pontos de interesse usando o detector FAST
            keypoints = self.fastDetector.detect(frameGray, None)
            
            # Converte os keypoints para um array numpy
            keypoints_np = np.array([kp.pt for kp in keypoints], dtype=np.float32)
            
            # Adiciona os keypoints detectados à lista de características
            self.featuresDetected.append(keypoints_np)
            
            # Determina o índice das características detectadas para este frame
            idfeaturesDetected = len(self.featuresDetected) - 1
            
            # Se este é o primeiro frame, inicializa featuresTracked com as características detectadas
            # Para outros frames, atualiza featuresTracked com as novas características detectadas
            if self.idFrame == 0:
                self.featuresTracked.append(self.featuresDetected[idfeaturesDetected])
            else:
                # Ajuste: Isso atualiza o valor para o frame atual
                self.featuresTracked[self.idFrame - 1] = self.featuresDetected[idfeaturesDetected]
            
            # A linha comentada abaixo sugere que você está tentando logar as características detectadas.
            # Certifique-se de que dataLogger está definido corretamente e descomente a linha abaixo, se necessário.
            # self.dataLogger.info(f'\n featuresDetected ({idfeaturesDetected}) \n {self.featuresDetected[idfeaturesDetected]}')
    
    def TrackingFutures(self):
        # Parameters for lucas kanade optical flow
        LucasKanadeParams = dict(winSize=(21, 21),  # Janela um pouco maior para capturar mais contexto
                                 maxLevel=3,  # Considera mais níveis na pirâmide para lidar com movimentos maiores
                                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.01))  # Critérios mais estritos para precisão             
        
        if ( self.idFrame == self.idFramePreviuos + 5 ):
            self.idFramePreviuos = self.idFrame
            self.DetectingFeaturesFastMethod()
            # self.DetectingFutures()
            self.featuresTrackedReset = True
            self.mask = np.zeros_like(self.framesLoaded[0])
            # print("\n Features Tracked = Features Detected")
        else:
            self.featuresTrackedReset = False

        # Optical Flow only with previous features detected
        opticalFlow, status, _err = cv2.calcOpticalFlowPyrLK( self.framesLoaded[self.idFrame - 1], self.framesLoaded[self.idFrame],
                                                               self.featuresTracked[self.idFrame - 1] , None, **LucasKanadeParams)

        # Remove os pontos que não tiveram correspondência no featuresTracked atual
        self.featuresTracked[self.idFrame - 1] = self.featuresTracked[self.idFrame - 1][status[:, 0] == 1]

        # Save only newCorners that have done match
        self.featuresTracked.append(opticalFlow[status[:, 0] == 1])       
        
        # self.dataLogger.info(f'\n featuresTracked ({self.idFrame}) \n {self.featuresTracked[self.idFrameTracked]}')
        return True

    def MatrixEssencial(self):
        
        self.essencialMatrix, mask = cv2.findEssentialMat(self.featuresTracked[self.idFrame - 1], self.featuresTracked[self.idFrame], self.intrinsicParameters, method = cv2.RANSAC, prob = 0.99, threshold = 0.1, maxIters = 100)

        self.DecomporMatrizEssencial()

        # self.process_frame()

        # self.dataLogger.info(f'\n matrizEssencial \n {self.essencialMatrix}')
        # self.dataLogger.info(f'\n rotation \n {self.rotationMatrix}')
        # self.dataLogger.info(f'\n essencialMatrixTranslation \n {self.translationMatrix}')
    
    def DecomporMatrizEssencial(self):
        # Recupera as matrizes de rotação e translação da matriz essencial
        _, self.rotationMatrix, self.translationMatrix, _ = cv2.recoverPose( self.essencialMatrix, self.featuresTracked[self.idFrame - 1], self.featuresTracked[self.idFrame], self.intrinsicParameters)

    def draw_epilines(self, img1, img2, ):
        """Desenha linhas epipolares para pontos correspondentes entre dois frames."""
        # Converter para cores se as imagens forem em escala de cinza
        img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR) if img1.ndim == 2 else img1
        img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR) if img2.ndim == 2 else img2
        
        # Calcular linhas epipolares nos dois frames
        lines1 = cv2.computeCorrespondEpilines(self.featuresTracked[self.idFrame].reshape(-1, 1, 2), 2, self.essencialMatrix)
        lines1 = lines1.reshape(-1, 3)
        lines2 = cv2.computeCorrespondEpilines(self.featuresTracked[self.idFrame - 1].reshape(-1, 1, 2), 1, self.essencialMatrix)
        lines2 = lines2.reshape(-1, 3)
        
        # Desenhar as linhas nos frames
        for r, pt1, pt2 in zip(lines1, self.featuresTracked[self.idFrame - 1], self.featuresTracked[self.idFrame]):
            color = tuple(np.random.randint(0, 255, 3).tolist())
            x0, y0 = map(int, [0, -r[2]/r[1] ])
            x1, y1 = map(int, [img1.shape[1], -(r[2]+r[0]*img1.shape[1])/r[1] ])
            img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1)
            img1 = cv2.circle(img1, tuple(pt1), 5, color, -1)
        for r, pt1, pt2 in zip(lines2, self.featuresTracked[self.idFrame], self.featuresTracked[self.idFrame - 1]):
            color = tuple(np.random.randint(0, 255, 3).tolist())
            x0, y0 = map(int, [0, -r[2]/r[1] ])
            x1, y1 = map(int, [img2.shape[1], -(r[2]+r[0]*img2.shape[1])/r[1] ])
            img2 = cv2.line(img2, (x0, y0), (x1, y1), color, 1)
            img2 = cv2.circle(img2, tuple(pt2), 5, color, -1)
            
        return img1, img2

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
    
    def PrintEpipolarGeometry(self, frame1, frame2, essential_matrix):
        # Decompose the essential matrix to get the fundamental matrix
        _, fundamental_matrix, _ = cv2.SVDecomp(essential_matrix)

        idFrameTracked = self.idFrame - 1

        # Ensure that both frames are in grayscale
        frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        # Detect features in both frames
        features1 = self.featuresTracked[self.idFrame - 1]
        features2 = self.featuresTracked[self.idFrame]

        # Draw epipolar lines on both frames
        lines2 = cv2.computeCorrespondEpilines(features1.reshape(-1, 1, 2), 1, fundamental_matrix)
        lines2 = lines2.reshape(-1, 3)
        lines1 = cv2.computeCorrespondEpilines(features2.reshape(-1, 1, 2), 2, fundamental_matrix)
        lines1 = lines1.reshape(-1, 3)

        frame1_with_lines = cv2.cvtColor(frame1, cv2.COLOR_GRAY2BGR)
        frame2_with_lines = cv2.cvtColor(frame2, cv2.COLOR_GRAY2BGR)

        for r, pt1, pt2 in zip(lines2, self.featuresTracked[idFrameTracked], self.featuresTracked[idFrameTracked - 1]):
            color = tuple(np.random.randint(0, 255, 3).tolist())
            x0, y0 = map(int, [0, -r[2]/r[1]])
            x1, y1 = map(int, [frame1.shape[1], -(r[0]*frame1.shape[1]+r[2])/r[1]])
            frame1_with_lines = cv2.line(frame1_with_lines, (x0, y0), (x1, y1), color, 2)
            frame1_with_lines = cv2.circle(frame1_with_lines, (x0, y0), 2, color, -1)
            frame2_with_lines = cv2.circle(frame2_with_lines, (x1, y1), 2, color, -1)

        for r, pt1, pt2 in zip(lines1, features2, features1):
            color = tuple(np.random.randint(0, 255, 3).tolist())
            x0, y0 = map(int, [0, -r[2]/r[1]])
            x1, y1 = map(int, [frame2.shape[1], -(r[0]*frame2.shape[1]+r[2])/r[1]])

            frame2_with_lines = cv2.line(frame2_with_lines, (x0, y0), (x1, y1), color, 2)
            frame2_with_lines = cv2.circle(frame2_with_lines, (x0, y0), 2, color, -1)
            frame1_with_lines = cv2.circle(frame1_with_lines, (x1, y1), 2, color, -1)
  
        # Display the frames with epipolar lines
        # cv2.imshow('Epipolar Lines', cv2.hconcat([frame1_with_lines, frame2_with_lines]) )

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
                    # cv2.line(combined_frame, prev_pt, (current_pt[0] + self.framesLoaded[self.idFrame - 1].shape[1], current_pt[1]), color, 1)
                    # Desenhar circulos sem preenchimento nas extremidades da linha
                    cv2.circle(combined_frame, prev_pt, 1, color, -1)
                    cv2.circle(combined_frame, (current_pt[0] + self.framesLoaded[self.idFrame - 1].shape[1], current_pt[1]), 1, color, -1)

                    # Rodar e Redimensionar o frame para a largura e altura desejadas
                    # combined_frame = cv2.resize(combined_frame, (FrameWidth, FrameHeight))
                # cv2.imshow('Combined Frames', combined_frame)

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
        if self.numPlots > 0:
            if self.numPlots is 1:     
                self.ax2d.legend()
                self.ax3d.legend()
                self.error.legend()
            plt.show()
            self.fig1d.savefig("Resultados/PlotError.pdf")
            self.fig2d.savefig("Resultados/Trajectory2D.pdf")
            self.fig3d.savefig("Resultados/Trajectory3D.pdf")

        else:
            print("No data to plot.")

    def AddPointsToAxis(self, trajectory, type):
        # Function to plot the trajectory

        if type is 'GroundTruth':
            self.xValuesGroundTruth.append(trajectory[0, 3])
            self.yValuesGroundTruth.append(trajectory[1, 3])
            self.zValuesGroundTruth.append(trajectory[2, 3])
            # print(f"GroundTruth : x: {trajectory[0, 3]}, y: {trajectory[1, 3]}, z: {trajectory[2, 3]}" )
            # self.dataLogger.info(f"GroundTruth : x: {trajectory[0, 3]}, y: {trajectory[1, 3]},  z: {trajectory[2, 3]}" )


        if type is 'Trajectory':
            # multiply trajectory by -1 for inverte for really trajecotry
            x = trajectory[0, 3] * (1)
            y = trajectory[1, 3] * (-1)
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
        self.imageTrajectory = np.ones((1000, 1920, 3), dtype=np.uint8) * 255  # image branco        
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

def main():
    try:
        idCamera = 0
        numFramesToLoad = 2000

        # instancias
        dataLogger = ConfigDataLogger()
        vo = VisualOdometry(dataLogger, idCamera, numFramesToLoad)
        trajectory = Trajectory(dataLogger, vo)

        vo.LoadFrames()

        vo.DetectingFeaturesFastMethod()
        vo.LoadFrames()

        for i in tqdm(range(len(vo.framesStored)- len(vo.featuresDetected))):
            vo.TrackingFutures()
            vo.MatrixEssencial()
            trajectory.AddPointsToAxis(trajectory.GetTrajectory(), trajectory.typeTrajectory)
            
            # Load number of iterations            
            vo.LoadFrames()
            
    # except IndexError:
    except MemoryError:
        print("Erro: Fim de programa.")
    
    return 1

if __name__ == '__main__':
    main()