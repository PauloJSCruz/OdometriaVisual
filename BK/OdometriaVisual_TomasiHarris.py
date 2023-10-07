# region Imports
import cv2
import numpy as np
import matplotlib.pylab as plt
import os
import logging
import pandas as pd
import random
import time
from datetime import datetime
import math
from tqdm import tqdm
# endregion

#region configs
SaveImage = False
# region save images
# Flag para guardar as imagens, se false apenas mostra a imagem final
DataTimeNow = datetime.now()
if SaveImage is True:
    OutputFolder = f'2 - OutputOdometria/{DataTimeNow.strftime("%d.%m.%Y_%H.%M.%S")}/'
    os.makedirs(OutputFolder, exist_ok=True)
# endregion

# fig, plotsFinish = plt.subplots(2, 2)
# # Remove o subplot vazio em plots[1, 2]
# fig.delaxes(plotsFinish[1, 1])


fig2 = plt.figure(figsize=(7,6))

# Configuração básica do log
OutputFolderDataLogger = f'DataLogger/DataLogger_{DataTimeNow.strftime("%d.%m.%Y")}'
os.makedirs(OutputFolderDataLogger, exist_ok=True)
logging.basicConfig(filename=f'{OutputFolderDataLogger}/dataLogger_{DataTimeNow.strftime("%H.%M.%S")}.txt', level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# Criando um objeto de log
dataLogger = logging.getLogger('dataLogger')

# Initialize variables for camera motion and position
prevCameraMotion = np.eye(4)
prevCameraPosition = np.zeros(3)
#endregion


def CreateVideoWithKittiFrames(numFramesVideo):

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
    for listImage in tqdm(listImages):
        image = cv2.imread(os.path.join(imagesDir, listImage))
        videoWrite.write(image)
        # print(listImage)
    # Libertar Recursos
    videoWrite.release()
    dataLogger.info('\n Kitti video is ready \n')


def DrawTracks(frame, mask, goodNew, goodOld, color):
    # Desenhar as linhas que representam o fluxo �ptico nos quadros
    for i, (new, old) in enumerate(zip(goodNew, goodOld)):
        a, b = new.ravel()
        c, d = old.ravel()
        a, b, c, d = int(a), int(b), int(c), int(d)  # Convert to integers
        mask = cv2.line(mask, (a, b), (c, d), (0, 255, 0), 2)
        frame = cv2.circle(frame, (a, b), 5, (0, 0, 255), 2)

    frameMask = cv2.add(frame, mask)
    return frameMask


def PrintFrame(textOnFrame):
    # Definir a largura e altura desejadas do frame
    FrameWidth = 1024
    FrameHeight = 700

    for i, (frame, text, value) in enumerate(textOnFrame):
        PrintTextOnFrame(frame, f"{text}", value, (50, 50 + (35 * i)))

    # Rodar e Redimensionar o frame para a largura e altura desejadas
    frame = cv2.resize(frame, (FrameWidth, FrameHeight))
    cv2.imshow('frame', frame)


def PrintTextOnFrame(Frame, texto, valor, pos):
    # Adicione texto ao frame
    texto = texto + ": " + str(valor)
    posicao = pos  # Posição do texto no frame
    fonte = cv2.FONT_HERSHEY_SIMPLEX  # Estilo da fonte
    escala = 1  # Escala do texto
    cor = (255, 255, 0)  # Cor do texto (verde no exemplo)
    espessura = 2  # Espessura da linha do texto

    cv2.putText(Frame, texto, posicao, fonte, escala, cor, espessura)


def LoadVideo():
    # region Load Video and path for save image
    # videoPath = "Recursos/VideoCorredor.mp4"
    # videoPath = "Recursos/VideoVoltaCompleta.mp4"
    # videoPath = "Recursos/Robotica1080.mp4"    
    # videoPath = "Recursos/KittiVideo_curva.mp4"
    # videoPath = "Recursos/KittiVideo_reta.mp4"    
    # videoPath = "Recursos/KittiVideo_percLongo.mp4"
    videoPath = "Recursos/KittiVideo.mp4"
    # endregion

    dataLogger.info(f'Video: {videoPath}')

    #CapturedVideo = cv2.VideoCapture(0) # Live
    CapturedVideo = cv2.VideoCapture(videoPath)    
    # Verificar se o video foi carregado corretamente
    if not CapturedVideo.isOpened():
         print("Erro ao abrir o vídeo.")
         exit()

    return CapturedVideo


def CaptureNewFrame(CapturedVideo):
    # region Frame
    # Read a frame from the video
    CaptureState, frame = CapturedVideo.read()
    if not CaptureState:
        CapturedVideo.release()
        cv2.destroyAllWindows()
        frame = []
        frameCanny = []
        return frame, frameCanny

    # Frame = CortarMetadeInferior(Frame)

    # endregion

    # region Preprocessing
    # Convert Frame RGB on Gray scale
    frameGray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    # Applying the canny algorithm
    frameCanny = cv2.Canny(frameGray, 250, 255, None, 3)
    # endregion
        
    # frameCanny =  frameGray  
    # # Aplique o filtro Gaussian Blur
    # # Defina o tamanho do kernel do filtro (deve ser ímpar)
    # tamanho_kernel = (5, 5)
    # desvio_padrao = 15  # Valor maior para mais desfoque
    # frameCanny = cv2.GaussianBlur(frameGray, tamanho_kernel, desvio_padrao)

    # cv2.imshow('Imagem Original', frameGray)
    # cv2.imshow('Imagem Desfocada', frameCanny)

    return frame, frameCanny


def DetectCorners(frameCanny):
    # region Detetor de cantos de Shi-Tomasi
    cornersDetected = []
    # # region Parâmetros do detector de cantos Shi-Tomasi
    ShiTomasiParams = dict(maxCorners=1000,
                           qualityLevel=0.1,
                           minDistance=50,
                           blockSize=7)
    # # endregion

    # region Parâmetros do detector de cantos Shi-Tomasi
    # ShiTomasiParams = dict(maxCorners=2000,
    #                        qualityLevel=0.2,
    #                        minDistance=25,
    #                        blockSize=7)
    # endregion

    cornersDetected = cv2.goodFeaturesToTrack(frameCanny, mask=None, **ShiTomasiParams, useHarrisDetector=True, k=0.04)
    
    return cornersDetected
    # endregion


def TrackFeatures(prevFrameGray, FrameGray, prevTrackedCorners):
    matchCorners = {}
    # region Parameters for lucas kanade optical flow
    LucasKanadeParams = dict(winSize=(25, 25),
                             maxLevel=2,
                             criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.01))
    # endregion
    
    # Calcular o fluxo ótico usando o método Lucas-Kanade
    newCorners, status, _err = cv2.calcOpticalFlowPyrLK( prevFrameGray, FrameGray, prevTrackedCorners, None, **LucasKanadeParams)
    # Save only newCorners that have done match
    trackedCorners = newCorners[status[:, 0] == 1]
    goodOldCorners = prevTrackedCorners[status[:, 0] == 1]
    differentiationCorners = []

    for i in range(len(trackedCorners)):
        differentiationCorners.append(trackedCorners[i] - goodOldCorners[i])
        matchCorners[i] = {
            "NewCanto": trackedCorners[i].tolist(),
            "PrevCanto": goodOldCorners[i].tolist(),
            "Differentiation": trackedCorners[i] - goodOldCorners[i]
        }
    for chave, valor in matchCorners.items():
        dataLogger.info(f'{chave}: {valor}')

    # Inicialize os arrays para armazenar os pontos x e y
    pointsX = []
    pointsY = []

    # Itere sobre a lista differentiationCorners e extraia os pontos x e y
    for array in differentiationCorners:
        x = array[:, 0]  # Extrai a primeira coluna (x) do array
        y = array[:, 1]  # Extrai a segunda coluna (y) do array
        pointsX.extend(x)  # Adiciona os pontos x ao array points_x
        pointsY.extend(y)  # Adiciona os pontos y ao array points_y

    # Converta os arrays para NumPy arrays se necessário
    pointsX = np.array(pointsX)
    pointsY = np.array(pointsY)

    # Criar um array para os índices dos pontos, para que possamos usá-los como identificadores
    indices = np.arange(len(pointsX))
    indices = np.arange(len(pointsY))

    
    # plt.ylabel('Eixo Y do Plot 2')
    # # Plot dos pontos x e y em um gráfico de dispersão
    # plt.scatter(indices, pointsX, label='Pontos X', color='green', marker='.')
    # plt.scatter(indices, pointsY, label='Pontos Y', color='red', marker='x')

    # # Configurações do gráfico
    # plt.title('Gráfico de Dispersão dos Pontos X e Y')
    # plt.xlabel('Índices dos Pontos')
    # plt.ylabel('Valores dos Pontos')
    # plt.legend()

    # # Exibir o gráfico
    # plt.show()
    # plt.pause(0.1)

    if newCorners is None:
        return None
    
    

    # dataLogger.info(f'\n goodOldCorners {goodOldCorners}')
    # dataLogger.info(f'\n trackedCorners {trackedCorners}')
    return trackedCorners, goodOldCorners


def MatrizFundamental(trackedCorners, goodOldCorners):
    fundamentalMatrix, mask = cv2.findFundamentalMat(trackedCorners, goodOldCorners, cv2.FM_RANSAC)

    # Refinar a matriz fundamental
    trackedCorners = trackedCorners[mask.ravel() == 1]
    goodOldCorners = goodOldCorners[mask.ravel() == 1]
    fundamentalMatrix, _ = cv2.findFundamentalMat(   trackedCorners, goodOldCorners, cv2.FM_8POINT)
    fundamentalMatrix = cv2.correctMatches(   fundamentalMatrix, trackedCorners.T, goodOldCorners.T)

    dataLogger.info(f'\n fundamentalMatrix \n {fundamentalMatrix}')
    return fundamentalMatrix


def MatrizEssencial(trackedCorners, goodOldCorners, intrinsicParameters):

    matrizEssencial, mask = cv2.findEssentialMat(goodOldCorners, trackedCorners, intrinsicParameters, method=cv2.RANSAC, prob=0.999, threshold=1.0)


    essencialMatrixRotation, essencialMatrixTranslation = DecomporMatrizEssencial( matrizEssencial, goodOldCorners, trackedCorners, intrinsicParameters)

    dataLogger.info(f'\n matrizEssencial \n {matrizEssencial}')
    dataLogger.info(f'\n rotation \n {essencialMatrixRotation}')
    dataLogger.info(f'\n essencialMatrixTranslation \n {essencialMatrixTranslation}')

    return essencialMatrixRotation, essencialMatrixTranslation


def DecomporMatrizEssencial(essentialMatrix, goodOldCorners, trackedCorners, cameraMatrix):
    # Recupera as matrizes de rotação e translação da matriz essencial
    _, breakDownRotation, breakDownTranslation, _ = cv2.recoverPose( essentialMatrix, goodOldCorners, trackedCorners, cameraMatrix)
    return breakDownRotation, breakDownTranslation


def PlotFrames(frames):
    # region plots parameters
    figFrames, axsFrames = plt.subplots(1, 3, figsize=(24, 12), dpi=300)
    # Reduz o espaçamento horizontal entre os subplots
    plt.subplots_adjust(wspace=0.01)
    # Reduz o espaçamento vertical entre os subplots
    plt.subplots_adjust(hspace=0.01)
    # endregion

    for i, data in enumerate(frames):
        frame, title = data
        axsFrames[i].imshow(frame, cmap='gray' if len(
            frame.shape) == 2 else None)
        axsFrames[i].set_title(title)
        axsFrames[i].axis('off')
    plt.savefig(
        f'{OutputFolder}{DataTimeNow.strftime("%d.%m.%Y_%H.%M.%S")}-5Plots.png', bbox_inches='tight')
    # plt.show(block=False)
    plt.pause(0.1)
    for ax in axsFrames:
        ax.clear()


def LoadCalibrationCamara(indexCamera):
    projectionMatrix = []
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
        intrinsicParameters = projectionMatrix[indexCamera][:, :3]
        print(intrinsicParameters)

    dataLogger.info(f'\n intrinsicParameters \n {intrinsicParameters}')
    # dataLogger.info(f'\n distortioncoefficient \n {distortioncoefficient}')
    return intrinsicParameters

# Function to update camera motion and position
def updateCameraMotionPosition(relativeRotation, relativeTranslation):
    global prevCameraMotion, prevCameraPosition

    # Construct the camera pose matrix from relative rotation and translation
    poseMatrix = np.eye(4)
    poseMatrix[:3, :3] = relativeRotation
    poseMatrix[:3, 3] = relativeTranslation.flatten()  # Flatten the (3, 1) array to (3,)

    # Update the camera motion by concatenating with the relative pose matrix
    cameraMotion = np.dot(prevCameraMotion, poseMatrix)

    # Update the camera position using the previous camera position and motion
    cameraPosition = prevCameraPosition + np.dot(prevCameraMotion[:3, :3], relativeTranslation.flatten())  # Flatten the (3, 1) array to (3,)

    # Update the previous camera motion and position with the new values
    prevCameraMotion = cameraMotion
    prevCameraPosition = cameraPosition

    return cameraMotion, cameraPosition

# Function to plot the trajectory
def plotTrajectory(trajectory):
    xValues = [pos[0] for pos in trajectory]
    yValues = [pos[2] for pos in trajectory]

    plt.figure()
    plt.plot(xValues, yValues, 'b-')
    plt.scatter(xValues, yValues, color='red', label='Camera Positions')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Camera Trajectory')
    plt.legend()
    plt.grid()
    plt.show()

def DrawTrajectory(groundTruth, trajectory, cameraPosition):
    # Criar um image em branco
    image = np.ones((700, 1024, 3), dtype=np.uint8) * 255  # image branco
     # Convert the camera positions to pixel coordinates on the image
    imageWidth, imageHeight = image.shape[1], image.shape[0]
    centerX, centerY = int(imageWidth / 2), int(imageHeight / 2)
    
    # scaledRealTrajectory = [(centerX + int(posReal[0]), centerY - int(posReal[1])) for posReal in groundTruth]
    scaledTrajectory = [(centerX + int(pos[0]), centerY - int(pos[2])) for pos in trajectory]

    # Draw the trajectory on the image as a line
    for i in range(1, len(scaledTrajectory)):
        cv2.line(image, scaledTrajectory[i - 1], scaledTrajectory[i], (0, 0, 255), 2)

    # Draw the current position as a red circle
    cv2.circle(image, (centerX + int(cameraPosition[0]), centerY - int(cameraPosition[1])), 5, (0, 255, 0), -1)


    # Add text with X, Y, and Z coordinates at the current position
    # textGroundTruth = f"X: {groundTruth[0]:.2f}, Y: {groundTruth[1]:.2f}, Z: {groundTruth[2]:.2f}"
    # textPositionGroundTruth = (centerX + int(groundTruth[0]), centerY - int(groundTruth[1]) - 20)
    # cv2.putText(image, textGroundTruth, textPositionGroundTruth, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    text = f"X: {cameraPosition[0]:.2f}, Y: {cameraPosition[1]:.2f}, Z: {cameraPosition[2]:.2f}"
    textPosition = (centerX + int(cameraPosition[0]), centerY - int(cameraPosition[1]) - 20)
    cv2.putText(image, text, textPosition, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # # Draw the camera positions as red circles on the image
    # for position in trajectory:
    #     cv2.circle(image, (int(position[0]), int(position[1])), 5, (0, 0, 255), -1)

    # # Draw the current camera position as a green circle on the image
    # cv2.circle(image, (int(cameraPosition[0]), int(cameraPosition[1])), 5, (0, 255, 0), -1)

    cv2.imshow('Frame with Trajectory', image)

    return image

def VisualizeMatches(image1, kp1, image2, kp2, match):
    """
    Visualize corresponding matches in two images

    Arguments:
    image1 -- the first image in a matched image pair
    kp1 -- list of the keypoints in the first image
    image2 -- the second image in a matched image pair
    kp2 -- list of the keypoints in the second image
    match -- list of matched features from the pair of images

    Returns:
    image_matches -- an image showing the corresponding matches on both image1 and image2 or None if you don't use this function
    """
    image_matches = cv2.drawMatches(image1,kp1,image2,kp2,match,None)
    plt.figure(figsize=(16, 6), dpi=100)
    plt.imshow(image_matches)
    
def GroundTruth2(numFrame, groundTruth):

    poses = pd.read_csv( 'Recursos/data_odometry_poses/dataset/poses/00.txt' , delimiter= ' ' , header= None ) 
    # print ( 'Tamanho do dataframe de pose:' , poses.shape) 
    poses.head()
    
    if groundTruth is None:
        groundTruth = np.zeros((len(poses), 3, 4))
        groundTruth[0] = np.array(poses.iloc[0]).reshape((3, 4))
        groundTruth[1] = np.array(poses.iloc[1]).reshape((3, 4))

    groundTruth[numFrame] = np.array(poses.iloc[numFrame]).reshape((3, 4))

    # %matplotlib widget
    
    traj = fig2.add_subplot(111, projection='3d')
    traj.plot(groundTruth[:,:,3][:,0], groundTruth[:,:,3][:,1], groundTruth[:,:,3][:,2])
    traj.set_xlabel('x')
    traj.set_ylabel('y')
    traj.set_zlabel('z')

    return groundTruth

def GroundTruth(numFrame):
    poses = pd.read_csv( 'Recursos/data_odometry_poses/dataset/poses/00.txt' , delimiter= ' ' , header= None ) 
    # print ( 'Tamanho do dataframe de pose:' , poses.shape) 
    print(f'poses: {poses.head()}%')

    return np.array(poses.iloc[numFrame]).reshape((3, 4))  


def main():
    # DataTimeNow, OutputFolder, plotsFinish, dataLogger = Init()
    fps = 0
    fpsMedia = 0
    cornersAverage = 0
    CountFrames = 0
    countIterations = 0
    ResetCorners = 5
    textOnFrame = []
    numFramesVideo = 2000

    # Initialize an empty list to store the camera positions for constructing the trajectory
    cameraTrajectory = [[0, 0,  0]]

    groundTruth = [[0, 0,  0]]

    prevCameraMotion = np.zeros((4, 3))
    prevCameraMotion[:3, :3] = np.eye(3)
    prevCameraPosition = np.zeros(3)

    intrinsicParameters = LoadCalibrationCamara(1)

    if os.path.exists("Recursos/KittiVideo.mp4") is False:
        CreateVideoWithKittiFrames(numFramesVideo)
    CapturedVideo = LoadVideo()

    # 1st Frame
    firstFrame, secondFrameCanny = CaptureNewFrame(CapturedVideo)   
    CountFrames += 1
    prevDetectedCorners = DetectCorners(secondFrameCanny)
    mask = np.zeros_like(firstFrame)
    # Process 2nd Frame
    frame, prevFrameCanny = CaptureNewFrame(CapturedVideo)
    frameCanny =  prevFrameCanny.copy()
    CountFrames += 1
    trackedCorners, goodOldCorners = TrackFeatures(secondFrameCanny, prevFrameCanny, prevDetectedCorners)
    
    
    # groundTruth = GroundTruth(CountFrames, groundTruth)

    while True:
        countIterations += 1
        CountFrames     += 1
        dataLogger.warning(f'Iteration: {countIterations}')
        # start time to calculate FPS
        start = time.time()

        loadingVideo = (CountFrames * 100) / numFramesVideo

        # Core Code
        if(countIterations > 1):        
            frame, frameCanny = CaptureNewFrame(CapturedVideo)
            # Stop programe when video have finished
            if len(frame) == 0:
                break


            trackedCorners, goodOldCorners = TrackFeatures(prevFrameCanny, frameCanny, prevDetectedCorners)

            # FrameSidebySide(prevFrame, frame, goodOldCorners, trackedCorners)

            # Rastriar novos cantos no caso de serem inferiores ao limite minimo, ou no caso do contador de frames atignir um valor predeterminado
            if trackedCorners is None or len(trackedCorners) < 10 or countIterations >= ResetCorners:
            # if trackedCorners is None or len(trackedCorners) < 10:
                countIterations = 1
                prevDetectedCorners = DetectCorners(prevFrameCanny)
                while prevDetectedCorners is None:
                    frame, frameCanny = CaptureNewFrame(CapturedVideo)
                    prevDetectedCorners = DetectCorners(frameCanny)
                trackedCorners, goodOldCorners = TrackFeatures(prevFrameCanny, frameCanny, prevDetectedCorners)
                mask = np.zeros_like(frame)
                
        # VisualizeMatches(prevFrameCanny, goodOldCorners, frameCanny, trackedCorners)

        frameMask = DrawTracks(frame, mask, trackedCorners, goodOldCorners, color=None)

        # Obter a matriz Essencial
        relativeRotation, relativeTranslation = MatrizEssencial(trackedCorners, goodOldCorners, intrinsicParameters)

        # Update camera motion and position
        cameraMotion, cameraPosition = updateCameraMotionPosition(relativeRotation, relativeTranslation)

        # Append the current camera position to the cameraTrajectory list
        cameraTrajectory.append(cameraPosition)

        groundTruth.append(GroundTruth(CountFrames-2))

        # Display the trajectory and camera positions on the frame
        DrawTrajectory(groundTruth, cameraTrajectory, cameraPosition)
        #######################################################################################################################################
        
        # Output Finish
        # if(countIterations > 200):
        # cameraMotion, cameraPosition = UpdateCameraMotion(
        #     relativeRotation, relativeTranslation, prevCameraMotion, prevCameraPosition)


        
        # Print indications on frame
        if prevDetectedCorners is not None:
            textOnFrame.append([frameMask, "FPS", round(fps, 2)])
            textOnFrame.append([frameMask, "FrameId", CountFrames])
            textOnFrame.append([frameMask, "trackedCorners", len(trackedCorners)])
            textOnFrame.append([frameMask, f"CountFrames", len(trackedCorners)])
            textOnFrame.append([frameMask, f"Reset features", ResetCorners])
            textOnFrame.append([frameMask, f"Loading %", loadingVideo])
            PrintFrame(textOnFrame)
            # Reset Text on Frame
            textOnFrame = []

        # Update de previous variables
        # prevCameraMotion = cameraMotion
        # prevCameraPosition = cameraPosition
        prevFrame = frame.copy()
        prevFrameCanny = frameCanny.copy()
        prevDetectedCorners = trackedCorners.copy()
                
        # calculate the FPS
        # # End time
        end = time.time() 
        fps = 1 / (end-start)
        fpsMedia = fpsMedia + fps
        cornersAverage += len(trackedCorners)
        print(f'Loading: {loadingVideo}%')
        cv2.waitKey(10)

    # Plot the camera trajectory
    plotTrajectory(cameraTrajectory)

    # Calculate the average
    fpsMedia /= countIterations
    cornersAverage /= countIterations
    print(f"\nFrames por segundo: {fpsMedia}\n")
    print(f"newCorners media: {cornersAverage}\n")


if __name__ == '__main__':
    main()
