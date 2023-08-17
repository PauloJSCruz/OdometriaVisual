# region Imports
import cv2
import numpy as np
import math
import matplotlib.pylab as plt
import time
from datetime import datetime
import os
import random
import logging
# endregion
SaveImage = False

# region save images
# Flag para guardar as imagens, se false apenas mostra a imagem final
DataTimeNow = datetime.now()
if SaveImage is True:
    OutputFolder = f'2 - OutputOdometria/{DataTimeNow.strftime("%d.%m.%Y_%H.%M.%S")}/'
    os.makedirs(OutputFolder, exist_ok=True)
# endregion

fig, plotsFinish = plt.subplots(2, 2)
# Remove o subplot vazio em plots[1, 2]
fig.delaxes(plotsFinish[1, 1])

# Configuração básica do log
OutputFolderDataLogger = f'DataLogger/DataLogger_{DataTimeNow.strftime("%d.%m.%Y")}'
os.makedirs(OutputFolderDataLogger, exist_ok=True)
logging.basicConfig(filename=f'{OutputFolderDataLogger}/dataLogger_{DataTimeNow.strftime("%H.%M.%S")}.txt', level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# Criando um objeto de log
dataLogger = logging.getLogger('dataLogger')

# region Not Used
class LinearModel:
    def __init__(self):
        self.slope = 0
        self.intercept = 0

    def fit(self, data):
        x = [point[0] for point in data]
        y = [point[1] for point in data]
        if len(x) > 0:
            self.slope, self.intercept = np.polyfit(x, y, 1)

    def distance(self, point):
        x, y = point
        expected_y = self.slope * x + self.intercept
        return abs(y - expected_y)


# region Print Ransac
def RansacPrint(frame, newCorners):
    # region Definir os parâmetros para o algoritmo RANSAC
    numOfCoordinatesMin = 10  # Número mínimo de pontos para ajustar o modelo
    k = 100  # Número de iterações do RANSAC
    distanceToConsiderInlier = 25  # Limiar para considerar um ponto como inlier
    minInlinersForAccept = 10  # Número mínimo de inliers para aceitar o modelo
    # endregion

    # Criar uma instância do modelo linear
    linearModel = LinearModel()

    # Converter os pontos em um formato adequado para o RANSAC
    data = [(point[0][0], point[0][1]) for point in newCorners]

    # Executar o algoritmo RANSAC
    bestModel, bestConsensusSet = Ransac(data, linearModel, numOfCoordinatesMin, k, distanceToConsiderInlier, minInlinersForAccept)

    if bestModel is not None:
        x = np.array([point[0] for point in bestConsensusSet])
        y = bestModel.slope * x + bestModel.intercept
        for i in range(len(x) - 1):
            cv2.line(frame, (int(x[i]), int(y[i])), (int(
                x[i+1]), int(y[i+1])), (0, 255, 0), 2)
        DrawOutliersAndInliers(
            frame, data, bestModel, distanceToConsiderInlier)


def DrawOutliersAndInliers(frame, data, model, distanceToConsiderInlier):
    # Função para desenhar outliers e inliers no quadro

    for point in data:
        # Verificar se o ponto é um outlier ou inlier
        if model.distance(point) > distanceToConsiderInlier:
            color = (0, 0, 255)  # Vermelho (outlier)
        else:
            color = (0, 255, 0)  # Verde (inlier)
        # Converter as coordenadas do ponto para inteiros
        center = (int(point[0]), int(point[1]))
        # Desenhar a bola no quadro
        cv2.circle(frame, center, 5, color, -1)


def DrawTracks(frame, mask, goodNew, goodOld, color):
    # Desenhar as linhas que representam o fluxo óptico nos quadros
    for i, (new, old) in enumerate(zip(goodNew, goodOld)):
        a, b = new.ravel()
        c, d = old.ravel()
        a, b, c, d = int(a), int(b), int(c), int(d)  # Convert to integers
        mask = cv2.line(mask, (a, b), (c, d), (0, 255, 0), 2)
        frame = cv2.circle(frame, (a, b), 5, (0, 0, 255), 2)

    frameMask = cv2.add(frame, mask)
    return frameMask
# endregion


def ReadFilesTxt(file):
    with open(file, 'r') as f:
        linhas = f.readlines()
        # Inicialize uma lista vazia para armazenar os valores da matriz
        valores = []
        # Percorra cada linha
        for linha in linhas:
            # Remova os caracteres de espaço em branco e os colchetes
            linha = linha.strip().strip('[]')

            # Divida a linha em elementos separados por espaço em branco
            elementos = linha.split()

            # Converta cada elemento para float e adicione-o à lista de valores
            valores.extend(map(float, elementos))
    return valores


def CortarMetadeInferior(frame):
    altura, largura, _ = frame.shape
    metade_inferior = frame[altura//2:altura, :]
    return metade_inferior


def Ransac(data, model, numOfCoordinatesMin, k, distanceToConsiderInlier, minInlinersForAccept):
    bestModel = None
    bestConsensusSet = None
    maxInliers = 0

    if len(data) < numOfCoordinatesMin:
        numOfCoordinatesMin = len(data)

    for i in range(k):
        sample = random.sample(data, numOfCoordinatesMin)
        maybeInliers = [point for point in data if point not in sample]
        model.fit(sample)
        consensusSet = sample.copy()

        for point in maybeInliers:
            if model.distance(point) < distanceToConsiderInlier:
                consensusSet.append(point)

        if len(consensusSet) > minInlinersForAccept:
            if len(consensusSet) > maxInliers:
                maxInliers = len(consensusSet)
                bestModel = model
                bestConsensusSet = consensusSet

    return bestModel, bestConsensusSet


def FrameSidebySide(prevFrame, frame, prevDetectedCorners, trackedCorners):
    # Teste
    # Definir a largura e altura desejadas do frame
    FrameWidth = 1680
    FrameHeight = 700
    combined_frame = cv2.hconcat((frame, prevFrame))
    # Desenhar uma linha entre os pontos nos quadros atual e anterior
    if prevDetectedCorners is not None:
        for i in range(len(trackedCorners)):
            prev_pt = tuple(map(int, prevDetectedCorners[i].ravel()))
            current_pt = tuple(map(int, trackedCorners[i].ravel()))
            # Gerar uma cor RGB aleatória
            color = tuple(np.random.randint(0, 255, 3).tolist())
            # Desenhar a linha com a cor aleatória no quadro combinado
            cv2.line(combined_frame, prev_pt, (current_pt[0] + prevFrame.shape[1], current_pt[1]), color, 1)
            # Desenhar círculos sem preenchimento nas extremidades da linha
            cv2.circle(combined_frame, prev_pt, 5, color, -1)
            cv2.circle(combined_frame, (current_pt[0] + prevFrame.shape[1], current_pt[1]), 5, color, -1)

            # Rodar e Redimensionar o frame para a largura e altura desejadas
            combined_frame = cv2.resize(combined_frame, (FrameWidth, FrameHeight))
            cv2.imshow('Combined Frames', combined_frame)
    # 
# endregion


def CaptureNewKittiFrame():
    imagesDir = "Recursos\image_l"
    listImages = sorted(os.listdir(imagesDir))
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
    # Libertar Recursos
    videoWrite.release()
    dataLogger.info('\n Kitti video is ready \n')


def DisplayFrame(textOnFrame):
    # Definir a largura e altura desejadas do frame
    FrameWidth = 1024
    FrameHeight = 700

    for i, (frame, text, value) in enumerate(textOnFrame):
        PrintOnFrame(frame, f"{text}", value, (50, 50 + (35 * i)))

    # Rodar e Redimensionar o frame para a largura e altura desejadas
    frame = cv2.resize(frame, (FrameWidth, FrameHeight))
    cv2.imshow('frame', frame)


def PrintOnFrame(Frame, texto, valor, pos):
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
    # VideoPath = "Recursos/VideoCorredor.mp4"
    # VideoPath = "Recursos/VideoVoltaCompleta.mp4"
    # videoPath = "Recursos/Robotica1080.mp4"    
    videoPath = "Recursos/KittiVideo_curva.mp4"
    # videoPath = "Recursos/KittiVideo_reta.mp4"    
    # videoPath = "Recursos/KittiVideo_percLongo.mp4"
    dataLogger.info(f'Video: {videoPath}')    
    return cv2.VideoCapture(videoPath)
    # # CapturedVideo = cv2.VideoCapture(0)    # Live
    # # Verificar se o video foi carregado corretamente
    # if not CapturedVideo.isOpened():
    #     print("Erro ao abrir o vídeo.")
    #     exit()
    # endregion


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
    # frameCanny = cv2.Canny(frameGray, 250, 255, None, 3)
    # endregion
        
    frameCanny =  frameGray  
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


# Print Output
def UpdateCameraMotion(R, T, prevCameraMotion, prevCameraPosition):
    # Atualiza a estimativa de movimento da câmera
    cameraMotion = np.dot(prevCameraMotion, np.hstack((R, T)))

    # Atualiza a posição estimada da câmera no mundo
    cameraPosition = prevCameraPosition + np.dot(prevCameraMotion[:, :3], T)

    # Reduz o espaçamento horizontal entre os subplots
    # plt.subplots_adjust(wspace=0.01)
    # Reduz o espaçamento vertical entre os subplots
    # plt.subplots_adjust(hspace=0.01)

    plotTrajectory = plotsFinish[0, 0]
    # Plot da cameraPosition
    plotTrajectory.set_xlabel('X')
    plotTrajectory.set_ylabel('Y')
    plotTrajectory.axis('equal')
    plotTrajectory.set_title('Plot Camera Position')
    plotTrajectory.plot(cameraPosition[0], cameraPosition[1], 'ro')

    # Plot da cameraMotion
    # region plot 3D
    # # Adiciona o subplot 3D em plots[1, 2]
    # plotMotion3D = plotsFinish[1, 1]
    # plotMotion3D = fig.add_subplot(2, 2, 4, projection='3d')
    # plotMotion3D.quiver(
    #     0, 0, 0, cameraMotion[0, 3], cameraMotion[1, 3], cameraMotion[2, 3])
    # plotMotion3D.set_xlim([-1, 1])
    # plotMotion3D.set_ylim([-1, 1])
    # plotMotion3D.set_zlim([-1, 1])

    # plotMotion3D.set_xlabel('X')
    # plotMotion3D.set_ylabel('Y')
    # plotMotion3D.set_zlabel('Z')
    # plotMotion3D.set_title('Plot Camera Motion')
    # endregion

    plt.tight_layout()
    plt.show()
    plt.pause(0.1)

    return cameraMotion, cameraPosition


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


def LoadCalibrationCamara():
    projectionMatrix = []
    with open('CalibrationCam/calib_reta.txt') as fileCalib:
        for line in fileCalib: 
            elementos = np.fromstring(line, dtype=np.float64, sep=' ')
            matriz = np.reshape(elementos, (3,4))
            projectionMatrix.append(matriz)

        for i, matriz in enumerate(projectionMatrix):
            print(f"Matriz P{i}:")
            print(projectionMatrix)

        # [: -> todas as linhas da matriz , :3 -> primeiras três colunas da matriz]
        intrinsicParameters = projectionMatrix[0][:, :3]
        print(intrinsicParameters)

    dataLogger.info(f'\n intrinsicParameters \n {intrinsicParameters}')
    # dataLogger.info(f'\n distortioncoefficient \n {distortioncoefficient}')
    return intrinsicParameters


##############################################################################################################################################

# Initialize variables for camera motion and position
prevCameraMotion = np.eye(4)
prevCameraPosition = np.zeros(3)

# Function to update camera motion and position
def updateCameraMotionPosition(relativeRotation, relativeTranslation):
    global prevCameraMotion, prevCameraPosition

    # Construct the camera pose matrix from relative rotation and translation
    pose_matrix = np.eye(4)
    pose_matrix[:3, :3] = relativeRotation
    pose_matrix[:3, 3] = relativeTranslation.flatten()  # Flatten the (3, 1) array to (3,)

    # Update the camera motion by concatenating with the relative pose matrix
    cameraMotion = np.dot(prevCameraMotion, pose_matrix)

    # Update the camera position using the previous camera position and motion
    cameraPosition = prevCameraPosition + np.dot(prevCameraMotion[:3, :3], relativeTranslation.flatten())  # Flatten the (3, 1) array to (3,)

    # Update the previous camera motion and position with the new values
    prevCameraMotion = cameraMotion
    prevCameraPosition = cameraPosition

    return cameraMotion, cameraPosition

# Function to plot the trajectory
def plotTrajectory(trajectory):
    x_values = [pos[0] for pos in trajectory]
    y_values = [pos[1] for pos in trajectory]

    plt.figure()
    plt.plot(x_values, y_values, 'b-')
    plt.scatter(x_values, y_values, color='red', label='Camera Positions')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Camera Trajectory')
    plt.legend()
    plt.grid()
    plt.show()

def DrawTrajectory(trajectory, cameraPosition):
    # Criar um canvas em branco
    image = np.ones((700, 1024, 3), dtype=np.uint8) * 255  # Canvas branco
     # Convert the camera positions to pixel coordinates on the canvas
    canvas_width, canvas_height = image.shape[1], image.shape[0]
    center_x, center_y = int(canvas_width / 2), int(canvas_height / 2)
    scaled_trajectory = [(center_x + int(pos[0]), center_y - int(pos[1])) for pos in trajectory]

    # Draw the trajectory on the canvas as a line
    for i in range(1, len(scaled_trajectory)):
        cv2.line(image, scaled_trajectory[i - 1], scaled_trajectory[i], (0, 0, 255), 2)

    # Draw the current position as a red circle
    cv2.circle(image, (center_x + int(cameraPosition[0]), center_y - int(cameraPosition[1])), 5, (0, 255, 0), -1)

    # Add text with X, Y, and Z coordinates at the current position
    text = f"X: {cameraPosition[0]:.2f}, Y: {cameraPosition[1]:.2f}, Z: {cameraPosition[2]:.2f}"
    text_position = (center_x + int(cameraPosition[0]), center_y - int(cameraPosition[1]) - 20)
    cv2.putText(image, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # # Draw the camera positions as red circles on the image
    # for position in trajectory:
    #     cv2.circle(image, (int(position[0]), int(position[1])), 5, (0, 0, 255), -1)

    # # Draw the current camera position as a green circle on the image
    # cv2.circle(image, (int(cameraPosition[0]), int(cameraPosition[1])), 5, (0, 255, 0), -1)

    cv2.imshow('Frame with Trajectory', image)

    return image

##############################################################################################################################################

def main():
    # DataTimeNow, OutputFolder, plotsFinish, dataLogger = Init()
    fps = 0
    fpsMedia = 0
    cornersAverage = 0
    CountFrames = 0
    countIterations = 0
    ResetCorners = 1
    textOnFrame = []

    # Initialize an empty list to store the camera positions for constructing the trajectory
    cameraTrajectory = [[0, 0,  0]]

    prevCameraMotion = np.zeros((4, 3))
    prevCameraMotion[:3, :3] = np.eye(3)
    prevCameraPosition = np.zeros(3)

    intrinsicParameters = LoadCalibrationCamara()

    # CaptureNewKittiFrame()
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

    while True:
        countIterations += 1
        CountFrames     += 1
        dataLogger.warning(f'Iteration: {countIterations}')
        # start time to calculate FPS
        start = time.time()

        # Core Code
        if(countIterations > 1):        
            frame, frameCanny = CaptureNewFrame(CapturedVideo)
            # Stop programe when video have finished
            if len(frame) == 0:
                break

            trackedCorners, goodOldCorners = TrackFeatures(prevFrameCanny, frameCanny, prevDetectedCorners)

            # FrameSidebySide(prevFrame, frame, goodOldCorners, trackedCorners)

            # Rastriar novos cantos no caso de serem inferiores ao limite minimo, ou no caso do contador de frames atignir um valor predeterminado
            if trackedCorners is None or len(trackedCorners) < 10 or CountFrames >= ResetCorners:
            # if trackedCorners is None or len(trackedCorners) < 10:
                CountFrames = 0
                prevDetectedCorners = DetectCorners(prevFrameCanny)
                while prevDetectedCorners is None:
                    frame, frameCanny = CaptureNewFrame(CapturedVideo)
                    prevDetectedCorners = DetectCorners(frameCanny)
                trackedCorners, goodOldCorners = TrackFeatures(prevFrameCanny, frameCanny, prevDetectedCorners)
                mask = np.zeros_like(frame)
            

        frameMask = DrawTracks(frame, mask, trackedCorners, goodOldCorners, color=None)

        # Obter a matriz Essencial
        relativeRotation, relativeTranslation = MatrizEssencial(trackedCorners, goodOldCorners, intrinsicParameters)

        # Update camera motion and position
        cameraMotion, cameraPosition = updateCameraMotionPosition(relativeRotation, relativeTranslation)

        # Append the current camera position to the cameraTrajectory list
        cameraTrajectory.append(cameraPosition)

        # Display the trajectory and camera positions on the frame
        DrawTrajectory(cameraTrajectory, cameraPosition)
        #######################################################################################################################################
        

        # Output Finish
        # if(countIterations > 200):
        # cameraMotion, cameraPosition = UpdateCameraMotion(
        #     relativeRotation, relativeTranslation, prevCameraMotion, prevCameraPosition)


        
        # Print indications on frame
        if prevDetectedCorners is not None:
            textOnFrame.append([frameMask, "FPS", round(fps, 2)])
            textOnFrame.append([frameMask, "FrameId", len(trackedCorners)])
            textOnFrame.append([frameMask, "trackedCorners", len(trackedCorners)])
            textOnFrame.append([frameMask, f"CountFrames {ResetCorners}", CountFrames])
            DisplayFrame(textOnFrame)
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
