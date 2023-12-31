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

# fig2, plots2 = plt.subplots(2, 2, dpi=300)

# Configuração básica do log
OutputFolderDataLogger = f'DataLogger/DataLogger_{DataTimeNow.strftime("%d.%m.%Y")}'
os.makedirs(OutputFolderDataLogger, exist_ok=True)
logging.basicConfig(filename=f'{OutputFolderDataLogger}/dataLogger_{DataTimeNow.strftime("%H.%M.%S")}.txt', level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# Criando um objeto de log
dataLogger = logging.getLogger('dataLogger')


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


def PrintOnFrame(Frame, texto, valor, pos):
    # Adicione texto ao frame
    texto = texto + ": " + str(valor)
    posicao = pos  # Posição do texto no frame
    fonte = cv2.FONT_HERSHEY_SIMPLEX  # Estilo da fonte
    escala = 1  # Escala do texto
    cor = (255, 255, 0)  # Cor do texto (verde no exemplo)
    espessura = 2  # Espessura da linha do texto

    cv2.putText(Frame, texto, posicao, fonte, escala, cor, espessura)

# region Print Ransac


def RansacPrint(frame, Corners):
    # region Definir os parâmetros para o algoritmo RANSAC
    numOfCoordinatesMin = 10  # Número mínimo de pontos para ajustar o modelo
    k = 100  # Número de iterações do RANSAC
    distanceToConsiderInlier = 25  # Limiar para considerar um ponto como inlier
    minInlinersForAccept = 10  # Número mínimo de inliers para aceitar o modelo
    # endregion

    # Criar uma instância do modelo linear
    linearModel = LinearModel()

    # Converter os pontos em um formato adequado para o RANSAC
    data = [(point[0][0], point[0][1]) for point in Corners]

    # Executar o algoritmo RANSAC
    bestModel, bestConsensusSet = Ransac(
        data, linearModel, numOfCoordinatesMin, k, distanceToConsiderInlier, minInlinersForAccept)

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


def LoadVideo():
    # region Load Video and path for save image
    # VideoPath = "Recursos/VideoCorredor.mp4"
    # VideoPath = "Recursos/VideoVoltaCompleta.mp4"
    VideoPath = "Recursos/Robotica1080.mp4"
    return cv2.VideoCapture(VideoPath)
    # # CapturedVideo = cv2.VideoCapture(0)    # Live
    # # Verificar se o video foi carregado corretamente
    # if not CapturedVideo.isOpened():
    #     print("Erro ao abrir o vídeo.")
    #     exit()
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

# Not Used
def CortarMetadeInferior(frame):
    altura, largura, _ = frame.shape
    metade_inferior = frame[altura//2:altura, :]
    return metade_inferior

# Not Used
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


def CaptureNewFrame(CapturedVideo):
    # Definir a largura e altura desejadas do frame
    FrameWidth = 1024
    FrameHeight = 700
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
    # Rodar e Redimensionar o frame para a largura e altura desejadas
    frame = cv2.resize(frame, (FrameWidth, FrameHeight))    
    # endregion

    # region Preprocessing
    # Convert Frame RGB on Gray scale
    frameGray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # Applying the canny algorithm
    frameCanny = cv2.Canny(frameGray, 250, 255, None, 3)
    # endregion

    return frame, frameCanny


def DisplayFrame(textOnFrame):
    # Definir a largura e altura desejadas do frame
    FrameWidth = 1024
    FrameHeight = 700

    for i, (frame, text, value) in enumerate(textOnFrame):
        PrintOnFrame(frame, f"{text}", value, (50, 50 + (35 * i)))

    # Rodar e Redimensionar o frame para a largura e altura desejadas
    frame = cv2.resize(frame, (FrameWidth, FrameHeight))
    cv2.imshow('frame', frame)


def DetectCorners(frameCanny):
    # region Detetor de cantos de Shi-Tomasi
    cornersDetected = []
    # region Parâmetros do detector de cantos Shi-Tomasi
    ShiTomasiParams = dict(maxCorners=100,
                           qualityLevel=0.1,
                           minDistance=50,
                           blockSize=7)
    # endregion

    cornersDetected = cv2.goodFeaturesToTrack(
        frameCanny, mask=None, **ShiTomasiParams, useHarrisDetector=True, k=0.04)
    return cornersDetected
# endregion


def TrackFeatures(prevFrameGray, FrameGray, prevCorners):
    # region Parameters for lucas kanade optical flow
    LucasKanadeParams = dict(winSize=(25, 25),
                             maxLevel=2,
                             criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.01))
    # endregion
    # Create some random colors
    color = np.random.randint(0, 255, (100, 3))

    # Calcular o fluxo ótico usando o método Lucas-Kanade
    # corners, status, _ = cv2.calcOpticalFlowPyrLK(prevFrameGray, FrameGray, prevCorners, None, **LucasKanadeParams)

    p0 = prevCorners.reshape(-1, 1, 2)
    corners, status, _err = cv2.calcOpticalFlowPyrLK(
        prevFrameGray, FrameGray, p0, None, **LucasKanadeParams)
    p0r, status, _err = cv2.calcOpticalFlowPyrLK(
        FrameGray, prevFrameGray, corners, None, **LucasKanadeParams)
    d = abs(p0-p0r).reshape(-1, 2).max(-1)
    good = d < 1

    new_trajectories = []
    # Manter apenas os cantos que foram seguidos com sucesso
    # for trajectory, (x, y), good_flag in zip(goodCorners, corners.reshape(-1, 2), good):
    #     if not good_flag:
    #             continue
    #     trajectory.append((x, y))
    #     if len(trajectory) > 40:
    #         del trajectory[0]
    #     new_trajectories.append(trajectory)

    goodCorners = corners[status[:, 0] == 1]
    goodOldCorners = prevCorners[status[:, 0] == 1]

    if corners is None:
        return None

    dataLogger.info(f'\n goodOldCorners \n {goodOldCorners}')
    dataLogger.info(f'\n goodCorners \n {goodCorners}')
    return corners, goodCorners, goodOldCorners


def MatrizFundamental(goodCorners, goodOldCorners):
    fundamentalMatrix, mask = cv2.findFundamentalMat(
        goodCorners, goodOldCorners, cv2.FM_RANSAC)

    # Refinar a matriz fundamental
    goodCorners = goodCorners[mask.ravel() == 1]
    goodOldCorners = goodOldCorners[mask.ravel() == 1]
    fundamentalMatrix, _ = cv2.findFundamentalMat(
        goodCorners, goodOldCorners, cv2.FM_8POINT)
    fundamentalMatrix = cv2.correctMatches(
        fundamentalMatrix, goodCorners.T, goodOldCorners.T)

    dataLogger.info(f'\n fundamentalMatrix \n {fundamentalMatrix}')
    return fundamentalMatrix


def MatrizEssencial(goodCorners, goodOldCorners, intrinsicParameters):
    matrizEssencial, mask = cv2.findEssentialMat(
        goodCorners, goodOldCorners, intrinsicParameters)
    essencialMatrixRotation, essencialMatrixTranslation = DecomporMatrizEssencial(
        matrizEssencial, goodCorners, goodOldCorners, intrinsicParameters)

    dataLogger.info(f'\n matrizEssencial \n {matrizEssencial}')
    dataLogger.info(f'\n rotation \n {essencialMatrixRotation}')
    dataLogger.info(
        f'\n essencialMatrixTranslation \n {essencialMatrixTranslation}')

    return essencialMatrixRotation, essencialMatrixTranslation


def DecomporMatrizEssencial(essentialMatrix, goodCorners, goodOldCorners,  cameraMatrix):
    # Recupera as matrizes de rotação e translação da matriz essencial
    _, breakDownRotation, breakDownTranslation, _ = cv2.recoverPose(
        essentialMatrix, goodCorners, goodOldCorners, cameraMatrix)

    dataLogger.info(f'\n breakDownRotation \n {breakDownRotation}')
    dataLogger.info(f'\n breakDownTranslation \n {breakDownTranslation}')
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
    plotMotion3D = plotsFinish[1, 1]
    # Plot da cameraPosition
    plotTrajectory.set_xlabel('X')
    plotTrajectory.set_ylabel('Y')
    plotTrajectory.axis('equal')
    plotTrajectory.set_title('Plot Camera Position')
    plotTrajectory.plot(cameraPosition[0], cameraPosition[1], 'ro')

    # Plot da cameraMotion

    # Adiciona o subplot 3D em plots[1, 2]
    plotMotion3D = fig.add_subplot(2, 2, 4, projection='3d')
    plotMotion3D.quiver(
        0, 0, 0, cameraMotion[0, 3], cameraMotion[1, 3], cameraMotion[2, 3])
    plotMotion3D.set_xlim([-1, 1])
    plotMotion3D.set_ylim([-1, 1])
    plotMotion3D.set_zlim([-1, 1])

    plotMotion3D.set_xlabel('X')
    plotMotion3D.set_ylabel('Y')
    plotMotion3D.set_zlabel('Z')
    plotMotion3D.set_title('Plot Camera Motion')

    plt.tight_layout()
    plt.show()
    plt.pause(0.01)

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
    plt.pause(0.001)
    for ax in axsFrames:
        ax.cla()


def LoadCalibrationCamara():
    valores = ReadFilesTxt('CalibrationCam/cameramatrix.txt')
    # Converta a lista de valores em uma matriz NumPy
    intrinsicParameters = np.array(valores).reshape((3, 3))

    valores = ReadFilesTxt('CalibrationCam/distortioncoefficient.txt')
    distortioncoefficient = np.array(valores).reshape((1, 5))

    # projectionMatrix = np.reshape(params, (3, 4))
    return intrinsicParameters, distortioncoefficient


def Init():

    #     # region save images
    #     # Flag para guardar as imagens, se false apenas mostra a imagem final
    #     if SaveImage is True:
    #         DataTimeNow = datetime.now()
    #         OutputFolder = f'2 - OutputOdometria/{DataTimeNow.strftime("%d.%m.%Y_%H.%M.%S")}/'
    #         os.makedirs(OutputFolder, exist_ok=True)
    #     # endregion

    #     fig, plotsFinish = plt.subplots(2, 2)
    #     # Remove o subplot vazio em plots[1, 2]
    #     fig.delaxes(plotsFinish[1, 1])

    #     fig2, plots2 = plt.subplots(2, 2, dpi=300)

    #     # Configuração básica do log
    #     logging.basicConfig(filename=f'dataLogger_{DataTimeNow.strftime("%d.%m.%Y_%H.%M.%S")}.txt', level=logging.DEBUG)
    #     # Criando um objeto de log
    #     dataLogger = logging.getLogger('dataLogger')

    #     return DataTimeNow, OutputFolder, plotsFinish, dataLogger
    return 1


def main():
    # DataTimeNow, OutputFolder, plotsFinish, dataLogger = Init()
    fps = 0
    fpsMedia = 0
    cornersMedia = 0
    CountFrames = 0
    CountMedia = 0
    ResetCorners = 25
    textOnFrame = [[]]
    Corners = [[]]
    # Data logger

    prevCameraMotion = np.zeros((4, 3))
    prevCameraMotion[:3, :3] = np.eye(3)
    prevCameraPosition = np.zeros(3)

    intrinsicParameters, distortioncoefficient = LoadCalibrationCamara()

    CapturedVideo = LoadVideo()
    firstFrame, prevFrameCanny = CaptureNewFrame(CapturedVideo)
    prevCorners = DetectCorners(prevFrameCanny)
    mask = np.zeros_like(firstFrame)

    while True:
        CountMedia += 1
        dataLogger.warning(f'\n Iteration: {CountMedia}')
        # start time to calculate FPS
        start = time.time()
        frame, frameCanny = CaptureNewFrame(CapturedVideo)
        if len(frame) == 0:
            break
        CountFrames = CountFrames + 1

        Corners, goodCorners, goodOldCorners = TrackFeatures(
            prevFrameCanny, frameCanny, prevCorners)

        if Corners is None or len(Corners) < 10 or CountFrames >= ResetCorners:
            CountFrames = 0
            prevCorners = DetectCorners(prevFrameCanny)

            while prevCorners is None:
                frame, frameCanny = CaptureNewFrame(CapturedVideo)
                prevCorners = DetectCorners(frameCanny)
            Corners, goodCorners, goodOldCorners = TrackFeatures(
                prevFrameCanny, frameCanny, prevCorners)
            mask = np.zeros_like(frame)

        frameMask = DrawTracks(frame, mask, goodCorners,
                               goodOldCorners, color=None)

        # Obter a matriz Fundamental
        # matrizFundamental = MatrizFundamental(goodCorners, goodOldCorners)

        # Obter a matriz Essencial
        relativeRotation, relativeTranslation = MatrizEssencial(
            goodCorners, goodOldCorners, intrinsicParameters)

        # Output Finish
        if(CountMedia > 200):
            cameraMotion, cameraPosition = UpdateCameraMotion(
                relativeRotation, relativeTranslation, prevCameraMotion, prevCameraPosition)

        if prevCorners is not None:
            textOnFrame.append([frameMask, "fps", round(fps, 2)])
            textOnFrame.append([frameMask, "Corners", len(Corners)])
            textOnFrame.append(
                [frameMask, f"CountFrames {ResetCorners}", CountFrames])
            DisplayFrame(textOnFrame)

        # prevCameraMotion = cameraMotion
        # prevCameraPosition = cameraPosition
        prevFrameCanny = frameCanny.copy()
        prevCorners = goodCorners

        # Data logger
        dataLogger.info(f'\n prevCorners \n {prevCorners}')
        dataLogger.info(f'\n Corners \n {Corners}')

        # End time
        end = time.time()
        # calculate the FPS for current frame detection
        fps = 1 / (end-start)
        fpsMedia = fpsMedia + fps
        cornersMedia += len(Corners)
        textOnFrame = []
        cv2.waitKey(5)
    fpsMedia = fpsMedia / CountMedia
    cornersMedia = cornersMedia / CountMedia
    print(f"\nFrames por segundo: {fpsMedia}\n")
    print(f"Corners media: {cornersMedia}\n")


if __name__ == '__main__':
    main()
