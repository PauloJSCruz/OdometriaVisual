# region Imports
import cv2
import numpy as np
import math
import matplotlib.pylab as plt
import time
from datetime import datetime
import os
import random
# endregion
SaveImage = False


# region plots
# region plots parameters
fig, axs = plt.subplots(1, 3, figsize=(24, 12), dpi=300)
# Reduz o espaçamento horizontal entre os subplots
plt.subplots_adjust(wspace=0.01)
# Reduz o espaçamento vertical entre os subplots
plt.subplots_adjust(hspace=0.01)
FlagPlot = False
# endregion


def PlotFrames(frames):
    for i, data in enumerate(frames):
        frame, title = data
        axs[i].imshow(frame, cmap='gray' if len(frame.shape) == 2 else None)
        axs[i].set_title(title)
        axs[i].axis('off')
    plt.savefig(
        f'{OutputFolder}{DataTimeNow.strftime("%d.%m.%Y_%H.%M.%S")}-5Plots.png', bbox_inches='tight')
    # plt.show(block=False)
    # plt.pause(0.001)
    for ax in axs:
        ax.cla()
# endregion


# region save images
# Flag para guardar as imagens, se false apenas mostra a imagem final
if SaveImage is True:
    DataTimeNow = datetime.now()
    OutputFolder = f'2 - OutputOdometria/{DataTimeNow.strftime("%d.%m.%Y_%H.%M.%S")}/'
    os.makedirs(OutputFolder, exist_ok=True)
# endregion

# region Parâmetros do detector de cantos Shi-Tomasi
ShiTomasiParams = dict(maxCorners=100,
                       qualityLevel=0.1,
                       minDistance=50,
                       blockSize=7)
# endregion

# region Parameters for lucas kanade optical flow
LucasKanadeParams = dict(winSize=(25, 25),
                         maxLevel=2,
                         criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.01))
# endregion

# region Global variables

Frame = None
FrameGray = None
FrameCanny = None

mask = None

prevFrame = None
prevFrameGray = None
prevFrameCanny = None
prevCorners = []

# endregion


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


def CortarMetadeInferior(frame):
    altura, largura, _ = frame.shape
    metade_inferior = frame[altura//2:altura, :]
    return metade_inferior


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
        exit()

    # Frame = CortarMetadeInferior(Frame)

    # print(CapturedVideo.get(cv2.CAP_PROP_POS_FRAMES))

    # Rodar e Redimensionar o frame para a largura e altura desejadas
    # Frame = cv2.rotate(Frame, rotateCode = cv2.ROTATE_90_CLOCKWISE)
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
    sizeTextOnFrame = len(textOnFrame)
    for i, (frame, text, value) in enumerate(textOnFrame):
        PrintOnFrame(frame, f"{text}", value, (50, 50 + (35 * i)))
    return cv2.imshow('frame', frame)


def PrintOnFrame(Frame, texto, valor, pos):
    # Adicione texto ao frame
    texto = texto + ": " + str(valor)
    posicao = pos  # Posição do texto no frame
    fonte = cv2.FONT_HERSHEY_SIMPLEX  # Estilo da fonte
    escala = 1  # Escala do texto
    cor = (255, 255, 0)  # Cor do texto (verde no exemplo)
    espessura = 2  # Espessura da linha do texto

    cv2.putText(Frame, texto, posicao, fonte, escala, cor, espessura)


def RansacPrint(Corners):
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
            cv2.line(Frame, (int(x[i]), int(y[i])), (int(
                x[i+1]), int(y[i+1])), (0, 255, 0), 2)
        DrawOutliersAndInliers(
            Frame, data, bestModel, distanceToConsiderInlier)


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


def DetectCorners(frameCanny):
    global mask
    # region Detetor de cantos de Shi-Tomasi
    cornersDetected = []
    cornersDetected = cv2.goodFeaturesToTrack(
        frameCanny, mask=None, **ShiTomasiParams, useHarrisDetector=True, k=0.04)
    # endregion
    mask = np.zeros_like(prevFrame)
    return cornersDetected


def TrackFeatures(prevFrameGray, FrameGray, prevCorners):
    global LucasKanadeParams
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

    return corners, goodCorners, goodOldCorners


def MatrizFundamental(goodCorners, goodOldCorners):
    F, mask = cv2.findFundamentalMat(
        goodCorners, goodOldCorners, cv2.FM_RANSAC)

    # Refinar a matriz fundamental
    goodCorners = goodCorners[mask.ravel() == 1]
    goodOldCorners = goodOldCorners[mask.ravel() == 1]
    F, _ = cv2.findFundamentalMat(goodCorners, goodOldCorners, cv2.FM_8POINT)
    F = cv2.correctMatches(F, goodCorners.T, goodOldCorners.T)

    return F


def MatrizEssencial(goodCorners, goodOldCorners, intrinsicParameters):
    matrizEssencial, mask = cv2.findEssentialMat(
        goodCorners, goodOldCorners, intrinsicParameters,)
    rotation, translation = DecomporMatrizEssencial(
        matrizEssencial, goodOldCorners, goodCorners)
    return rotation, translation


def DecomporMatrizEssencial(matrizEssencial, goodOldCorners, goodCorners):
    rotation1, rotation2, translation = cv2.decomposeEssentialMat(
        matrizEssencial, goodOldCorners, goodCorners)
    return rotation1, rotation2, translation


def LoadCalibrationCamara():
    valores = ReadFilesTxt('cameramatrix.txt')
    # Converta a lista de valores em uma matriz NumPy
    intrinsicParameters = np.array(valores).reshape((3, 3))

    valores = ReadFilesTxt('distortioncoefficient.txt')
    distortioncoefficient = np.array(valores).reshape((1, 5))

    # projectionMatrix = np.reshape(params, (3, 4))
    return intrinsicParameters, distortioncoefficient


def main():
    CountFrames = 0
    ResetCorners = 25
    textOnFrame = []
    Corners = []
    fps = 0

    intrinsicParameters, distortioncoefficient = LoadCalibrationCamara()

    CapturedVideo = LoadVideo()
    firstFrame, prevFrameCanny = CaptureNewFrame(CapturedVideo)
    prevCorners = DetectCorners(prevFrameCanny)
    mask = np.zeros_like(firstFrame)

    while True:
        # start time to calculate FPS
        start = time.time()
        frame, frameCanny = CaptureNewFrame(CapturedVideo)
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
        matrizFundamental = MatrizFundamental(goodCorners, goodOldCorners)

        # Obter a matriz Essencial
        matrizEssencial = MatrizEssencial(
            goodCorners, goodOldCorners, intrinsicParameters)

        if prevCorners is not None:
            textOnFrame.append([frameMask, "fps", round(fps, 2)])
            textOnFrame.append([frameMask, "Corners", len(Corners)])
            textOnFrame.append(
                [frameMask, f"CountFrames {ResetCorners}", CountFrames])
            DisplayFrame(textOnFrame)

        prevFrameCanny = frameCanny.copy()
        prevCorners = goodCorners

        # region Output
        # if SaveImage is True:
        #     # Salvar o frame como imagem
        #     DataTimeNow = datetime.now()
        #     # cv2.imwrite(f'{OutputFolder}{CapturedVideo.get(cv2.CAP_PROP_POS_FRAMES)}-{now.strftime("%d.%m.%Y_%H.%M.%S")}.png' , Frame)
        #     # cv2.imwrite(f'{OutputFolder}{DataTimeNow.strftime("%d.%m.%Y_%H.%M.%S")}-1FrameGray.png' , FrameGray)
        #     # cv2.imwrite(f'{OutputFolder}{DataTimeNow.strftime("%d.%m.%Y_%H.%M.%S")}-2FrameCanny.png' , FrameCanny)
        #     # cv2.imwrite(f'{OutputFolder}{DataTimeNow.strftime("%d.%m.%Y_%H.%M.%S")}-3OutputFrame.png' , Frame)
        #     cv2.imwrite(f'{OutputFolder}{DataTimeNow.strftime("%d.%m.%Y_%H.%M.%S")}-4OutputFrame.png' , FrameMask)
        #     # PlotFrames([(FrameGray, "FrameGray"), (FrameCanny, "FrameCanny"), (FrameMask, "FrameMask")])

        # if cv2.waitKey(5) & 0xFF == ord('q'):
        #     break
        # endregion

        # End time
        end = time.time()
        # calculate the FPS for current frame detection
        fps = 1 / (end-start)

        textOnFrame = []
        cv2.waitKey(5)


if __name__ == '__main__':
    main()
