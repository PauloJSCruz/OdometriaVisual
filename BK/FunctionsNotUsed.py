
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

# Configura��o b�sica do log
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
    # region Definir os par�metros para o algoritmo RANSAC
    numOfCoordinatesMin = 10  # N�mero m�nimo de pontos para ajustar o modelo
    k = 100  # N�mero de itera��es do RANSAC
    distanceToConsiderInlier = 25  # Limiar para considerar um ponto como inlier
    minInlinersForAccept = 10  # N�mero m�nimo de inliers para aceitar o modelo
    # endregion

    # Criar uma inst�ncia do modelo linear
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
    # Fun��o para desenhar outliers e inliers no quadro

    for point in data:
        # Verificar se o ponto � um outlier ou inlier
        if model.distance(point) > distanceToConsiderInlier:
            color = (0, 0, 255)  # Vermelho (outlier)
        else:
            color = (0, 255, 0)  # Verde (inlier)
        # Converter as coordenadas do ponto para inteiros
        center = (int(point[0]), int(point[1]))
        # Desenhar a bola no quadro
        cv2.circle(frame, center, 5, color, -1)


def ReadFilesTxt(file):
    with open(file, 'r') as f:
        linhas = f.readlines()
        # Inicialize uma lista vazia para armazenar os valores da matriz
        valores = []
        # Percorra cada linha
        for linha in linhas:
            # Remova os caracteres de espa�o em branco e os colchetes
            linha = linha.strip().strip('[]')

            # Divida a linha em elementos separados por espa�o em branco
            elementos = linha.split()

            # Converta cada elemento para float e adicione-o � lista de valores
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
            # Gerar uma cor RGB aleat�ria
            color = tuple(np.random.randint(0, 255, 3).tolist())
            # Desenhar a linha com a cor aleat�ria no quadro combinado
            cv2.line(combined_frame, prev_pt, (current_pt[0] + prevFrame.shape[1], current_pt[1]), color, 1)
            # Desenhar c�rculos sem preenchimento nas extremidades da linha
            cv2.circle(combined_frame, prev_pt, 5, color, -1)
            cv2.circle(combined_frame, (current_pt[0] + prevFrame.shape[1], current_pt[1]), 5, color, -1)

            # Rodar e Redimensionar o frame para a largura e altura desejadas
            combined_frame = cv2.resize(combined_frame, (FrameWidth, FrameHeight))
            cv2.imshow('Combined Frames', combined_frame)
    # 
# endregion

  

# Print Output
def UpdateCameraMotion(R, T, prevCameraMotion, prevCameraPosition):
    # Atualiza a estimativa de movimento da c�mera
    cameraMotion = np.dot(prevCameraMotion, np.hstack((R, T)))

    # Atualiza a posi��o estimada da c�mera no mundo
    cameraPosition = prevCameraPosition + np.dot(prevCameraMotion[:, :3], T)

    # Reduz o espa�amento horizontal entre os subplots
    # plt.subplots_adjust(wspace=0.01)
    # Reduz o espa�amento vertical entre os subplots
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
    # Reduz o espa�amento horizontal entre os subplots
    plt.subplots_adjust(wspace=0.01)
    # Reduz o espa�amento vertical entre os subplots
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