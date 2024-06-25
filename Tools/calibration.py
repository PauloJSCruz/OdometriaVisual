import numpy as np
import cv2
import glob

# Definições
squareSize = 2.5  # Tamanho do quadrado no tabuleiro em centimetros
patternSize = (9, 6)  # Número de cantos internos no tabuleiro (colunas, linhas)

# Preparar pontos de objeto
objectPoints = np.zeros((np.prod(patternSize), 3), np.float32)
objectPoints[:, :2] = np.indices(patternSize).T.reshape(-1, 2)
objectPoints *= squareSize

# Arrays para armazenar pontos de objeto e pontos de imagem de todas as imagens.
pointsOnRealWorld = []  # pontos 3d no espaço real
pointsOnCameraPlan = []  # pontos 2d no plano da imagem.

# Leitura de imagens
imagesList = glob.glob('FotosCalibracao/*.png')  # Substitua pelo caminho correto para suas imagens

for fname in imagesList:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Encontrar os cantos do tabuleiro
    ret, corners = cv2.findChessboardCorners(gray, patternSize, None)

    # Se encontrados, adicionar pontos de objeto, pontos de imagem
    if ret:
        pointsOnRealWorld.append(objectPoints)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        pointsOnCameraPlan.append(corners2)

        # Desenhar e exibir os cantos
        img = cv2.drawChessboardCorners(img, patternSize, corners2, ret)
        cv2.imshow('Img', img)
        cv2.waitKey(500)

cv2.destroyAllWindows()

# Calibração da câmera
calibrationError, cameraMatrix, distCoeffs, rotationVectors, translationVectors = cv2.calibrateCamera(pointsOnRealWorld, pointsOnCameraPlan, gray.shape[::-1], None, None)

# Exibindo os parâmetros da câmera
print("Matriz da Câmera:", cameraMatrix)
print("Coeficientes de distorção:", distCoeffs)
print("Vetores de rotação:", rotationVectors)
print("Vetores de translação:", translationVectors)
print("Error de Calibração:", calibrationError)

# Calculando erro médio de reprojeção
meanError = 0
totalPoints = 0
for i in range(len(pointsOnRealWorld)):
    imagePoints2, _ = cv2.projectPoints(pointsOnRealWorld[i], rotationVectors[i], translationVectors[i], cameraMatrix, distCoeffs)
    error = cv2.norm(pointsOnCameraPlan[i], imagePoints2, cv2.NORM_L2) / len(imagePoints2)
    totalPoints += len(pointsOnCameraPlan[i])
    meanError += error
meanError /= len(pointsOnRealWorld)
print("Erro médio de reprojeção: ", meanError)

# Salvando os parâmetros da câmera
np.savez('calibration_data', mtx=cameraMatrix, dist=distCoeffs, rvecs=rotationVectors, tvecs=translationVectors)

# Função para desfazer distorção
def undistortImage(imagePath, saved_params='calibration_data.npz'):
    # Carregar os dados salvos
    with np.load(saved_params) as X:
        cameraMatrix, distCoeffs = X['mtx'], X['dist']
    img = cv2.imread(imagePath)
    original_img = img.copy()  # Copia da imagem original para comparação
    h, w = img.shape[:2]
    newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, (w, h), 1, (w, h))
    
    # Desfazer distorção
    dst = cv2.undistort(img, cameraMatrix, distCoeffs, None, newCameraMatrix)
    
    # Recortar a imagem
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    cv2.imshow('Original Image', original_img)
    cv2.imshow('Calibrated Image', dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Teste a função com uma imagem
undistortImage('FotosCalibracao/image_0022.png')  # Substitua pelo caminho correto para uma imagem de teste
