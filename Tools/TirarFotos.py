import cv2
import os
import time

def capture_images(save_path, num_images=20, delay=5):
    """
    Captura uma série de imagens da câmera conectada ao computador após uma pré-visualização de alguns segundos e salva em um diretório especificado.
    
    Parâmetros:
        save_path (str): O caminho onde as imagens serão salvas.
        num_images (int): Número de imagens a serem capturadas.
        preview_time (int): Duração da pré-visualização antes de começar a capturar (em segundos).
    """
    # Criar o diretório se não existir
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # Inicializar a câmera
    CapturedVideo = cv2.VideoCapture(0)
    CapturedVideo.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    if not CapturedVideo.isOpened():
        raise IOError("Não foi possível abrir a câmera")
    
    # Contador para as imagens
    count = 0
    print("Pré-visualização da captura de imagens...")
    preview_time = time.time()
    while True:
        ret, frame = CapturedVideo.read()
        if not ret:
            print("Falha ao capturar imagem; saindo...")
            break
        
        # Mostrar a imagem capturada
        cv2.imshow('Camera Feed', frame)
        
        # Verificar se o tempo de pré-visualização passou antes de começar a salvar as imagens~
        realTime = time.time()
        if  realTime > preview_time + delay:
            preview_time = realTime
            if count < num_images:
                img_name = f"image_{count:04d}.png"
                img_path = os.path.join(save_path, img_name)
                cv2.imwrite(img_path, frame)
                print(f"Imagem {count + 1} salva em: {img_path}")
                count += 1
            else:
                break

        # # Pressione 'q' para sair
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Limpeza
    CapturedVideo.release()
    cv2.destroyAllWindows()
    print("Captura de imagens concluída.")

if __name__ == "__main__":
    save_path = 'FotosCalibracao'
    num_images = 30
    delay = 5  # tempo de visualização em segundos antes de começar a capturar
    capture_images(save_path, num_images, delay)
