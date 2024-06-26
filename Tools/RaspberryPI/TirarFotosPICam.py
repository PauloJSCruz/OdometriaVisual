import cv2
import os
import time
from picamera2 import Picamera2

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
    picam2 = Picamera2()

    # Configurar a camera
    camera_config = picam2.create_still_configuration(main={"size": (3280, 2464)})
    picam2.configure(camera_config)
        
    picam2.set_controls({
                "Brightness": 0.0,  # Brilho: -1 (escuro) a 1 (claro). Valor padr�o: 0.5.
                "Contrast": 1.5,  # Contraste: 0 (baixo contraste) a 32.0 (alto contraste). Valor padr�o: 1.0.
                "ExposureTime": 20000,  # Tempo de exposi��o: em microsegundos. Valor padr�o: 20000.
                "AnalogueGain": 1.8,  # Ganho anal�gico: normalmente de 1.0 a 16.0. Valor padr�o: 2.0.
                "AwbMode": 0,  # Modo de balan�o de branco autom�tico: 0 (off), 1 (auto), 2 (tungsten), 3 (fluorescent), 4 (indoor), 5 (daylight), 6 (cloudy)
                "Saturation": 0.1,  # Satura��o: 0 (dessaturado) a 1 (muito saturado). Valor padr�o: 1.0.
                "Sharpness": 2,  # Nitidez: 0 (baixa nitidez) a 1 (alta nitidez). Valor padr�o: 1.0.
                "NoiseReductionMode": 2,  # Modo de redu��o de ru�do: 0 (off), 1 (fast), 2 (high_quality)
            })

    picam2.start() 
    
    # Contador para as imagens
    count = 0
    print("Pré-visualização da captura de imagens...")
    preview_time = time.time()
    config = 0.0
    while True:
        image = picam2.capture_array()
        # Mostrar a imagem capturada
        cv2.imshow('Camera Feed', image)  
        cv2.waitKey(1)
        # Verificar se o tempo de pré-visualização passou antes de começar a salvar as imagens~
        realTime = time.time()
        if  realTime > preview_time + delay:
            
            preview_time = realTime
            if count < num_images:
                img_name = f"image_{count:04d}.png"
                img_path = os.path.join(save_path, img_name)
                cv2.imwrite(img_path, image)
                print(f"Imagem {count + 1} salva em: {img_path}")
                count += 1
            else:
                break

            delay = 5

        # # Pressione 'q' para sair
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Limpeza
    cv2.destroyAllWindows()
    print("Captura de imagens concluída.")

if __name__ == "__main__":
    save_path = 'FotosPICamera'
    num_images = 50
    delay = 50  # tempo de visualização em segundos antes de começar a capturar
    capture_images(save_path, num_images, delay)
