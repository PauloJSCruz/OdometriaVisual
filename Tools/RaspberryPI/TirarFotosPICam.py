import cv2
import os
import time
from picamera2 import Picamera2

def capture_images(save_path, num_images=20):
    """
    Captura uma serie de imagens da camera conectada ao computador ao pressionar a tecla "Enter" e salva em um diretorio especificado.
    
    Parametros:
        save_path (str): O caminho onde as imagens serao salvas.
        num_images (int): Numero de imagens a serem capturadas.
    """
    # Criar o diretorio se nao existir
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # Inicializar a camera
    picam2 = Picamera2()

    # Configurar a camera
    camera_config = picam2.create_still_configuration(main={"size": (3280, 2464)})
    picam2.configure(camera_config)
        
    picam2.set_controls({
                "Brightness": 0.0,  # Brilho: -1 (escuro) a 1 (claro). Valor padrao: 0.5.
                "Contrast": 1.5,  # Contraste: 0 (baixo contraste) a 32.0 (alto contraste). Valor padrao: 1.0.
                "ExposureTime": 20000,  # Tempo de exposicao: em microsegundos. Valor padrao: 20000.
                "AnalogueGain": 1.8,  # Ganha analogico: normalmente de 1.0 a 16.0. Valor padrao: 2.0.
                "AwbMode": 0,  # Modo de balanco de branco automatico: 0 (off), 1 (auto), 2 (tungstenio), 3 (fluorescente), 4 (interno), 5 (luz do dia), 6 (nublado)
                "Saturation": 0.1,  # Saturacao: 0 (dessaturado) a 1 (muito saturado). Valor padrao: 1.0.
                "Sharpness": 2,  # Nitidez: 0 (baixa nitidez) a 1 (alta nitidez). Valor padrao: 1.0.
                "NoiseReductionMode": 2,  # Modo de reducao de ruido: 0 (off), 1 (rapido), 2 (alta qualidade)
            })

    picam2.start() 
    
    # Contador para as imagens
    count = 0
    print("Pressione 'Enter' para capturar uma imagem. Pressione 'q' para sair.")
    
    while count < num_images:
        image = picam2.capture_array()
        
        # Mostrar a imagem capturada de forma eficiente
        cv2.imshow('Camera Feed', cv2.resize(image, (640, 480)))  
        
        key = cv2.waitKey(1) & 0xFF
        if key == 13:  # Codigo da tecla "Enter"
            img_name = f"image_{count:04d}.png"
            img_path = os.path.join(save_path, img_name)
            cv2.imwrite(img_path, image)
            print(f"Imagem {count + 1} salva em: {img_path}")
            count += 1
            
            # Libera a mem�ria da imagem ap�s salv�-la
            del image
        elif key == ord('q'):  # Pressione 'q' para sair
            break
        
        # Adicionar uma pequena pausa para reduzir a carga de processamento
        time.sleep(0.1)
    
    # Limpeza
    cv2.destroyAllWindows()
    picam2.stop()
    print("Captura de imagens concluida.")

if __name__ == "__main__":
    save_path = 'FotosPICamera'
    num_images = 200
    capture_images(save_path, num_images)
