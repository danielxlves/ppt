import cv2
import numpy as np

# Captura de vídeo da webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Converte a imagem em um array NumPy
    frame_array = np.array(frame)

    # Aplica uma operação de transformação: inverter as cores (negativo)
    inverted_frame = 255 - frame_array

    # Exibe a imagem invertida
    cv2.imshow('NumPy Image Manipulation', inverted_frame)

    # Pressiona 'q' para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
