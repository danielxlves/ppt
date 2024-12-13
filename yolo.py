from ultralytics import YOLO
import cv2

# Carrega o modelo YOLOv8 pré-treinado
model = YOLO('yolov8x.pt')  # Substitua pelo caminho para o arquivo do modelo

# Captura de vídeo da webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Realiza a detecção de objetos
    results = model(frame)

    # Exibe as caixas delimitadoras e as classes dos objetos detectados
    results[0].show()  # Acessa o primeiro item da lista e chama o método show

    # Pressiona 'q' para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
