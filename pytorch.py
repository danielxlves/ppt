import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import cv2
import numpy as np

# Carrega um modelo pré-treinado
model = models.resnet18(pretrained=True)
model.eval()

# Função para transformar a imagem para o formato correto
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Captura de vídeo da webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Converte a imagem em um formato que o modelo pode processar
    input_tensor = transform(frame)
    input_batch = input_tensor.unsqueeze(0)

    # Faz a previsão
    with torch.no_grad():
        output = model(input_batch)
    _, predicted_class = torch.max(output, 1)

    # Exibe a previsão na tela
    label = str(predicted_class.item())
    cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Exibe o resultado
    cv2.imshow('PyTorch Image Classification', frame)

    # Pressiona 'q' para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
