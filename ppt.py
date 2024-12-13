import cv2
import mediapipe as mp
import numpy as np

# Inicializa o Mediapipe para detecção de mãos e OpenCV para detecção de rostos
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)
mp_draw = mp.solutions.drawing_utils
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Função para detectar o gesto
def get_hand_gesture(landmarks):
    thumb = landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle = landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring = landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky = landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    
    # Definir gestos com base nas relações entre as posições dos dedos
    if thumb.x < index.x and thumb.x < middle.x:  # Mão fechada (pedra)
        return "Pedra"
    elif index.x > middle.x and index.x > ring.x and index.x > pinky.x:  # Mão aberta (papel)
        return "Papel"
    elif index.x < middle.x and middle.x < ring.x and ring.x < pinky.x:  # Mão com dois dedos estendidos (tesoura)
        return "Tesoura"
    return "Desconhecido"

# Função para determinar o vencedor
def determine_winner(player1_choice, player2_choice):
    if player1_choice == player2_choice:
        return "Empate"
    elif (player1_choice == "Pedra" and player2_choice == "Tesoura") or \
         (player1_choice == "Papel" and player2_choice == "Pedra") or \
         (player1_choice == "Tesoura" and player2_choice == "Papel"):
        return "Jogador 1 venceu!"
    else:
        return "Jogador 2 venceu!"

# Função para detectar rostos e identificar jogadores
def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    return faces

# Captura de vídeo da webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detectar rostos para identificar os jogadores
    faces = detect_faces(frame)

    # Converte a imagem para RGB para o Mediapipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    player_choices = []

    if results.multi_hand_landmarks:
        # Para cada mão detectada
        for i, landmarks in enumerate(results.multi_hand_landmarks):
            # Desenha os pontos das mãos na tela
            mp_draw.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

            # Identifica o gesto da mão
            hand_gesture = get_hand_gesture(landmarks)
            player_choices.append(hand_gesture)

        # Verifica se há duas mãos detectadas (duas pessoas jogando)
        if len(player_choices) == 2:
            player1_choice = player_choices[0]
            player2_choice = player_choices[1]

            print(f"Jogador 1 escolheu: {player1_choice}")
            print(f"Jogador 2 escolheu: {player2_choice}")

            # Determina o vencedor
            result = determine_winner(player1_choice, player2_choice)
            cv2.putText(frame, result, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Desenha retângulos ao redor dos rostos detectados (para indicar jogadores)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Exibe o vídeo com o resultado
    cv2.imshow('Pedra, Papel e Tesoura - Jogo de 2 Jogadores', frame)

    # Pressiona 'q' para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
