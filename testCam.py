"""import cv2
cap= cv2.VideoCapture(0)
while True:
 ret, frame= cap.read()
 if not ret:
  print("Algo esta fallando")
 new_color= cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
 cv2.imshow("View", new_color)
 if cv2.waitKey(1) & 0xFF == ord("q"):
    break"""

"""import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)

with mp.solutions.hands.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("No se puede leer el video")
            break

        print("load!...") 
        # Convierte la imagen a RGB
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

        # Para procesar la imagen, envía una imagen RGB a MediaPipe Hands.
        results = hands.process(image)

        # Dibuja los puntos clave de las manos en la imagen.
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    image, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

        # Muestra la imagen con los puntos clave dibujados.
        cv2.imshow('MediaPipe Hands', image)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()"""
"""import cv2

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) > 10000:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

    cv2.imshow("frame", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
"""
"""import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)

with mp.solutions.hands.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        cv2.imshow("f",image)
        if not success:
            print("No se puede leer el video")
            break

        print("load!...") 
        # Convierte la imagen a RGB
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

        # Para procesar la imagen, envía una imagen RGB a MediaPipe Hands.
        results = hands.process(image)

        # Dibuja los puntos clave de las manos en la imagen.
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imshow("juego", image)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        # Define las reglas del juego
        rules = {
            "rock": {"scissors": "win", "paper": "lose"},
            "paper": {"rock": "win", "scissors": "lose"},
            "scissors": {"paper": "win", "rock": "lose"}
        }
        
        # Define las posiciones de los dedos que representan cada opción
        options = {
            "rock": [1],
            "paper": [0],
            "scissors": [8]
        }
        
        # Detecta los puntos clave de las manos
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                fingers = []
                for finger in hand_landmarks.landmark[1:]:
                    if finger.y < hand_landmarks.landmark[0].y:
                        fingers.append(1)
                    else:
                        fingers.append(0)
                
                # Determina la elección del jugador
                player_choice = None
                for option in options:
                    if all([fingers[i] for i in options[option]]):
                        player_choice = option
                
                # Si se ha detectado una elección del jugador
                if player_choice is not None:
                    # Determina la elección del ordenador
                    computer_choice = random.choice(list(rules[player_choice].keys()))
                    
                    # Determina el resultado del juego
                    """
"""import cv2
import mediapipe as mp
import random

cap = cv2.VideoCapture(0)

with mp.solutions.hands.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("No se puede leer el video")
            break

        print("load!...") 
        # Convierte la imagen a RGB
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

        # Para procesar la imagen, envía una imagen RGB a MediaPipe Hands.
        results = hands.process(image)

        # Dibuja los puntos clave de las manos en la imagen.
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Define las reglas del juego
        rules = {
            "rock": {"scissors": "win", "paper": "lose"},
            "paper": {"rock": "win", "scissors": "lose"},
            "scissors": {"paper": "win", "rock": "lose"}
        }
        
        # Define las posiciones de los dedos que representan cada opción
        options = {
            "rock": [1],
            "paper": [0],
            "scissors": [8]
        }
        
        # Detecta los puntos clave de las manos
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                fingers = []
                for finger in hand_landmarks.landmark[1:]:
                    if finger.y < hand_landmarks.landmark[0].y:
                        fingers.append(1)
                    else:
                        fingers.append(0)
                
                # Determina la elección del jugador
                player_choice = None
                for option in options:
                    if all([fingers[i] for i in options[option]]):
                        print("4444")
                        print(option)
                        player_choice = option
                
                # Si se ha detectado una elección del jugador
                if player_choice is not None:
                    # Determina la elección del ordenador
                    computer_choice = random.choice(list(rules[player_choice].keys()))
                    
                    # Determina el resultado del juego
                    result = rules[player_choice][computer_choice]
                    
                    # Muestra la elección de cada jugador y el resultado del juego
                    print(f"Jugador: {player_choice}")
                    print(f"Ordenador: {computer_choice}")
                    if result == "win":
                        print("¡Ganaste!")
                    elif result == "lose":
                        print("Perdiste.")
                    else:
                        print("Empate.")
                    
                    # Reinicia el juego
                    print("Reiniciando...")
                    break

        # Muestra la imagen con los puntos clave dibujados.
        cv2.imshow('MediaPipe Hands', image)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
"""
import cv2
import mediapipe as mp
import time
import random
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
# Inicializar variables
player_score = 0
computer_score = 0
tie_score = 0
# Inicializar detector de manos
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)
# Inicializar captura de video
cap = cv2.VideoCapture(0)
while True:
    # Leer imagen desde la cámara
    ret, image = cap.read()
    # Convertir imagen a RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Detectar manos en la imagen
    results = hands.process(image)
    # Dibujar puntos en las manos detectadas
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            # Obtener posiciones de los dedos
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
           ####probar que datos da cuando eligo un juego de manos y asi poner en las condicionales
            # Determinar qué gesto se está realizando (tijeras, papel o roca)
            if index_tip.y < middle_tip.y and thumb_tip.x < index_tip.x:
                player_gesture = 'scissors'
            elif index_tip.y > middle_tip.y and thumb_tip.x > index_tip.x:
                player_gesture = 'paper'
            #else:
            elif index_tip.y < middle_tip.y and thumb_tip.x < index_tip.x: 
                player_gesture = 'rock'
            # Generar gesto aleatorio para la computadora
            computer_gesture = random.choice(['rock', 'paper', 'scissors'])
            # Determinar ganador de la ronda
            if player_gesture == computer_gesture:
                tie_score += 1
                winner_text = 'Tie!'
            elif (player_gesture == 'rock' and computer_gesture == 'scissors' or 
                  player_gesture == 'paper' and computer_gesture == 'rock' or 
                  player_gesture == 'scissors' and computer_gesture == 'paper'):
                player_score += 1
                winner_text = 'Player wins!'
            else:
                computer_score += 1
                winner_text = 'Computer wins!'
            print("user: ")
            print(player_gesture)
            print("computer: ")
            print(computer_gesture)
            # Mostrar resultado de la ronda en pantalla
            cv2.putText(image, f'Player: {player_score}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(image, f'Computer: {computer_score}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(image, f'Ties: {tie_score}', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(image, winner_text, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    # Mostrar imagen en pantalla
    cv2.imshow('Rock Paper Scissors', image)
    # Esperar tecla ESC para salir del programa
    if cv2.waitKey(1) == 27:
        break
    # Esperar antes de comenzar una nueva ronda
    time.sleep(3)
# Liberar recursos utilizados por OpenCV y MediaPipe
cap.release()
cv2.destroyAllWindows()
hands.close()
