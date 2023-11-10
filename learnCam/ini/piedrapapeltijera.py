import mediapipe as mp
import random
import time
import cv2

mp_drawing= mp.solutions.drawing_utils
mp_hands= mp.solutions.hands
player_score=0
computer_score=0
tie_score=0
hands= mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)
cap= cv2.VideoCapture(0)
while True:
    ret, frame= cap.read()
    image= cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result= hands.process(image)
    cv2.imshow("View", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    if result.multi_hand_landmarks:
        for multi_hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, multi_hand_landmarks, mp_hands.HAND_CONNECTIONS)
            ###Debo verificar contador para empezar y tambien para iniciar juego o el iniciador del contador
            cv2.imshow("View", image)
            thumb_tip= multi_hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip= multi_hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_tip= multi_hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            if index_tip.y < middle_tip.y and thumb_tip.x < index_tip.x:
                player_gesture= 'scissors'
                print("scissors")
            elif index_tip.y > middle_tip.y and thumb_tip.x > index_tip.x:
                player_gesture= 'paper'
                print("paper")
            ##else:        
            elif index_tip.y < middle_tip.y and thumb_tip.x > index_tip.x:
                player_gesture= 'rock'
                print("rock")    
            else: 
                print("no reconoce")    
                break
            computer_gesture= random.choice(['paper','scissors','rock'])
            if player_gesture == computer_gesture:
                tie_score+=1
                winner_text= 'Tie!'
                ##figura_elegida= player_gesture
            elif (player_gesture == 'rock' and computer_gesture == 'scissors' or
                  player_gesture == 'paper' and computer_gesture == 'rock' or
                  player_gesture == 'scissors' and computer_gesture == 'paper'):
                player_score+=1
                winner_text= 'Player wins!'    
            else:
                computer_score+=1
                winner_text= 'Computer wins!'  
                ##figura_elegida= computer_gesture
            figura_elegida= player_gesture       
            cv2.putText(image, f'Player: {player_score}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)      
            cv2.putText(image, f'Computer: {computer_score}', (10, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 2)
            cv2.putText(image, f'Ties: {tie_score}', (10, 110), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 2)
            cv2.putText(image, winner_text, (10, 150), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 2)
            cv2.putText(image, f'Figura elegida: {figura_elegida}', (190, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 2)
            cv2.imshow("View game", image)        
cap.release()
cv2.destroyAllWindows()
hands.close()

