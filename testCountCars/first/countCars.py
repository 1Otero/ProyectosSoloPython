#C:\Users\truji\Downloads\videoCars.mp4
import cv2
import numpy as np
##import imutils
cap= cv2.VideoCapture("C:/Users/truji/Downloads/videoCars.mp4")
###substraer fondo O crear fondo -> cv2.createBackgroundSubtractorMOG() este es mejor que -> cv2.createBackgroundSubtractorMOG2()  
fgbg= cv2.bgsegm.createBackgroundSubtractorMOG()
##fgbg= cv2.createBackgroundSubtractorMOG2()
###Kernel mejorar la imagen binaria de la substracion de fondo
kernel= cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
car_counter=0
while True:
   if cap.isOpened():
    ret, frame= cap.read()
    if not ret:
     print("Error capturando fotogramas")
    frame= cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    frame= cv2.resize(frame, (840, 640))
    #especifica los puntos extremos del area a analizar
    ##area_pts= np.array([[330, 216], [frame.shape[1]-80, 216], [frame.shape[1]-80, 271], [330, 271]])
    #x(x superior izq), x(y superior de cuadro izq). x(x inferior der)-80, x(y superior de cuadro der). x(x superior der)-80, x(y inferior de cuadro der). x(x inferior izq), x(y inferior de cuadro izq)
    area_pts= np.array([[230, 370], [frame.shape[1]-80, 370], [frame.shape[1]-80, 575], [230, 575]])
    ###Esto pasa a blanco y negro la imagen
    imAux= np.zeros(shape=(frame.shape[:2]), dtype= np.uint8)
    ###Pinta contorno del area_pts con 255 gris y sin margen - fuera del cuadro todo es negro
    imAux= cv2.drawContours(imAux, [area_pts], -1, (255), -1)
    ###Hace que el frame se coloque entre lo visible dentro el cuadro y no visible 
    ###Mask hace que el recuadro elegido en gris anteriormente se vea el video o imagen
    image_area= cv2.bitwise_and(frame, frame, mask=imAux)
    ### hace que se vea en blanco y negro y contronee movimiento de lo blanco
    fgmask= fgbg.apply(image_area)
    ###Hace que los puntos blancos grandes se contorneen mejor
    fgmask= cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    fgmask= cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)
    fgmask= cv2.dilate(fgmask, None, iterations=10)
    ##cnts= cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    cnts= cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    for cnt in cnts:
      ##if cv2.contourArea(cnt) > 1500:
       if cv2.contourArea(cnt) > 800:
        x, y, w, h= cv2.boundingRect(cnt)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,255), 2)
        print(x + y)
        print(x + h)
        ##if 820 < (x + y) < 840:
        if 818 < (x + y) < 840:
          print("paso")
          cv2.line(frame, (510, 370), (510, 575), (0, 255, 0), 4)    
          car_counter= car_counter + 1
    cv2.drawContours(frame, [area_pts], -1, (255, 0, 255), 2)
    ##cv2.line(frame, (450, 216), (450, 271), (0, 255, 255), 1)
    #x(posicion x superior), x(y posicio superior). x(x inferior), x(y inferior)
    cv2.line(frame, (510, 370), (510, 575), (255, 255, 255), 1)
    cv2.rectangle(frame, (frame.shape[1]-80, 370), (frame.shape[1]-4, 575), (0,255,0), 2)
    cv2.putText(frame, 'Car', (frame.shape[1]-78, 450), cv2.FONT_HERSHEY_COMPLEX, 1.2, (0, 255, 0), 2)
    cv2.putText(frame, str(car_counter), (frame.shape[1]-55, 500), cv2.FONT_HERSHEY_COMPLEX, 1.2, (0,255,0), 2)
    cv2.imshow("View", frame)    
    ###cv2.imshow("ViewUno", fgmask)
    if cv2.waitKey(70) & 0xFF == ord("q"):
      break
    
cap.release()
cv2.destroyAllWindows()
