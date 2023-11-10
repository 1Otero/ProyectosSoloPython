from django.shortcuts import render
from django.http import request, HttpResponse
import cv2
import threading
dato= {"id": 1, "name": "uno"}
datos= [{"id": 1, "name": "uno"},{"id": 2, "name": "dos"}]
def read_cam():
    cap= cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error iniciando fotogramas")
    while True:
       try: 
        ret, frame= cap.read()   
        if not ret:
           print("Error capturando fotogramas")
        cv2.imshow("View", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
       except Exception as e:
           print("Error camera and thread")
           print(e)
           break
    cv2.destroyAllWindows()    
    cap.release()
# Create your views here.
def client(request):
    t= threading.Thread(target= read_cam)
    t.start() 
    ###return HttpResponse([1, 7, (3, 3), {3, 3}])
    dato["name"]= "client"
    return render(request, "readcamuno/templates/index.html", dato)
def user(request):
   dato['name']= 'user'
   return render(request, "readcamuno/templates/about.html", dato)
def guess(request):
   datos[0]['name']= 'user'
   datos[1]['name']= 'guess'
   return render(request, "readcamuno/templates/about.html", {'datos': datos})
   