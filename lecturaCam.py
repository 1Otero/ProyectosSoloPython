import cv2
import numpy as np
import tensorflow as tf
#model= tf.keras.models.load_model("path_to_my_model.h5")
uri= 'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz'
#model=tf.keras.models.load_model(uri)
##model= tf.keras.models.Model(uri)
model = tf.saved_model.load(uri)
class_label= ['Perro', 'Gato']
cap= cv2.VideoCapture(0)
if not cap.isOpened():
  print("Error opened cam")
  exit()
while True:
  ret, frame= cap.read()
  ##frame= cv2.resize(frame, (140, 180))
  if not ret:
    print("error capturando el fotograma")
  frame= cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
   
  frame= cv2.resize(frame, (224, 224))
  frame= frame / 255.0

  #input_data= np.expand_dims(frame, axis= 0) 
  #predictions= model.predict(input_data)
  #predicted_class= np.arpmax(predictions, axis=1)[0]
  #print(predicted_class)
  input_tensor = tf.convert_to_tensor(np.expand_dims(frame, axis=0), dtype=tf.float32)
  predictions = model(input_tensor)

  cv2.imshow("cam black and white", frame)
  cv2.waitKey(2)
  if cv2.waitKey(1) & 0xFF == ord("x"):
    break
cap.release()
cv2.destroyAllWindows()