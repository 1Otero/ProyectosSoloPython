import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input, decode_predictions

model= tf.keras.models.load_model("model.h5")
#uri= 'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz'
class_label= ['Perro', 'Gato']
cap= cv2.VideoCapture(0)
if not cap.isOpened():
  print("Error opened cam")
  exit()
while True:
  ret, frame= cap.read()
  cv2.imshow("view", frame)
  if cv2.waitKey(1) & 0xFF == ord("x"):
    break
  img= image.load_img("./f.jpg", target_size=(224, 224))
  ##frame= cv2.resize(frame, (140, 180))
  if not ret:
    print("error capturando el fotograma")
  ##frame= cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
  frame= cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  ##frame= cv2.resize(frame, (224, 224))
  ##frame= cv2.resize(frame, (299, 299))
  frame= frame / 255.0
  frame_numpy= image.img_to_array(frame)

  img_numpy= image.img_to_array(img)
  ##if img_numpy.shape[-1] == 1:
      ##img_numpy= tf.image.grayscale_to_rgb(img_numpy)
  ##input_data= np.expand_dims(frame_numpy, axis=0)
  img_input_data= np.expand_dims(img_numpy, axis=0) 
  #preprocess_input(x)
  ##input_data= preprocess_input(input_data)
  img_input_data= preprocess_input(img_input_data)
  predictions= model.predict(img_input_data)
  ##predictions= model.predict(input_data)
  print(predictions)
  #
  decoded_predictions= decode_predictions(predictions, top=1)[0]
  ##cv2.imshow("cam black and white", frame)
  ##if cv2.waitKey(1) & 0xFF == ord("x"):
    ##break
cap.release()
cv2.destroyAllWindows()