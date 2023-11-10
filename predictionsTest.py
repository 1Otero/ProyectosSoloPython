"""import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input, decode_predictions

# Carga el modelo previamente entrenado
model = tf.keras.models.load_model("model.h5")

# Etiquetas de clase
class_labels = ['Perro', 'Gato']  # Ajusta las etiquetas según tu modelo

# Inicializa la cámara (0 para la cámara predeterminada)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error al abrir la cámara")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error al capturar el fotograma")
        break

    # Redimensiona el fotograma a 224x224 (ajusta según tu modelo)
    frame = cv2.resize(frame, (224, 224))

    # Preprocesa la imagen
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = frame / 255.0
    input_data = np.expand_dims(frame, axis=0)
    input_data = preprocess_input(input_data)
    print(input_data)
    # Realiza la predicción
    predictions = model.predict(input_data)

    # Decodifica las predicciones
    decoded_predictions = decode_predictions(predictions, top=1)[0]

    # Muestra la etiqueta de clase predicha en el fotograma
    label = f"Clase: {decoded_predictions[0][1]}, Probabilidad: {decoded_predictions[0][2]:.2f}"
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Muestra el fotograma en una ventana
    cv2.imshow("Video en tiempo real", frame)

    # Si se presiona 'q', sal del bucle
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera la cámara y cierra todas las ventanas
cap.release()
cv2.destroyAllWindows()
"""
"""import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input

# Carga el modelo previamente entrenado
model = tf.keras.models.load_model("model.h5")

# Etiquetas de clase
class_labels = ['Gato','Perro','Persona','Humano']  # Ajusta las etiquetas según tu modelo

# Inicializa la cámara (0 para la cámara predeterminada)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error al abrir la cámara")
    exit()

while True:
    ret, frame = cap.read()
    # Muestra el fotograma en una ventana
    cv2.imshow("Video en tiempo real", frame)

    # Si se presiona 'q', sal del bucle
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if not ret:
        print("Error al capturar el fotograma")
        break

    # Redimensiona el fotograma a 224x224 (ajusta según tu modelo)
    frame = cv2.resize(frame, (224, 224))

    # Preprocesa la imagen
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = frame / 255.0
    input_data = np.expand_dims(frame, axis=0)
    input_data = preprocess_input(input_data)

    # Realiza la predicción
    predictions = model.predict(input_data)
    print(predictions)
    # Obtiene la etiqueta de clase con la probabilidad más alta
    predicted_class = np.argmax(predictions[0])
    print(predicted_class)
    class_name = class_labels[predicted_class]
    confidence = predictions[0][predicted_class]

    # Muestra la etiqueta de clase predicha y su confianza en el fotograma
    label = f"Clase: {class_name}, Probabilidad: {confidence:.2f}"
    print(label)
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
"""

"""# Muestra el fotograma en una ventana
    cv2.imshow("Video en tiempo real", frame)

    # Si se presiona 'q', sal del bucle
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
"""
# Libera la cámara y cierra todas las ventanas
"""cap.release()
cv2.destroyAllWindows()"""

"""
import cv2
import numpy as np

# Inicializa la cámara (0 para la cámara predeterminada)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error al abrir la cámara")
    exit()

# Captura el primer fotograma
ret, prev_frame = cap.read()
if not ret:
    print("Error al capturar el primer fotograma")
    exit()

while True:
    # Captura el fotograma actual
    ret, frame = cap.read()

    if not ret:
        print("Error al capturar el fotograma actual")
        break

    # Convierte los fotogramas a escala de grises para facilitar la detección de cambios
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calcula la diferencia entre el fotograma actual y el anterior
    frame_diff = cv2.absdiff(prev_gray, current_gray)

    # Aplica un umbral para resaltar los cambios significativos
    _, thresholded = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)

    # Encuentra los contornos de los objetos en movimiento
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Dibuja rectángulos alrededor de los objetos en movimiento
    for contour in contours:
        if cv2.contourArea(contour) > 100:  # Filtra objetos pequeños
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Muestra el fotograma con los objetos en movimiento resaltados
    cv2.imshow("Movimiento detectado", frame)

    # Actualiza el fotograma anterior
    prev_frame = frame.copy()

    # Si se presiona 'q', sal del bucle
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera la cámara y cierra la ventana
cap.release()
cv2.destroyAllWindows()
"""
"""
import cv2
import numpy as np

# Inicializa la cámara (0 para la cámara predeterminada)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error al abrir la cámara")
    exit()

# Captura el primer fotograma
ret, prev_frame = cap.read()
if not ret:
    print("Error al capturar el primer fotograma")
    exit()

while True:
    # Captura el fotograma actual
    ret, frame = cap.read()

    if not ret:
        print("Error al capturar el fotograma actual")
        break

    # Convierte los fotogramas a escala de grises para facilitar la detección de cambios
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calcula la diferencia entre el fotograma actual y el anterior
    frame_diff = cv2.absdiff(prev_gray, current_gray)

    # Aplica un umbral para resaltar los cambios significativos
    _, thresholded = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)

    # Encuentra los contornos de los objetos en movimiento
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Encuentra el contorno más grande (puedes ajustar esto según tus necesidades)
    max_contour = None
    max_contour_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_contour_area:
            max_contour = contour
            max_contour_area = area

    if max_contour is not None:
        # Aproxima el contorno a un polígono con menos vértices
        epsilon = 0.02 * cv2.arcLength(max_contour, True)
        approx = cv2.approxPolyDP(max_contour, epsilon, True)

        # Dibuja la línea siguiendo el contorno aproximado
        cv2.polylines(frame, [approx], isClosed=True, color=(0, 255, 0), thickness=2)

    # Muestra el fotograma con la línea que sigue el contorno del objeto en movimiento
    cv2.imshow("Contorno del objeto en movimiento", frame)

    # Actualiza el fotograma anterior
    prev_frame = frame.copy()

    # Si se presiona 'q', sal del bucle
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera la cámara y cierra la ventana
cap.release()
cv2.destroyAllWindows()
"""
"""
import cv2

# Inicializa la cámara (0 para la cámara predeterminada)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error al abrir la cámara")
    exit()

# Crea un clasificador de caras
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error al capturar el fotograma actual")
        break

    # Convierte la imagen a escala de grises (opcional)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detecta caras en la imagen
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Dibuja rectángulos alrededor de las caras detectadas
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Detección de caras", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Libera la cámara y cierra la ventana
cap.release()
cv2.destroyAllWindows()
"""
"""
import cv2
import numpy as np

# Inicializa la cámara (0 para la cámara predeterminada)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error al abrir la cámara")
    exit()

# Variables para el seguimiento de objetos
first_frame = None
min_area = 500  # Área mínima para considerar un objeto como movimiento

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error al capturar el fotograma actual")
        break

    # Convierte el fotograma actual a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if first_frame is None:
        first_frame = gray
        continue

    # Calcula la diferencia entre el fotograma actual y el primero
    frame_delta = cv2.absdiff(first_frame, gray)

    # Aplica un umbral para detectar píxeles en movimiento
    thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=1)

    # Encuentra contornos de las áreas en movimiento
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue

        # Dibuja un rectángulo alrededor del área en movimiento
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Dibuja una línea (gris en este caso) en la parte superior de la detección de movimiento
        cv2.line(frame, (x, y), (x + w, y), (128, 128, 128), 2)

    cv2.imshow("Detección de Movimiento", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
"""
"""
import cv2
import pyopenpose as op

# Inicializa OpenPose
params = dict()
params["model_folder"] = "/path/to/openpose/models"  # Ruta al directorio de modelos de OpenPose
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

# Inicializa la cámara (0 para la cámara predeterminada)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error al abrir la cámara")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error al capturar el fotograma actual")
        break

    # Procesa el fotograma con OpenPose
    datum = op.Datum()
    datum.cvInputData = frame
    opWrapper.emplaceAndPop([datum])

    # Dibuja los puntos clave del cuerpo en el fotograma
    frame = datum.cvOutputData

    cv2.imshow("Detección de Puntos Clave del Cuerpo", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
"""