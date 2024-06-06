import numpy as np
from process import extract_features, load_dataset
from fit import create_model, train_model
from prediction import record_audio, predict
import threading

# Función para grabar audio en un hilo separado
def record_audio_thread():
    record_audio('audio.wav')

# Cargar el dataset
dataset_path = r'C:\Users\Nelson\Desktop\Prueba\datasets\Nelson'
X_train, y_train, label_map = load_dataset(dataset_path)

# Verificar si los datos se han cargado correctamente
print(f"X_train.shape: {X_train.shape}")
print(f"y_train.shape: {y_train.shape}")
if X_train.size == 0 or y_train.size == 0:
    raise ValueError("El conjunto de datos está vacío o no se ha cargado correctamente.")

# Crear y entrenar el modelo
model = create_model(input_shape=X_train.shape[1], num_classes=len(label_map))
model = train_model(model, X_train, y_train)

# Iniciar grabación de audio en un hilo separado
audio_thread = threading.Thread(target=record_audio_thread)
audio_thread.start()

# Esperar hasta que se presione una tecla para salir
print("Presiona cualquier tecla para detener la grabación...")
input()

# Esperar a que el hilo de audio termine
audio_thread.join()

# Predecir el audio grabado en tiempo real
predicted_label = predict(model, 'audio.wav', label_map)
print(f'La etiqueta predicha es: {predicted_label}')
