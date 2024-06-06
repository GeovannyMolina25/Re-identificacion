import librosa
import numpy as np
import os

def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        return np.mean(mfccs.T, axis=0)
    except Exception as e:
        print(f"Error al extraer características de {file_path}: {e}")
        return None
    
def load_dataset(dataset_path):
    features = []
    labels = []
    label_map = {}
    label_counter = 0

    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.wav'):
                file_path = os.path.join(root, file)
                print(f"Procesando archivo: {file_path}")
                feature = extract_features(file_path)
                if feature is not None:
                    features.append(feature)
                    label = os.path.basename(root)  # Usar el nombre del directorio como etiqueta
                    if label not in label_map:
                        label_map[label] = label_counter
                        label_counter += 1
                    labels.append(label_map[label])
                else:
                    print(f"Característica no extraída para: {file_path}")

    features = np.array(features)
    labels = np.array(labels)
    print(f"Total de archivos procesados: {len(features)}")
    return features, labels, label_map

