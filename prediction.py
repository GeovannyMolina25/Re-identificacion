# prediccion.py
import pyaudio
import wave
import numpy as np
from process import extract_features

def record_audio(filename, duration=5, fs=44100):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=fs, input=True, frames_per_buffer=1024)
    frames = []

    for _ in range(0, int(fs / 1024 * duration)):
        data = stream.read(1024)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(filename, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(fs)
    wf.writeframes(b''.join(frames))
    wf.close()

def predict(model, audio_file, label_map):
    features = extract_features(audio_file)
    features = np.expand_dims(features, axis=0)
    prediction = model.predict(features)
    predicted_label = np.argmax(prediction)
    
    # Invertir el label_map para obtener el nombre del label
    inv_label_map = {v: k for k, v in label_map.items()}
    return inv_label_map[predicted_label]
