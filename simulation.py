import cv2
from ultralytics import YOLO
import math
from tkinter import *
from PIL import Image, ImageTk
import serial
import cvzone
import pyaudio
import numpy as np
import threading
import time

# Paramètres pour la capture vidéo
cap = cv2.VideoCapture(1)
model = YOLO('best.pt')
classnames = ['ambulance vehicle', 'fire truck vehicle', 'police vehicle']

# Configuration du port série pour communiquer avec l'Arduino
PORT = 'COM5'  # Modifier le port en fonction de votre configuration
BAUDRATE = 9600
arduino = serial.Serial(PORT, BAUDRATE)
time.sleep(2)  # Attente de la stabilité de la connexion

# Paramètres d'enregistrement audio
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
SEUIL = 5000  # À ajuster selon votre environnement
audio_detected = False

# Création de la fenêtre Tkintercd 
root = Tk()
root.title("YOLO Detection")
label = Label(root)
label.pack()

def detecter_sirene(data):
    puissance = np.mean(np.abs(data))
    return puissance > SEUIL

def audio_detection():
    global audio_detected
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    print("* Enregistrement audio démarré *")
    try:
        while True:
            data = stream.read(CHUNK)
            data_np = np.frombuffer(data, dtype=np.int16)
            if detecter_sirene(data_np):
                print("Sirène détectée !")
                audio_detected = True
            else:
                audio_detected = False
    except KeyboardInterrupt:
        print("Interruption par l'utilisateur")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

def update_frame():
    global audio_detected
    ret, frame = cap.read()
    frame = cv2.resize(frame, (640, 480))
    result = model(frame, stream=True)
    visual_detected = False

    for info in result:
        boxes = info.boxes
        for box in boxes:
            confidence = box.conf[0]
            confidence = math.ceil(confidence * 100)
            Class = int(box.cls[0])
            if confidence > 60:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 5)
                cvzone.putTextRect(frame, f'{classnames[Class]} {confidence}%', [x1 + 8, y1 + 100], scale=1.5, thickness=2)
                if Class in [0, 1, 2]:
                    print("Véhicule d'urgence détecté !")
                    visual_detected = True

    print(f"audio_detected: {audio_detected}, visual_detected: {visual_detected}")

    if visual_detected and audio_detected:
        print("Véhicule d'urgence avec sirène détecté ! Envoi du signal à l'Arduino...")
        arduino.write(b'1')
        time.sleep(0.1)  # Très court délai pour laisser l'Arduino traiter le signal
        visual_detected = False
        audio_detected = False
    else:
        arduino.write(b'0')

    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    label.imgtk = imgtk
    label.configure(image=imgtk)
    label.after(10, update_frame)

# Démarrage de la détection audio dans un thread séparé
audio_thread = threading.Thread(target=audio_detection)
audio_thread.start()

# Démarrage de la mise à jour de la trame
update_frame()

# Lancement de la boucle principale Tkinter
root.mainloop()

# Libération des ressources
cap.release()
arduino.close()