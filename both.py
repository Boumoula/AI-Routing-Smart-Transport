import cv2
from ultralytics import YOLO
import math
from tkinter import *
from PIL import Image, ImageTk
import serial
import cvzone
import pyaudio
import numpy as np
import serial
import time

# Initialisation de la capture vidéo à partir de la webcam
cap = cv2.VideoCapture(0)

# Chargement du modèle YOLO pré-entraîné ('best.pt') pour la détection des objets
model = YOLO('best.pt')

# Définition des nouvelles classes à détecter
classnames = ['ambulance vehicle', 'fire truck vehicle', 'police vehicle']

# Création de la fenêtre Tkinter
root = Tk()
root.title("YOLO Detection")

# Création d'un label pour afficher l'image
label = Label(root)
label.pack()

# Configuration du port série pour communiquer avec l'Arduino
ser = serial.Serial('COM5', 9600)  # Modifier le port COM selon votre configuration

def update_frame():
    ret, frame = cap.read()
    frame = cv2.resize(frame, (640,480))
    
    # Utilisation du modèle YOLO pour détecter les objets dans la trame
    result = model(frame, stream=True)
    
    # Parcours des résultats de détection pour chaque objet détecté
    for info in result:
        boxes = info.boxes
        for box in boxes:
            confidence = box.conf[0]
            confidence = math.ceil(confidence * 100)
            Class = int(box.cls[0])
            
            if confidence > 60:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Dessin d'un rectangle autour de l'objet détecté
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 5)
                
                # Affichage du label de la classe avec la confiance détectée
                cvzone.putTextRect(frame, f'{classnames[Class]} {confidence}%', [x1 + 8, y1 + 100], scale=1.5, thickness=2)
                
                # Si une ambulance est détectée, envoyer un signal à l'Arduino pour allumer la LED
                if Class == 0 or Class == 1 or Class == 2:  # Indice de la classe "ambulance vehicle"
                    print("Ambulance détectée ! Envoi du signal à l'Arduino...")
                    ser.write(b'1')  # Envoyer le signal à l'Arduino pour allumer la LED
    
    # Conversion de l'image en format compatible avec Tkinter
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    label.imgtk = imgtk
    label.configure(image=imgtk)
    
    # Mise à jour de la trame toutes les 10ms
    label.after(10, update_frame)

# Démarrage de la mise à jour de la trame
update_frame()

# Lancement de la boucle principale Tkinter
root.mainloop()



# Paramètres d'enregistrement audio
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 5  # Durée d'enregistrement

# Seuil de détection de la sirène
SEUIL = 10000  # À ajuster selon votre environnement

# Configuration du port série pour communiquer avec Arduino
PORT = 'COM5'  # Modifier le port en fonction de votre configuration
BAUDRATE = 9600

def detecter_sirene(data):
    # Calculer la puissance du signal audio
    puissance = np.mean(np.abs(data))
    if puissance > SEUIL:
        return True
    else:
        return False

def allumer_led():
    arduino = serial.Serial(PORT, BAUDRATE)
    time.sleep(2)  # Attente de la stabilité de la connexion
    arduino.write(b'1')  # Envoyer la commande pour allumer la LED
    arduino.close()

def main():
    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("* Enregistrement audio démarré *")

    while True:
        data = stream.read(CHUNK)
        data_np = np.frombuffer(data, dtype=np.int16)

        if detecter_sirene(data_np):
            print("Sirène détectée !")
            allumer_led()

    stream.stop_stream()
    stream.close()
    p.terminate()

if __name__ == "__main__":
    main()
# Libération de la capture vidéo et fermeture du port série
cap.release()
ser.close()

