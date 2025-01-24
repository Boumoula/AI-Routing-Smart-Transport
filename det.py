import pyaudio
import numpy as np
import serial
import time

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