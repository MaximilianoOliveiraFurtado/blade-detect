import cv2
import torch
import json
import requests
from ultralytics import YOLO
import os

# Carregar configurações de e-mail a partir de um arquivo JSON
CONFIG_FILE = "config.json"
if not os.path.exists(CONFIG_FILE):
    print(f"[ERRO] Arquivo de configuração '{CONFIG_FILE}' não encontrado.")
    exit(1)

with open(CONFIG_FILE, "r") as f:
    config = json.load(f)

SENDGRID_API_KEY = config["SENDGRID_API_KEY"]
EMAIL_SENDER = config["EMAIL_SENDER"]
EMAIL_RECEIVER = config["EMAIL_RECEIVER"]

# Configuração do modelo YOLOv8 pré-treinado
model = YOLO("yolov8n.pt")  # Modelo leve pré-treinado

# Classe que queremos detectar (baseado em COCO)
CUTTING_OBJECTS = {44: "bottle", 49: "knife", 50: "scissors"}  # IDs de facas e tesouras no COCO

def send_alert(detected_objects):
    """Envia um alerta por e-mail informando se foram encontrados ou não objetos cortantes."""
    if detected_objects:
        subject = "⚠️ Alerta: Objetos cortantes detectados!"
        body = f"Os seguintes objetos foram detectados: {', '.join(detected_objects)}."
    else:
        subject = "✅ Nenhum objeto cortante detectado"
        body = "Nenhum objeto cortante foi encontrado no vídeo."
    
    data = {
        "personalizations": [{"to": [{"email": EMAIL_RECEIVER}]}],
        "from": {"email": EMAIL_SENDER},
        "subject": subject,
        "content": [{"type": "text/plain", "value": body}]
    }
    
    headers = {
        "Authorization": f"Bearer {SENDGRID_API_KEY}",
        "Content-Type": "application/json"
    }
    
    response = requests.post("https://api.sendgrid.com/v3/mail/send", json=data, headers=headers)
    
    if response.status_code == 202:
        print("[ALERTA] E-mail enviado com sucesso via SendGrid!")
    else:
        print("[ERRO] Falha ao enviar e-mail:", response.text)

# Processamento do vídeo
video_path = config["VIDEO_PATH"]  # Altere para o caminho do seu vídeo

if not os.path.exists(video_path):
    print(f"[ERRO] Arquivo de vídeo não encontrado no caminho: {video_path}")
    exit(1)
else:
    print(f"[INFO] Arquivo de vídeo encontrado: {video_path}")

cap = cv2.VideoCapture(video_path)

alert_sent = False
frame_count = 0

detected_objects = set()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    if frame_count % 5 != 0:  # Processa apenas 1 frame a cada 5 para reduzir carga
        continue

    # Realiza a detecção
    results = model(frame)

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls)
            if class_id in CUTTING_OBJECTS:
                detected_objects.add(CUTTING_OBJECTS[class_id])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = CUTTING_OBJECTS[class_id]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    if detected_objects:
        print(f"[DETECÇÃO] Objetos cortantes detectados: {', '.join(detected_objects)}")
    
    cv2.imshow("Detecção", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

# Enviar alerta ao final do processamento
if not alert_sent:
    send_alert(detected_objects)
    alert_sent = True

print("[INFO] Processamento do vídeo concluído.")
