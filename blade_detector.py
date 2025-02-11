import cv2
import torch
import json
import requests
from ultralytics import YOLO
import os

# Carregar configurações de e-mail e sensibilidade a partir de um arquivo JSON
CONFIG_FILE = "config.json"
if not os.path.exists(CONFIG_FILE):
    print(f"[ERRO] Arquivo de configuração '{CONFIG_FILE}' não encontrado.")
    exit(1)

with open(CONFIG_FILE, "r") as f:
    config = json.load(f)

SENDGRID_API_KEY = config["SENDGRID_API_KEY"]
EMAIL_SENDER = config["EMAIL_SENDER"]
EMAIL_RECEIVER = config["EMAIL_RECEIVER"]
FRAME_SKIP_RATE = config.get("FRAME_SKIP_RATE", 1)  # Define taxa de processamento de frames
DETECTION_THRESHOLD = config.get("DETECTION_THRESHOLD", 1)  # Número mínimo de frames consecutivos
VIDEO_PATH = config.get("VIDEO_PATH", "assets/video1.mp4")  # Caminho do vídeo
CHANGE_RESOLUTION = config.get("CHANGE_RESOLUTION", False)
BORDER_PROCESSOR = config.get("BORDER_PROCESSOR", False)

# Configuração do modelo YOLOv8 pré-treinado
# model = YOLO("yolov8n.pt")  # Modelo leve pré-treinado
model = YOLO("runs/detect/train2/weights/best.pt") # Caminho do modelo treinado

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

# Verificar se o vídeo existe
if not os.path.exists(VIDEO_PATH):
    print(f"[ERRO] Arquivo de vídeo não encontrado no caminho: {VIDEO_PATH}")
    exit(1)
else:
    print(f"[INFO] Arquivo de vídeo encontrado: {VIDEO_PATH}")

cap = cv2.VideoCapture(VIDEO_PATH)

alert_sent = False
frame_count = 0
detection_count = 0
detected_objects = set()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    if frame_count % FRAME_SKIP_RATE != 0:  # Processa de acordo com a taxa configurada
        continue
    
    # Aumenta a resolução do frame para melhorar detecção
    if CHANGE_RESOLUTION:
        frame = cv2.resize(frame, (1280, 720))  # Ajuste conforme necessário

    # Aplicar pré-processamento para realçar bordas da lâmina
    if BORDER_PROCESSOR:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Converte para escala de cinza
        edges = cv2.Canny(gray, 50, 150)  # Aplica detector de bordas
        frame = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)  # Converte de volta para BGR

    # Realiza a detecção
    results = model(frame)

    detected_in_frame = False
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls)
            if class_id in CUTTING_OBJECTS:
                detected_in_frame = True
                detected_objects.add(CUTTING_OBJECTS[class_id])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = CUTTING_OBJECTS[class_id]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    if detected_in_frame:
        detection_count += 1
    else:
        detection_count = max(0, detection_count - 1)
    
    if detection_count >= DETECTION_THRESHOLD:
        print(f"[DETECÇÃO] Objeto cortante confirmado após {DETECTION_THRESHOLD} frames consecutivos.")
        break
    
    cv2.imshow("Detecção", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

# Enviar alerta ao final do processamento
if detection_count >= DETECTION_THRESHOLD and not alert_sent:
    send_alert(detected_objects)
    alert_sent = True

print("[INFO] Processamento do vídeo concluído.")
