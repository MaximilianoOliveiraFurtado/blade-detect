import cv2
import os
import random
import json

# Carregar configurações do dataset
CONFIG_FILE = "config.json"
if not os.path.exists(CONFIG_FILE):
    print(f"[ERRO] Arquivo de configuração '{CONFIG_FILE}' não encontrado.")
    exit(1)

with open(CONFIG_FILE, "r") as f:
    config = json.load(f)

VIDEO_PATH = "assets/video2.mp4"  # Caminho do vídeo
FRAME_SKIP_RATE = config.get("FRAME_SKIP_RATE", 1)  # Define taxa de processamento de frames

# Diretórios do dataset
DATASET_DIR = "dataset"
IMAGES_DIR = os.path.join(DATASET_DIR, "images")
LABELS_DIR = os.path.join(DATASET_DIR, "labels")
TRAIN_IMAGES_DIR = os.path.join(IMAGES_DIR, "train")
VAL_IMAGES_DIR = os.path.join(IMAGES_DIR, "val")
TRAIN_LABELS_DIR = os.path.join(LABELS_DIR, "train")
VAL_LABELS_DIR = os.path.join(LABELS_DIR, "val")

# Criar diretórios se não existirem
for d in [TRAIN_IMAGES_DIR, VAL_IMAGES_DIR, TRAIN_LABELS_DIR, VAL_LABELS_DIR]:
    os.makedirs(d, exist_ok=True)

# Extração de frames do vídeo para dataset
cap = cv2.VideoCapture(VIDEO_PATH)
frame_number = 0
frames_list = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    if frame_number % FRAME_SKIP_RATE == 0:
        frame_filename = f"frame_{frame_number}.jpg"
        frames_list.append(frame_filename)
        cv2.imwrite(os.path.join(IMAGES_DIR, frame_filename), frame)
    
    frame_number += 1

cap.release()
print("Frames extraídos com sucesso!")

# Dividir frames entre train e val (80/20)
random.shuffle(frames_list)
train_split = int(0.8 * len(frames_list))
train_frames = frames_list[:train_split]
val_frames = frames_list[train_split:]

for frame_filename in train_frames:
    os.rename(os.path.join(IMAGES_DIR, frame_filename), os.path.join(TRAIN_IMAGES_DIR, frame_filename))

for frame_filename in val_frames:
    os.rename(os.path.join(IMAGES_DIR, frame_filename), os.path.join(VAL_IMAGES_DIR, frame_filename))

print("Dataset estruturado corretamente em train/ e val/")

# Criar dataset.yaml com caminhos absolutos
DATASET_YAML = os.path.join(DATASET_DIR, "dataset.yaml")

train_path = os.path.abspath(TRAIN_IMAGES_DIR)
val_path = os.path.abspath(VAL_IMAGES_DIR)

with open(DATASET_YAML, "w") as f:
    f.write(f"""
path: {os.path.abspath(DATASET_DIR)}
train: {train_path}
val: {val_path}

nc: 1
names: ["knife"]
""")

print("Arquivo dataset.yaml gerado com sucesso!")
print(f"Dataset salvo em: {DATASET_YAML}")
