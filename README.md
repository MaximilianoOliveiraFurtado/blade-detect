# 🔪 Sharp Object Detection

Este projeto implementa um detector de objetos cortantes em vídeos utilizando **YOLOv8**. Ele permite detectar facas, tesouras e outros objetos perigosos em vídeos e enviar alertas por e-mail caso sejam identificados.

## 📌 **Requisitos**

Antes de começar, instale os seguintes pacotes:

```bash
pip install ultralytics opencv-python torch torchvision torchaudio labelImg requests
```

## 📂 **Estrutura do Projeto**

```
sharp-object-detection/
│── dataset/                   # Diretório para armazenar imagens e labels
│   ├── images/                # Frames extraídos do vídeo
│   │   ├── train/             # Imagens para treino
│   │   ├── val/               # Imagens para validação
│   ├── labels/                # Labels geradas pelo LabelImg
│   │   ├── train/             # Labels para treino
│   │   ├── val/               # Labels para validação
│   ├── dataset.yaml           # Configuração do YOLO
│── assets/
│   ├── video1.mp4             # Vídeo de exemplo
│── config.json                 # Configurações do modelo e e-mail
│── blade_detector.py           # Script para detecção de objetos cortantes
│── dataset_generator.py        # Geração do dataset a partir de um vídeo
│── training_model.py           # Treinamento do YOLOv8
│── README.md                   # Documentação do projeto
```

## 🚀 **Passo a Passo**

### **1️⃣ Gerar o Dataset**

Antes de treinar o modelo, extraia frames de um vídeo para criar o dataset:

```bash
python dataset_generator.py
```

Isso salvará as imagens extraídas no diretório `dataset/images/`.

### **2️⃣ Rotular as Imagens**

Agora, use o **LabelImg** para marcar os objetos cortantes nos frames:

```bash
labelImg dataset/
```

Ao abrir o LabelImg:

1. Selecione a pasta `dataset/images/`
2. Escolha a pasta de destino das anotações: `dataset/labels/`
3. Crie uma **bounding box** ao redor dos objetos cortantes (exemplo: faca)
4. Salve as anotações

### **3️⃣ Treinar o Modelo**

Após rotular as imagens, inicie o treinamento:

```bash
python training_model.py
```

Esse script usa o YOLOv8 para treinar um modelo personalizado e salvar os pesos na pasta `runs/detect/train/weights/best.pt`.

### **4️⃣ Executar a Detecção**

Agora, com o modelo treinado, execute a detecção em um vídeo:

```bash
python blade_detector.py
```

Se um objeto cortante for detectado, um alerta será enviado por e-mail.

---

## 🛠 \*\*Configuração do \*\*\`\`

Antes de rodar o detector, edite o arquivo `config.json` com as configurações corretas:

```json
{
  "SENDGRID_API_KEY": "sua_chave_api_sendgrid",
  "EMAIL_SENDER": "seuemail@exemplo.com",
  "EMAIL_RECEIVER": "destinatario@exemplo.com",
  "FRAME_SKIP_RATE": 1,
  "DETECTION_THRESHOLD": 1,
  "VIDEO_PATH": "assets/video1.mp4",
  "CHANGE_RESOLUTION": false,
  "BORDER_PROCESSOR": false
}
```

### **Parâmetros:**

- `FRAME_SKIP_RATE`: Define a frequência dos frames analisados
- `DETECTION_THRESHOLD`: Número mínimo de frames consecutivos com detecção antes de disparar o alerta
- `VIDEO_PATH`: Caminho do vídeo a ser processado
- `CHANGE_RESOLUTION`: Aumenta a resolução do frame (caso necessário)
- `BORDER_PROCESSOR`: Ativa realce de bordas para melhorar detecção

---

## 📌 **Observações Importantes**

✅ O YOLO **começa a numeração das classes em 0** (exemplo: `knife` = `0`). ✅ Se as facas não forem detectadas corretamente, verifique `CUTTING_OBJECTS` no código:

```python
CUTTING_OBJECTS = {0: "knife"}  # ID correto do treinamento
```

✅ Se houver detecção no log, mas o código não reconhecer, ajuste `conf`:

```python
results = model(frame, conf=0.05)
```

---

## 📬 **Contato**

Se precisar de ajuda ou quiser contribuir para o projeto, fique à vontade para abrir uma issue ou pull request!

🚀 **Happy Coding!**

