# ACC Project: Intelligent Adaptive Cruise Control (ACC) with Multimodal Perception and Voice Assistance

This project implements an advanced **Adaptive Cruise Control (ACC)** system that combines computer vision, natural language processing, and a speed-decision engine to enhance vehicle safety and driver experience. It was developed as part of a technical internship/project.

## Project Overview

The system is designed to perceive the driving environment, detect potential hazards, interpret traffic regulations, and interact with the driver through a voice-controlled assistant.

### Key Components:
1.  **Multimodal Perception Pipeline**: Uses multiple Deep Learning models (YOLO, TensorFlow, Roboflow) to detect road conditions, traffic signs, speed bumps, pedestrians, and weather.
2.  **Intelligent Speed Decision Engine**: Automatically calculates the optimal vehicle speed based on road type, damage, detected signs, and safety constraints.
3.  **Voice-Controlled Assistant**: A sophisticated interface allowing the driver to control vehicle settings (cruise speed, safety distance, eco mode) using natural language, featuring speech-to-text (Whisper), LLM-based reasoning (Groq/GPT), and text-to-speech (ElevenLabs).
4.  **Local Demo Interface**: A web-based dashboard to visualize the perception results and interact with the voice assistant in real-time.

---

## 🚀 Features

### 1. Computer Vision & Perception
*   **Road Classification**: Identifies road types (highway, urban, etc.) using a Keras/TensorFlow model.
*   **Object Detection**:
    *   **Traffic Signs**: Real-time detection of speed limits and road signs using YOLOv11.
    *   **Speed Bumps & Pedestrian Crosswalks**: Custom-trained YOLO models for specific hazard detection.
    *   **Off-road & Damage Detection**: Integration with Roboflow models for terrain and road condition analysis.
    *   **Vehicle Distance**: YOLOv8-based detection to maintain a safe distance from leading vehicles.
*   **Weather Analysis**: TFLite-based classifier to adjust driving parameters according to weather conditions (rain, fog, clear).

### 2. Voice Assistant & NLP
*   **Multilingual Support**: Powered by OpenAI Whisper for robust speech transcription.
*   **Intelligent Reasoning**: Uses Groq (Llama-3/GPT-4o class models) to understand complex natural language commands.
*   **Propose-Confirm Flow**: Critical commands (like significant speed changes) require verbal confirmation for safety.
*   **Natural TTS**: High-quality voice feedback using ElevenLabs.

### 3. Decision Engine
*   Aggregates data from all perception modules.
*   Applies safety overrides (e.g., slowing down for speed bumps, pedestrians, or rain).
*   Enforces speed limits detected from traffic signs.

---

## 📂 Project Structure

```text
.
├── local/acc-local-demo/   # Local Web Server and Demo UI
│   ├── server.py           # FastAPI backend
│   ├── pipeline.py         # Main integration logic for all models
│   ├── static/             # Frontend HTML/JS dashboard
│   └── requirements.txt    # Dependencies for the local demo
├── rapport/                # Project documentation and reports
├── *.ipynb                 # Development notebooks for model training and testing:
│   ├── pipeline.ipynb      # Initial pipeline prototyping
│   ├── weatherv2.ipynb     # Weather classifier training
│   ├── ppl-crosswalks.ipynb # Pedestrian/Crosswalk detection research
│   └── ...
└── README.md               # This file
```

---

## 🛠️ Getting Started (Local Demo)

### Prerequisites
*   Python 3.10+
*   `ffmpeg` installed on your system (required for audio processing).
*   API Keys for: Roboflow, Groq, and ElevenLabs.

### Installation
1.  Navigate to the demo directory:
    ```bash
    cd local/acc-local-demo/acc-local-demo
    ```
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Configure environment variables:
    ```bash
    cp .env.example .env
    # Edit .env and add your API keys
    ```
4.  Place trained model weights (`.keras`, `.pt`, `.tflite`) in the `acc-local-demo` folder.

### Running the Demo
```bash
python server.py
```
Open your browser to `http://localhost:8000`.

---

## 📊 Methodology & Research

The project involved several stages of development:
1.  **Data Collection & Annotation**: Using Roboflow and custom datasets for road hazards.
2.  **Model Training**: Fine-tuning YOLO models and training custom CNNs for road/weather classification.
3.  **Integration**: Developing a unified `pipeline.py` to handle asynchronous execution of multiple models.
4.  **Simulation & Testing**: Initial testing conducted in notebooks (`.ipynb`) before moving to the local server implementation.

---

## 👥 Authors
Khawla EL HAMDI.
