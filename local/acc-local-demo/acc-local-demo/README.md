# ACC demo — local, running your real pipeline

This is not a simulation. `pipeline.py` is your notebook code (road classifier, Roboflow
off-road/damage, YOLOv11 signs, your trained speed-bump and pedestrian/crosswalk YOLO models,
TFLite weather, YOLOv8n vehicle distance, the speed-decision engine, and the Whisper → Groq
openai/gpt-oss-120b → ElevenLabs voice assistant) wrapped behind a small local web server so a
browser UI can call it. Nothing about the model logic was changed.

Everything runs on your laptop. The browser talks to `http://localhost:8000`, which is your
own machine — nothing is uploaded anywhere except the API calls your notebook already made
(Roboflow, Groq, ElevenLabs).

## 1. Install dependencies

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

You'll also need `ffmpeg` on your PATH (Whisper uses it to decode audio; install via `brew install ffmpeg` on macOS, `sudo apt install ffmpeg` on Linux, or download from ffmpeg.org on Windows) and `git` (used once to auto-clone the traffic-sign repo).

## 2. Add your model files

Place these four files directly next to `server.py` (same folder):

- `road_classifier.keras`
- `speed_bump_detector.pt`
- `ped_crosswalk_detector.pt`
- `weather_classifier.tflite`

`yolov8n.pt` auto-downloads on first run. The traffic-sign repo auto-clones on first run
(needs internet once).

## 3. Add your API keys

```bash
cp .env.example .env
```

Fill in `.env`:

```
ROBOFLOW_API_KEY=...
ELEVENLABS_API_KEY=...
GROQ_API_KEY=...
```

## 4. Run it

```bash
python server.py
```

Open **http://localhost:8000** in your browser (Chrome recommended — needed for microphone
capture on the voice tab).

## What each tab does

- **Perception Pipeline** — upload a road image, it's sent to your local server, run through
  all your real models, and the actual `classify_full()` / `apply_overrides()` result comes
  back and renders live (confidence bars + the final speed decision).
- **Voice Assistant** — press the mic, speak (any language). The recording goes to your
  server, gets transcribed by your local Whisper model, sent to Groq/openai/gpt-oss-120b with your
  real tool schema and propose→confirm flow, and the reply comes back as real ElevenLabs
  audio (your chosen voice, `JBFqnCBsd6RMkjVDRZzb`) and plays in the browser. Typing is a
  fallback if you'd rather not use the mic mid-demo.

## If something 503s

Each model loads lazily on first use. If you see a "not found" or "missing API key" error for
a specific model, it just means that one file/key isn't in place yet — everything else still
works. Check the terminal running `server.py` for the exact missing path.

## Notes for the demo itself

- First request after starting the server will be slow (loading TensorFlow, YOLO weights,
  Whisper, etc. into memory). Run one warm-up image + one warm-up voice command before your
  supervisor arrives.
- The distance heuristics (safety distance, "pedestrian close") are still the placeholder
  thresholds from the notebook — worth a line in your talk track if asked, since they're
  flagged as uncalibrated in your own config.
