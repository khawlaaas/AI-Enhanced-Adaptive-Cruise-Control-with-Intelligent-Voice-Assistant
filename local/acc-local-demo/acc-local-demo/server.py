"""
Local server for the ACC demo. Runs your actual pipeline (pipeline.py) and serves the
frontend in static/. Everything stays on your laptop — no data leaves localhost.

Run:
    pip install -r requirements.txt
    cp .env.example .env   # then fill in your API keys
    # place road_classifier.keras, speed_bump_detector.pt, ped_crosswalk_detector.pt,
    # weather_classifier.tflite next to this file
    python server.py

Then open http://localhost:8000 in your browser.
"""

import base64
import os
import tempfile
import traceback

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import pipeline

app = FastAPI(title="ACC Pipeline — Local Demo")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

vehicle_state = {
    "cruise_speed_kmh": 50,
    "speed_limit_kmh": None,
    "eco_mode": False,
    "safety_distance_level": 2,
    "window_open": True,
}


def _model_not_ready(e: pipeline.ModelNotReady):
    raise HTTPException(status_code=503, detail=str(e))


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/classify")
async def classify(image: UploadFile = File(...)):
    suffix = os.path.splitext(image.filename or "upload.jpg")[1] or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await image.read())
        tmp_path = tmp.name
    try:
        result = pipeline.classify_full(tmp_path)
        return result
    except pipeline.ModelNotReady as e:
        _model_not_ready(e)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")
    finally:
        os.unlink(tmp_path)


def _audio_b64(text, language_code):
    """Best-effort TTS — if ElevenLabs isn't configured, returns None instead of failing the turn."""
    try:
        audio_bytes = pipeline.speak_to_bytes(text, language_code=language_code)
        return base64.b64encode(audio_bytes).decode("ascii")
    except pipeline.ModelNotReady:
        return None
    except Exception:
        traceback.print_exc()
        return None


@app.post("/api/voice/text")
async def voice_text(text: str = Form(...), personality: str = Form("professional"), language: str = Form("en")):
    global vehicle_state
    try:
        spoken_text, vehicle_state, needs_confirmation = pipeline.understand_and_respond(text, vehicle_state, personality)
    except pipeline.ModelNotReady as e:
        _model_not_ready(e)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")

    return {
        "spoken_text": spoken_text,
        "needs_confirmation": needs_confirmation,
        "vehicle_state": vehicle_state,
        "audio_base64": _audio_b64(spoken_text, language),
    }


@app.post("/api/voice/audio")
async def voice_audio(audio: UploadFile = File(...), personality: str = Form("professional")):
    global vehicle_state
    suffix = os.path.splitext(audio.filename or "clip.webm")[1] or ".webm"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name
    try:
        text, detected_language = pipeline.transcribe_audio(tmp_path)
        if not text:
            return {"heard_text": "", "spoken_text": None, "needs_confirmation": False, "vehicle_state": vehicle_state, "audio_base64": None}
        spoken_text, vehicle_state, needs_confirmation = pipeline.understand_and_respond(text, vehicle_state, personality)
        return {
            "heard_text": text,
            "detected_language": detected_language,
            "spoken_text": spoken_text,
            "needs_confirmation": needs_confirmation,
            "vehicle_state": vehicle_state,
            "audio_base64": _audio_b64(spoken_text, detected_language),
        }
    except pipeline.ModelNotReady as e:
        _model_not_ready(e)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")
    finally:
        os.unlink(tmp_path)


@app.post("/api/voice/confirm-text")
async def voice_confirm_text(text: str = Form(...), language: str = Form("en")):
    global vehicle_state
    try:
        vehicle_state, response_text, confirmed = pipeline.confirm_pending_command(text, vehicle_state)
        return {"response_text": response_text, "confirmed": confirmed, "vehicle_state": vehicle_state, "audio_base64": _audio_b64(response_text, language)}
    except pipeline.ModelNotReady as e:
        _model_not_ready(e)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")


@app.post("/api/voice/confirm-audio")
async def voice_confirm_audio(audio: UploadFile = File(...)):
    global vehicle_state
    suffix = os.path.splitext(audio.filename or "clip.webm")[1] or ".webm"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name
    try:
        text, detected_language = pipeline.transcribe_audio(tmp_path)
        if not text:
            return {"heard_text": "", "response_text": None, "confirmed": False, "vehicle_state": vehicle_state, "audio_base64": None}
        vehicle_state, response_text, confirmed = pipeline.confirm_pending_command(text, vehicle_state)
        return {
            "heard_text": text,
            "response_text": response_text,
            "confirmed": confirmed,
            "vehicle_state": vehicle_state,
            "audio_base64": _audio_b64(response_text, detected_language),
        }
    except pipeline.ModelNotReady as e:
        _model_not_ready(e)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")
    finally:
        os.unlink(tmp_path)


@app.get("/api/vehicle-state")
def get_vehicle_state():
    return vehicle_state


@app.get("/api/pending")
def get_pending():
    return {"pending": pipeline.has_pending_command()}


app.mount("/", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static"), html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
