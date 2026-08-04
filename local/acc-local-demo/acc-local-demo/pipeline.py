"""
Your actual ACC pipeline, ported 1:1 out of fullwithhistory.ipynb into an importable module.
No logic was changed — model loading, detection functions, speed decision logic, and the
Whisper -> Groq -> ElevenLabs voice assistant are the same code, just moved out of notebook
cells so server.py can call them.

Requires (same as the notebook):
- road_classifier.keras, speed_bump_detector.pt, ped_crosswalk_detector.pt, weather_classifier.tflite
  placed in the project root (or set MODEL_DIR below)
- ROBOFLOW_API_KEY, ELEVENLABS_API_KEY, GROQ_API_KEY in a .env file next to this module
- yolov8n.pt auto-downloads on first run
- the traffic-sign repo auto-clones on first run (needs internet once)
"""

import os
import re
import json
import numpy as np
import tensorflow as tf
from PIL import Image

from dotenv import load_dotenv

load_dotenv()

# ============================================================
# Config — identical values to the notebook
# ============================================================
MODEL_DIR = os.getenv("MODEL_DIR", os.path.dirname(os.path.abspath(__file__)))

IMG_SIZE = (224, 224)
CLASS_NAMES = ["autoroute", "urbaine", "rurale"]

LOCAL_MODEL_PATH = os.path.join(MODEL_DIR, "road_classifier.keras")

ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

OFFROAD_WORKSPACE = "sri-lab"
OFFROAD_PROJECT = "road-surface-classification-lgxl1"
OFFROAD_VERSION = 1
UNPAVED_CLASS_NAMES = {"unpaved"}
OFFROAD_CONFIDENCE_THRESHOLD = 0.5

DAMAGE_WORKSPACE = "roaddamage-msfnj"
DAMAGE_PROJECT = "road-damage-ww8ex"
DAMAGE_VERSION = 1
DAMAGE_CONFIDENCE_THRESHOLD = 0.5

SIGN_REPO_URL = "https://github.com/bhaskrr/traffic-sign-detection-using-yolov11.git"
SIGN_REPO_LOCAL_DIR = "traffic_sign_repo"
SIGN_WEIGHTS_FILENAME = "traffic_sign_detector.pt"  # confirmed real filename in the repo
SIGN_CONFIDENCE_THRESHOLD = 0.5
IGNORED_SIGN_CLASSES = {"all"}

ELEVENLABS_VOICE_ID = "JBFqnCBsd6RMkjVDRZzb"
ELEVENLABS_MODEL_ID = "eleven_multilingual_v2"

SPEED_BUMP_MODEL_PATH = os.path.join(MODEL_DIR, "speed_bump_detector.pt")
SPEED_BUMP_CONFIDENCE_THRESHOLD = 0.5

PED_CROSSWALK_MODEL_PATH = os.path.join(MODEL_DIR, "ped_crosswalk_detector.pt")
PEDESTRIAN_CONFIDENCE_THRESHOLD = 0.5
CROSSWALK_CONFIDENCE_THRESHOLD = 0.5
PEDESTRIAN_CLOSE_RELATIVE_HEIGHT_THRESHOLD = 0.30
PEDESTRIAN_CLOSE_RELATIVE_BOTTOM_THRESHOLD = 0.80

WEATHER_MODEL_PATH = os.path.join(MODEL_DIR, "weather_classifier.tflite")
WEATHER_CLASSES = ["clear", "adverse"]
WEATHER_CONFIDENCE_THRESHOLD = 0.7

VEHICLE_CLASSES = ["car", "truck", "bus"]
VEHICLE_CONFIDENCE_THRESHOLD = 0.4
CLOSE_RELATIVE_HEIGHT_THRESHOLD = 0.35
CLOSE_RELATIVE_BOTTOM_THRESHOLD = 0.85

BASE_SPEED_BY_ROAD_TYPE = {"autoroute": 120, "urbaine": 60, "rurale": 100, "off-road": 50}
DAMAGE_SPEED_REDUCTION = 0.30
CROSSWALK_SPEED_CAP = 30
SPEED_BUMP_SPEED_CAP = 30
SAFETY_DISTANCE_SPEED_REDUCTION = 0.20
WEATHER_SPEED_REDUCTION = 0.20
PEDESTRIAN_FAR_SPEED_REDUCTION = 0.30
AUTOMATIC_REASON_CODES = {"pedestrian_stop", "stop_sign", "red_light"}

LLM_MODEL = "llama-3.3-70b-versatile"
CONVERSATION_HISTORY_TURNS = 5

PERSONALITY_TONE = {
    "professional": "Tone: composed, precise and formal — like a professional co-pilot briefing the driver.",
    "friendly": "Tone: warm and casual, like a friendly companion chatting with the driver — still efficient.",
    "sport": "Tone: energetic and upbeat, short punchy phrases, like a co-driver hyped for the road ahead.",
    "zen": "Tone: calm, gentle and reassuring, unhurried phrasing that keeps the driver relaxed.",
}

BASE_SYSTEM_PROMPT = """You are the voice assistant embedded in an adaptive cruise control (ADAS) system in a car.

Always reply in the exact same language the user just spoke to you in, whatever language that is.

You have two jobs:
1. If the user gives a command to change something about the car (window, speed limit, cruise speed, eco mode, safety distance, speed up/slow down), call the vehicle_command tool with the right action (and value in km/h if relevant). IMPORTANT: nothing is applied yet at this point -- along with the tool call, phrase your spoken reply as a short CONFIRMATION QUESTION asking the driver if they want you to do it (e.g. "Do you want me to set cruise speed to 100?"), in the same language the user spoke. The action only actually happens after the driver confirms on the next turn.
2. For anything else -- greetings, small talk, "who are you", "what do you do", general questions -- just answer normally and conversationally, like any helpful voice assistant would. Keep replies short (1-3 sentences), since this is spoken out loud in a car, not read on a screen.

Only call the tool for real commands. Never invent vehicle behavior you were not asked for.
"""

VEHICLE_TOOL = {
    "type": "function",
    "function": {
        "name": "vehicle_command",
        "description": (
            "Call this when the user gives an instruction that changes something "
            "about the car (window, speed, cruise speed, eco mode, safety distance). "
            "Do NOT call this for questions, greetings, or general conversation."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "close_window", "open_window",
                        "set_speed_limit", "set_cruise_speed",
                        "enable_eco_mode",
                        "increase_safety_distance", "decrease_safety_distance",
                        "decrease_speed", "increase_speed",
                    ],
                },
                "value": {"type": "integer", "description": "Speed in km/h. Only used for set_speed_limit and set_cruise_speed."},
            },
            "required": ["action"],
        },
    },
}

CONFIRMATION_TOOL = {
    "type": "function",
    "function": {
        "name": "record_confirmation",
        "description": "Call this to record whether the driver confirmed or declined the pending action.",
        "parameters": {
            "type": "object",
            "properties": {"confirmed": {"type": "boolean", "description": "true if the driver agreed/said yes, false if they declined/said no."}},
            "required": ["confirmed"],
        },
    },
}

# ============================================================
# Lazy-loaded model handles — loaded on first use, cached after,
# so the server can start even if some model files aren't in place yet.
# ============================================================
_road_classifier = None
_offroad_model = None
_damage_model_rf = None
_sign_model = None
_bump_model = None
_ped_crosswalk_model = None
_weather_interpreter = None
_weather_input_details = None
_weather_output_details = None
_vehicle_model = None
_whisper_model = None
_groq_client = None
_elevenlabs_client = None
_rf = None


class ModelNotReady(Exception):
    """Raised when a required model file / API key isn't in place yet."""


def get_road_classifier():
    global _road_classifier
    if _road_classifier is None:
        if not os.path.exists(LOCAL_MODEL_PATH):
            raise ModelNotReady(f"{LOCAL_MODEL_PATH} not found — place your trained road_classifier.keras next to server.py.")
        _road_classifier = tf.keras.models.load_model(LOCAL_MODEL_PATH)
    return _road_classifier


def get_roboflow():
    global _rf
    if _rf is None:
        if not ROBOFLOW_API_KEY:
            raise ModelNotReady("ROBOFLOW_API_KEY missing from .env")
        from roboflow import Roboflow
        _rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    return _rf


def get_offroad_model():
    global _offroad_model
    if _offroad_model is None:
        rf = get_roboflow()
        project = rf.workspace(OFFROAD_WORKSPACE).project(OFFROAD_PROJECT)
        _offroad_model = project.version(OFFROAD_VERSION).model
    return _offroad_model


def get_damage_model():
    global _damage_model_rf
    if _damage_model_rf is None:
        rf = get_roboflow()
        project = rf.workspace(DAMAGE_WORKSPACE).project(DAMAGE_PROJECT)
        _damage_model_rf = project.version(DAMAGE_VERSION).model
    return _damage_model_rf


def get_sign_model():
    global _sign_model
    if _sign_model is None:
        from ultralytics import YOLO
        if not os.path.exists(SIGN_REPO_LOCAL_DIR):
            os.system(f"git clone {SIGN_REPO_URL} {SIGN_REPO_LOCAL_DIR}")
        model_dir = os.path.join(SIGN_REPO_LOCAL_DIR, "model")
        sign_weights_path = os.path.join(model_dir, SIGN_WEIGHTS_FILENAME)
        if not os.path.exists(sign_weights_path):
            raise ModelNotReady(f"{sign_weights_path} not found after cloning the sign repo.")
        _sign_model = YOLO(sign_weights_path)
    return _sign_model


def get_bump_model():
    global _bump_model
    if _bump_model is None:
        from ultralytics import YOLO
        if not os.path.exists(SPEED_BUMP_MODEL_PATH):
            raise ModelNotReady(f"{SPEED_BUMP_MODEL_PATH} not found — place your trained speed_bump_detector.pt next to server.py.")
        _bump_model = YOLO(SPEED_BUMP_MODEL_PATH)
    return _bump_model


def get_ped_crosswalk_model():
    global _ped_crosswalk_model
    if _ped_crosswalk_model is None:
        from ultralytics import YOLO
        if not os.path.exists(PED_CROSSWALK_MODEL_PATH):
            raise ModelNotReady(f"{PED_CROSSWALK_MODEL_PATH} not found — place your trained ped_crosswalk_detector.pt next to server.py.")
        _ped_crosswalk_model = YOLO(PED_CROSSWALK_MODEL_PATH)
    return _ped_crosswalk_model


def get_weather_interpreter():
    global _weather_interpreter, _weather_input_details, _weather_output_details
    if _weather_interpreter is None:
        if not os.path.exists(WEATHER_MODEL_PATH):
            raise ModelNotReady(f"{WEATHER_MODEL_PATH} not found — place your trained weather_classifier.tflite next to server.py.")
        _weather_interpreter = tf.lite.Interpreter(model_path=WEATHER_MODEL_PATH)
        _weather_interpreter.allocate_tensors()
        _weather_input_details = _weather_interpreter.get_input_details()
        _weather_output_details = _weather_interpreter.get_output_details()
    return _weather_interpreter, _weather_input_details, _weather_output_details


def get_vehicle_model():
    global _vehicle_model
    if _vehicle_model is None:
        from ultralytics import YOLO
        _vehicle_model = YOLO("yolov8n.pt")  # auto-downloads
    return _vehicle_model


def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        import whisper
        _whisper_model = whisper.load_model("base")
    return _whisper_model


def get_groq_client():
    global _groq_client
    if _groq_client is None:
        if not GROQ_API_KEY:
            raise ModelNotReady("GROQ_API_KEY missing from .env")
        from groq import Groq
        _groq_client = Groq(api_key=GROQ_API_KEY)
    return _groq_client


def get_elevenlabs_client():
    global _elevenlabs_client
    if _elevenlabs_client is None:
        if not ELEVENLABS_API_KEY:
            raise ModelNotReady("ELEVENLABS_API_KEY missing from .env")
        from elevenlabs.client import ElevenLabs
        _elevenlabs_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
    return _elevenlabs_client


# ============================================================
# Detection functions — unchanged from the notebook
# ============================================================
def detect_offroad(image_path):
    result = get_offroad_model().predict(image_path).json()
    predictions = result.get("predictions", [])
    if not predictions:
        return False, 0.0
    top = predictions[0] if isinstance(predictions, list) else predictions
    predicted_class = top.get("class") or top.get("top", "")
    confidence = top.get("confidence", 0.0)
    is_offroad = predicted_class in UNPAVED_CLASS_NAMES and confidence >= OFFROAD_CONFIDENCE_THRESHOLD
    return is_offroad, confidence


def detect_damage(image_path):
    result = get_damage_model().predict(image_path, confidence=int(DAMAGE_CONFIDENCE_THRESHOLD * 100), overlap=30).json()
    detections = result.get("predictions", [])
    parsed = [{"class": d.get("class"), "confidence": d.get("confidence", 0.0)} for d in detections]
    max_confidence = max((d["confidence"] for d in parsed), default=0.0)
    is_damaged = len(parsed) > 0
    return is_damaged, max_confidence, parsed


def detect_traffic_signs(image_path):
    results = get_sign_model().predict(image_path, verbose=False)
    detections = []
    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            cls_name = get_sign_model().names[int(box.cls[0])]
            if cls_name in IGNORED_SIGN_CLASSES or conf < SIGN_CONFIDENCE_THRESHOLD:
                continue
            detections.append({"class": cls_name, "confidence": conf})
    return detections


def detect_speed_bump(image_path):
    results = get_bump_model().predict(image_path, conf=SPEED_BUMP_CONFIDENCE_THRESHOLD, verbose=False)
    detections = []
    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            cls_name = get_bump_model().names[int(box.cls[0])]
            detections.append({"class": cls_name, "confidence": conf})
    bump_detections = [d for d in detections if d["class"] == "Speed-Bump"]
    bump_ahead = len(bump_detections) > 0
    max_confidence = max((d["confidence"] for d in bump_detections), default=0.0)
    return bump_ahead, max_confidence, detections


def detect_pedestrian_crosswalk(image_path):
    img = Image.open(image_path)
    frame_w, frame_h = img.size
    model = get_ped_crosswalk_model()
    results = model.predict(image_path, conf=min(PEDESTRIAN_CONFIDENCE_THRESHOLD, CROSSWALK_CONFIDENCE_THRESHOLD), verbose=False)
    detections = []
    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            cls_name = model.names[int(box.cls[0])]
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append({"class": cls_name, "confidence": conf, "relative_height": (y2 - y1) / frame_h, "relative_bottom": y2 / frame_h})

    pedestrian_detections = [d for d in detections if d["class"] == "person" and d["confidence"] >= PEDESTRIAN_CONFIDENCE_THRESHOLD]
    crosswalk_detections = [d for d in detections if d["class"] == "cross walk" and d["confidence"] >= CROSSWALK_CONFIDENCE_THRESHOLD]
    close_pedestrians = [
        d for d in pedestrian_detections
        if d["relative_height"] >= PEDESTRIAN_CLOSE_RELATIVE_HEIGHT_THRESHOLD or d["relative_bottom"] >= PEDESTRIAN_CLOSE_RELATIVE_BOTTOM_THRESHOLD
    ]
    pedestrian_close = len(close_pedestrians) > 0
    pedestrian_far = len(pedestrian_detections) > 0 and not pedestrian_close
    crosswalk_ahead = len(crosswalk_detections) > 0
    return pedestrian_close, pedestrian_far, crosswalk_ahead, detections


def detect_weather(image_path):
    interpreter, input_details, output_details = get_weather_interpreter()
    img = tf.keras.utils.load_img(image_path, target_size=IMG_SIZE)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0).astype(input_details[0]["dtype"])
    interpreter.set_tensor(input_details[0]["index"], img_array)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]["index"])[0]
    predicted_class = WEATHER_CLASSES[np.argmax(predictions)]
    confidence = float(np.max(predictions))
    if predicted_class == "adverse" and confidence < WEATHER_CONFIDENCE_THRESHOLD:
        return "clear", confidence
    return predicted_class, confidence


def detect_car_too_close(image_path):
    img = Image.open(image_path)
    frame_w, frame_h = img.size
    model = get_vehicle_model()
    results = model.predict(image_path, conf=VEHICLE_CONFIDENCE_THRESHOLD, verbose=False)
    detections = []
    for r in results:
        for box in r.boxes:
            cls_name = model.names[int(box.cls[0])]
            if cls_name not in VEHICLE_CLASSES:
                continue
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            relative_height = (y2 - y1) / frame_h
            relative_bottom = y2 / frame_h
            detections.append({"class": cls_name, "confidence": conf, "relative_height": relative_height, "relative_bottom": relative_bottom})
    close_vehicles = [d for d in detections if d["relative_height"] >= CLOSE_RELATIVE_HEIGHT_THRESHOLD or d["relative_bottom"] >= CLOSE_RELATIVE_BOTTOM_THRESHOLD]
    too_close = len(close_vehicles) > 0
    closest = max(close_vehicles, key=lambda d: d["relative_bottom"], default=None)
    return too_close, closest, detections


def get_base_speed(road_type, is_damaged):
    speed = BASE_SPEED_BY_ROAD_TYPE[road_type]
    if is_damaged:
        speed = speed * (1 - DAMAGE_SPEED_REDUCTION)
    return round(speed, 1)


def apply_overrides(current_speed, detected_signs, pedestrian_close=False, pedestrian_far=False,
                     crosswalk_ahead=False, bump_ahead=False, car_too_close=False, weather_condition="clear"):
    if pedestrian_close:
        return 0, "pedestrian_stop", {}
    for sign in detected_signs:
        if sign["class"] == "Stop":
            return 0, "stop_sign", {}
        if sign["class"] == "Red Light":
            return 0, "red_light", {}

    final_speed = current_speed
    reason_code = None
    reason_details = {}

    if crosswalk_ahead and CROSSWALK_SPEED_CAP < final_speed:
        final_speed = CROSSWALK_SPEED_CAP
        reason_code, reason_details = "crosswalk", {"speed": CROSSWALK_SPEED_CAP}
    if bump_ahead and SPEED_BUMP_SPEED_CAP < final_speed:
        final_speed = SPEED_BUMP_SPEED_CAP
        reason_code, reason_details = "speed_bump", {"speed": SPEED_BUMP_SPEED_CAP}
    for sign in detected_signs:
        match = re.search(r"Speed Limit (\d+)", sign["class"])
        if match:
            posted_limit = int(match.group(1))
            if posted_limit < final_speed:
                final_speed = posted_limit
                reason_code, reason_details = "speed_limit", {"speed": posted_limit}
    if pedestrian_far:
        final_speed = round(final_speed * (1 - PEDESTRIAN_FAR_SPEED_REDUCTION), 1)
        reason_code, reason_details = "pedestrian_far", {"speed": final_speed}
    if car_too_close:
        final_speed = round(final_speed * (1 - SAFETY_DISTANCE_SPEED_REDUCTION), 1)
        reason_code, reason_details = "safety_distance", {"speed": final_speed}
    if weather_condition == "adverse":
        final_speed = round(final_speed * (1 - WEATHER_SPEED_REDUCTION), 1)
        reason_code, reason_details = "weather", {"speed": final_speed}

    return final_speed, reason_code, reason_details


def classify_full(image_path):
    is_offroad, offroad_conf = detect_offroad(image_path)
    road_conf = offroad_conf
    if is_offroad:
        road_type = "off-road"
    else:
        img = tf.keras.utils.load_img(image_path, target_size=IMG_SIZE)
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        predictions = get_road_classifier().predict(img_array, verbose=0)
        road_type = CLASS_NAMES[np.argmax(predictions[0])]
        road_conf = float(np.max(predictions[0]))

    is_damaged, damage_conf, damage_detections = detect_damage(image_path)
    detected_signs = detect_traffic_signs(image_path)
    bump_ahead, bump_conf, bump_detections = detect_speed_bump(image_path)
    pedestrian_close, pedestrian_far, crosswalk_ahead, ped_detections = detect_pedestrian_crosswalk(image_path)
    ped_confs = [d["confidence"] for d in ped_detections if d["class"] == "person"]
    ped_conf = max(ped_confs) if ped_confs else 0.0
    car_too_close, closest_vehicle, vehicle_detections = detect_car_too_close(image_path)
    weather_condition, weather_conf = detect_weather(image_path)

    base_speed = get_base_speed(road_type, is_damaged)
    final_speed, reason_code, reason_details = apply_overrides(
        base_speed, detected_signs, pedestrian_close=pedestrian_close, pedestrian_far=pedestrian_far,
        crosswalk_ahead=crosswalk_ahead, bump_ahead=bump_ahead, car_too_close=car_too_close, weather_condition=weather_condition,
    )

    return {
        "road_type": road_type, "road_confidence": road_conf,
        "damaged": is_damaged, "damage_confidence": damage_conf,
        "detected_signs": detected_signs,
        "pedestrian_close": pedestrian_close, "pedestrian_far": pedestrian_far, "pedestrian_confidence": ped_conf,
        "crosswalk_ahead": crosswalk_ahead,
        "speed_bump_detected": bump_ahead, "speed_bump_confidence": bump_conf,
        "car_too_close": car_too_close, "vehicle_confidence": closest_vehicle["confidence"] if closest_vehicle else None,
        "weather_condition": weather_condition, "weather_confidence": weather_conf,
        "base_speed": base_speed, "final_speed_kmh": final_speed,
        "reason_code": reason_code, "reason_details": reason_details,
    }


# ============================================================
# Voice assistant — Whisper (STT) + Groq/Llama-3.3-70B (NLU/tools) + ElevenLabs (TTS)
# Server-side session state (single-driver local demo, same as the notebook's module globals).
# ============================================================
_pending_command = {"data": None, "question": None}
_conversation_history = []


def transcribe_audio(audio_path):
    result = get_whisper_model().transcribe(audio_path)
    return result["text"].strip(), result["language"]


def _append_history(user_text, assistant_text):
    _conversation_history.append({"role": "user", "text": user_text})
    _conversation_history.append({"role": "assistant", "text": assistant_text})
    max_entries = CONVERSATION_HISTORY_TURNS * 2
    if len(_conversation_history) > max_entries:
        del _conversation_history[: len(_conversation_history) - max_entries]


def _history_as_messages():
    return [{"role": turn["role"], "content": turn["text"]} for turn in _conversation_history]


def dispatch_command(action, value, vehicle_state):
    if action == "set_speed_limit" and value is not None:
        vehicle_state["speed_limit_kmh"] = value
    elif action == "set_cruise_speed" and value is not None:
        vehicle_state["cruise_speed_kmh"] = value
    elif action == "enable_eco_mode":
        vehicle_state["eco_mode"] = True
    elif action == "increase_safety_distance":
        vehicle_state["safety_distance_level"] += 1
    elif action == "decrease_safety_distance":
        vehicle_state["safety_distance_level"] = max(1, vehicle_state["safety_distance_level"] - 1)
    elif action == "decrease_speed":
        vehicle_state["cruise_speed_kmh"] = max(0, vehicle_state["cruise_speed_kmh"] - 10)
    elif action == "increase_speed":
        vehicle_state["cruise_speed_kmh"] += 10
    elif action == "close_window":
        vehicle_state["window_open"] = False
    elif action == "open_window":
        vehicle_state["window_open"] = True
    return vehicle_state


def _judge_yes_no(confirmation_text, question_asked):
    client = get_groq_client()
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": (
                f'You are the voice assistant in a car. You just asked the driver: "{question_asked}"\n'
                "The driver has now replied. Call record_confirmation with confirmed=true if they "
                "agreed/said yes, confirmed=false if they declined/said no.\n"
                "Also write one short spoken sentence in the same language the driver just used: "
                "if confirmed, say you're doing it now; if declined, acknowledge you won't change anything."
            )},
            {"role": "user", "content": confirmation_text},
        ],
        tools=[CONFIRMATION_TOOL],
        tool_choice="auto",
    )
    message = response.choices[0].message
    confirmed = False
    if message.tool_calls:
        for call in message.tool_calls:
            if call.function.name == "record_confirmation":
                args = json.loads(call.function.arguments)
                confirmed = bool(args.get("confirmed"))
    spoken_reply = (message.content or "").strip()
    if not spoken_reply:
        spoken_reply = "Done." if confirmed else "Okay, no changes made."
    return confirmed, spoken_reply


def understand_and_respond(user_text, vehicle_state, personality="professional"):
    system_prompt = BASE_SYSTEM_PROMPT + "\n\n" + PERSONALITY_TONE.get(personality, PERSONALITY_TONE["professional"])
    messages = [{"role": "system", "content": system_prompt}] + _history_as_messages() + [{"role": "user", "content": user_text}]
    client = get_groq_client()
    response = client.chat.completions.create(model=LLM_MODEL, messages=messages, tools=[VEHICLE_TOOL], tool_choice="auto")
    message = response.choices[0].message

    pending_action = None
    if message.tool_calls:
        for call in message.tool_calls:
            if call.function.name == "vehicle_command":
                args = json.loads(call.function.arguments)
                pending_action = {"action": args.get("action"), "value": args.get("value")}

    spoken_text = (message.content or "").strip()
    if not spoken_text:
        spoken_text = "Done." if pending_action is None else "Do you confirm?"

    if pending_action is not None:
        _pending_command["data"] = pending_action
        _pending_command["question"] = spoken_text

    _append_history(user_text, spoken_text)
    return spoken_text, vehicle_state, pending_action is not None


def confirm_pending_command(confirmation_text, vehicle_state):
    pending = _pending_command["data"]
    if pending is None:
        return vehicle_state, "There is no pending command to confirm.", False

    confirmed, spoken_reply = _judge_yes_no(confirmation_text, _pending_command["question"])
    if confirmed:
        vehicle_state = dispatch_command(pending["action"], pending["value"], vehicle_state)

    _append_history(confirmation_text, spoken_reply)
    _pending_command["data"] = None
    _pending_command["question"] = None
    return vehicle_state, spoken_reply, confirmed


def has_pending_command():
    return _pending_command["data"] is not None


def speak_to_bytes(text, voice_id=ELEVENLABS_VOICE_ID, language_code="en"):
    """Same as the notebook's speak(), but returns raw mp3 bytes instead of writing to disk —
    server.py bundles them as base64 for the browser's <audio> element."""
    client = get_elevenlabs_client()
    audio = client.text_to_speech.convert(
        text=text, voice_id=voice_id, model_id=ELEVENLABS_MODEL_ID,
        language_code=language_code, output_format="mp3_44100_128",
    )
    return b"".join(audio) if not isinstance(audio, (bytes, bytearray)) else audio
