

import queue
import numpy as np
import carla


# ---------------------------------------------------------------------------
# 1. Connection + vehicle/camera setup
# ---------------------------------------------------------------------------

def connect(host="127.0.0.1", port=2000, timeout=10.0, synchronous=True, fixed_delta=0.05):
    """
    Connect to a CARLA server already running on the company machine
    (./CarlaUE4.sh or CarlaUE4.exe must be started separately first).

    synchronous=True is strongly recommended: the client steps the sim
    with world.tick() instead of it running in realtime, so your (slower)
    Python pipeline can never "fall behind" a frame -- it just waits for
    tick() to return before pulling the next image. Determinism is a nice
    side effect too.
    """
    client = carla.Client(host, port)
    client.set_timeout(timeout)
    world = client.get_world()

    if synchronous:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = fixed_delta
        world.apply_settings(settings)

    return client, world


def spawn_vehicle_with_camera(world, vehicle_bp_name="vehicle.tesla.model3",
                               image_size=(224, 224), fov=90):
    """
    Spawns an ego vehicle plus one forward-facing RGB camera matching your
    classifier's input size, and wires the camera up to a thread-safe queue
    so the main loop can pull the newest frame synchronously each tick.
    """
    bp_lib = world.get_blueprint_library()
    vehicle_bp = bp_lib.find(vehicle_bp_name)
    spawn_point = world.get_map().get_spawn_points()[0]
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)

    camera_bp = bp_lib.find("sensor.camera.rgb")
    camera_bp.set_attribute("image_size_x", str(image_size[0]))
    camera_bp.set_attribute("image_size_y", str(image_size[1]))
    camera_bp.set_attribute("fov", str(fov))
    camera_transform = carla.Transform(carla.Location(x=1.5, z=1.6))
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

    image_q = queue.Queue()
    camera.listen(image_q.put)

    return vehicle, camera, image_q


def carla_image_to_array(carla_image):
    """carla.Image -> HxWx3 uint8 RGB numpy array (drops the alpha channel)."""
    arr = np.frombuffer(carla_image.raw_data, dtype=np.uint8)
    arr = arr.reshape((carla_image.height, carla_image.width, 4))
    return arr[:, :, :3][:, :, ::-1]  # BGRA -> RGB


def tick_and_get_frame(world, image_q, timeout=2.0):
    """
    Advances the simulation exactly one step and returns the camera frame
    for that step as a numpy array. Call this once per pipeline iteration
    instead of your old "upload from disk" cell.
    """
    world.tick()
    carla_image = image_q.get(timeout=timeout)
    return carla_image_to_array(carla_image)


# ---------------------------------------------------------------------------
# 2. Ground-truth perception functions
# ---------------------------------------------------------------------------

def get_weather_label(world, detailed=False):
    """
    Ground-truth weather instead of running the .tflite classifier.
    Your notebook's apply_overrides() only checks `weather_condition ==
    "adverse"`, so this returns "clear" or "adverse" by default to plug
    in directly. Pass detailed=True to also get the raw sim reading back
    for logging (fog/rain/overcast/clear), e.g. for eval-mode comparisons
    against your weather model's own label set.
    """
    w = world.get_weather()
    if w.fog_density > 30:
        raw = "fog"
    elif w.precipitation > 20 or w.wetness > 40:
        raw = "rain"
    elif w.cloudiness > 60:
        raw = "overcast"
    else:
        raw = "clear"

    bucket = "clear" if raw == "clear" else "adverse"
    return (bucket, raw) if detailed else bucket


def is_offroad(world_map, vehicle):
    """
    Ground truth off-road check. Replaces detect_offroad(). Returns
    (is_offroad: bool, confidence: float) to match your existing tuple
    shape -- confidence is always 1.0 since this isn't a model guess.
    """
    loc = vehicle.get_location()
    wp = world_map.get_waypoint(loc, project_to_road=False)
    if wp is None:
        return True, 1.0
    if wp.lane_type != carla.LaneType.Driving:
        return True, 1.0
    return False, 1.0


def get_speed_limit_kmh(vehicle):
    """
    Ground-truth posted speed limit at the vehicle's current position.
    CARLA already resolves the nearest applicable speed-limit sign for
    you -- replaces the traffic-sign YOLO model entirely for this purpose.
    """
    return vehicle.get_speed_limit()


def get_traffic_light_state(vehicle):
    """
    Returns "Red" / "Yellow" / "Green" / None (None = not at a light).
    Replaces sign-model red-light detection.
    """
    if not vehicle.is_at_traffic_light():
        return None
    tl = vehicle.get_traffic_light()
    return str(tl.get_state()) if tl else None


def get_crosswalk_ahead(world_map, vehicle, lookahead_m=15.0):
    """
    True if a crosswalk boundary point falls within lookahead_m of the
    vehicle along its current heading. Replaces the crosswalk-ahead half
    of detect_pedestrian_crosswalk().
    """
    vt = vehicle.get_transform()
    forward = vt.get_forward_vector()
    loc = vt.location
    for pt in world_map.get_crosswalks():
        dx, dy = pt.x - loc.x, pt.y - loc.y
        dist = (dx**2 + dy**2) ** 0.5
        if dist > lookahead_m:
            continue
        # is it roughly ahead, not behind?
        if dx * forward.x + dy * forward.y > 0:
            return True
    return False


def get_nearest_pedestrian(world, vehicle, close_m=8.0, far_m=20.0):
    """
    Ground-truth pedestrian proximity. Replaces the pedestrian-close/far
    heuristic in detect_pedestrian_crosswalk() with real distances instead
    of "box is big and low in frame".
    Returns (pedestrian_close, pedestrian_far, nearest_distance_or_None).
    """
    ego_loc = vehicle.get_location()
    nearest = None
    for actor in world.get_actors().filter("walker.pedestrian.*"):
        d = actor.get_location().distance(ego_loc)
        if nearest is None or d < nearest:
            nearest = d
    if nearest is None:
        return False, False, None
    return nearest <= close_m, nearest <= far_m, nearest


def get_nearest_vehicle_ahead(world, vehicle, max_check_m=40.0):
    """
    Ground-truth version of detect_car_too_close(). Uses real 3D distance
    to the nearest other vehicle roughly ahead of the ego car, instead of
    "box is tall / sits low in frame". Returns (too_close, distance_or_None).
    Calibrate `too_close_m` against your own safety-distance formula
    (e.g. current speed-based stopping distance) rather than a fixed value.
    """
    vt = vehicle.get_transform()
    forward = vt.get_forward_vector()
    ego_loc = vt.location
    nearest = None
    for other in world.get_actors().filter("vehicle.*"):
        if other.id == vehicle.id:
            continue
        dx = other.get_location().x - ego_loc.x
        dy = other.get_location().y - ego_loc.y
        if dx * forward.x + dy * forward.y <= 0:
            continue  # behind us, ignore
        d = (dx**2 + dy**2) ** 0.5
        if d > max_check_m:
            continue
        if nearest is None or d < nearest:
            nearest = d
    too_close_m = 10.0  # calibrate against your speed-based safety distance
    if nearest is None:
        return False, None
    return nearest <= too_close_m, nearest


# ---------------------------------------------------------------------------
# 3. Road type — no CARLA ground-truth tag, needs a lookup table
# ---------------------------------------------------------------------------

# Fill this in per town you actually test on. `world.get_map().name` gives
# you the town name (e.g. "Town04"), and road_id comes from the waypoint.
# Simplest viable version: one label per town (fine for MVP/testing).
TOWN_ROAD_TYPE = {
    "Town01": "urbaine",
    "Town02": "urbaine",
    "Town03": "urbaine",
    "Town04": "autoroute",  # has a highway loop
    "Town05": "urbaine",
    "Town06": "autoroute",  # highway interchanges
    "Town07": "rurale",     # rural, few junctions
    "Town10HD": "urbaine",
}


def get_road_type(world, world_map, vehicle, fallback_classifier=None, frame=None):
    """
    Best-effort road type. Order of preference:
      1. Per-town lookup table (instant, zero cost)
      2. Lane-count/speed-limit heuristic (still ground truth, no model)
      3. Your trained road_classifier keras model on the live frame, if you
         pass fallback_classifier + frame in (slowest, but most flexible if
         you need per-segment accuracy beyond a town-level guess)
    """
    town = world_map.name.split("/")[-1]
    if town in TOWN_ROAD_TYPE:
        return TOWN_ROAD_TYPE[town]

    wp = world_map.get_waypoint(vehicle.get_location())
    speed_limit = vehicle.get_speed_limit()
    lane_count = wp.get_left_lane() is not None or wp.get_right_lane() is not None
    if speed_limit >= 90:
        return "autoroute"
    if speed_limit <= 50 and lane_count:
        return "urbaine"
    if not lane_count and speed_limit > 50:
        return "rurale"

    if fallback_classifier is not None and frame is not None:
        # reuse your existing keras classifier on the live frame as a
        # last resort -- see notes at top of file
        import tensorflow as tf
        img = tf.image.resize(frame, (224, 224))
        img = np.expand_dims(img, axis=0)
        pred = fallback_classifier.predict(img, verbose=0)
        return ["autoroute", "urbaine", "rurale"][int(np.argmax(pred))]

    return "urbaine"  # safe default


# ---------------------------------------------------------------------------
# 3b. Overpass API (OpenStreetMap) — for the REAL car, or a CARLA map that
#     was built from real OSM data via CARLA's OSM importer.
#
#     This does NOT work against the stock Town01-Town10 maps -- they
#     aren't tied to any real location, so Overpass has nothing to look
#     up. Only use this path if you generated your own CARLA map from a
#     real .osm extract and kept the lat/lon origin you imported it with.
# ---------------------------------------------------------------------------

OSM_HIGHWAY_TO_ROAD_TYPE = {
    "motorway": "autoroute", "motorway_link": "autoroute", "trunk": "autoroute",
    "primary": "urbaine", "secondary": "urbaine", "tertiary": "urbaine",
    "residential": "urbaine", "living_street": "urbaine", "unclassified": "urbaine",
    "track": "rurale", "unpaved": "rurale",
}


def carla_location_to_latlon(location, origin_lat, origin_lon):
    """
    Rough flat-earth conversion from a CARLA carla.Location (meters) back
    to lat/lon, ONLY valid if this map was generated from real OSM data
    with (origin_lat, origin_lon) as the import origin. Fine over the
    typical few-km extent of a CARLA map; do not use for long distances.
    """
    import math
    lat = origin_lat + (location.y / 111_320.0)
    lon = origin_lon + (location.x / (111_320.0 * math.cos(math.radians(origin_lat))))
    return lat, lon


def get_road_type_overpass(lat, lon, radius_m=20, timeout=5):
    """
    Queries the Overpass API for the nearest tagged road to (lat, lon) and
    maps its `highway` tag to autoroute/urbaine/rurale. This is the same
    function you'd call in the real car with a live GPS fix -- keeping it
    identical here is the point, so CARLA testing actually exercises the
    real-deployment code path (only meaningful with an OSM-derived map,
    see module note above).
    """
    import requests
    query = f"""
    [out:json][timeout:{timeout}];
    way(around:{radius_m},{lat},{lon})["highway"];
    out tags 1;
    """
    resp = requests.post("https://overpass-api.de/api/interpreter", data=query, timeout=timeout)
    resp.raise_for_status()
    elements = resp.json().get("elements", [])
    if not elements:
        return "urbaine"  # safe default, no road found nearby
    highway = elements[0].get("tags", {}).get("highway", "")
    return OSM_HIGHWAY_TO_ROAD_TYPE.get(highway, "urbaine")


# ---------------------------------------------------------------------------
# 4. FAST MODE -- ground truth only, builds the exact dict shape
#    classify_full() used to produce, so apply_overrides()/propose_or_announce()
#    in your notebook don't need to change at all.
# ---------------------------------------------------------------------------

def build_result_from_ground_truth(world, world_map, vehicle,
                                    get_base_speed, apply_overrides,
                                    is_damaged=False, bump_ahead=False):
    """
    Pass in your notebook's own get_base_speed and apply_overrides
    functions (cell 32) unchanged -- this just supplies their inputs
    from CARLA instead of from your CV models.
    """
    offroad, _ = is_offroad(world_map, vehicle)
    road_type = "off-road" if offroad else get_road_type(world, world_map, vehicle)

    close_ped, far_ped, _ = get_nearest_pedestrian(world, vehicle)
    crosswalk_ahead = get_crosswalk_ahead(world_map, vehicle)
    too_close, _ = get_nearest_vehicle_ahead(world, vehicle)
    weather_condition = get_weather_label(world)

    # Reconstruct a detected_signs list in the same shape your sign model
    # produced, so apply_overrides()'s existing sign-scanning loop works
    # untouched.
    detected_signs = []
    light_state = get_traffic_light_state(vehicle)
    if light_state == "Red":
        detected_signs.append({"class": "Red Light", "confidence": 1.0})
    speed_limit = get_speed_limit_kmh(vehicle)
    if speed_limit:
        detected_signs.append({"class": f"Speed Limit {int(speed_limit)}", "confidence": 1.0})

    base_speed = get_base_speed(road_type, is_damaged)
    final_speed, reason_code, reason_details = apply_overrides(
        base_speed,
        detected_signs,
        pedestrian_close=close_ped,
        pedestrian_far=far_ped,
        crosswalk_ahead=crosswalk_ahead,
        bump_ahead=bump_ahead,
        car_too_close=too_close,
        weather_condition=weather_condition,
    )

    return {
        "road_type": road_type,
        "damaged": is_damaged,
        "detected_signs": detected_signs,
        "pedestrian_close": close_ped,
        "pedestrian_far": far_ped,
        "crosswalk_ahead": crosswalk_ahead,
        "speed_bump_detected": bump_ahead,
        "car_too_close": too_close,
        "weather_condition": weather_condition,
        "base_speed": base_speed,
        "final_speed_kmh": final_speed,
        "reason_code": reason_code,
        "reason_details": reason_details,
    }


# ---------------------------------------------------------------------------
# 5. EVAL MODE -- run your real models on the frame too, log both sides,
#    so you're still validating the CV models, not just bypassing them.
# ---------------------------------------------------------------------------

def log_eval_row(frame_idx, world, world_map, vehicle, frame_path, model_result, log_rows):
    """
    Call this once per frame in eval mode, after you've saved the frame to
    disk and run your existing classify_full(frame_path) on it. Appends a
    row comparing each model output to CARLA's ground truth. Dump
    log_rows to a CSV afterward with pandas for accuracy/precision numbers.
    """
    offroad_gt, _ = is_offroad(world_map, vehicle)
    close_gt, far_gt, _ = get_nearest_pedestrian(world, vehicle)
    crosswalk_gt = get_crosswalk_ahead(world_map, vehicle)
    too_close_gt, _ = get_nearest_vehicle_ahead(world, vehicle)
    weather_gt = get_weather_label(world)

    log_rows.append({
        "frame": frame_idx,
        "road_type_model": model_result["road_type"],
        "road_type_gt": "off-road" if offroad_gt else get_road_type(world, world_map, vehicle),
        "pedestrian_close_model": model_result["pedestrian_close"],
        "pedestrian_close_gt": close_gt,
        "pedestrian_far_model": model_result["pedestrian_far"],
        "pedestrian_far_gt": far_gt,
        "crosswalk_model": model_result["crosswalk_ahead"],
        "crosswalk_gt": crosswalk_gt,
        "car_too_close_model": model_result["car_too_close"],
        "car_too_close_gt": too_close_gt,
        "weather_model": model_result["weather_condition"],
        "weather_gt": weather_gt,
    })


# ---------------------------------------------------------------------------
# 6. Cleanup helper
# ---------------------------------------------------------------------------

def shutdown(world, vehicle, camera):
    camera.stop()
    camera.destroy()
    vehicle.destroy()
    settings = world.get_settings()
    settings.synchronous_mode = False
    world.apply_settings(settings)

