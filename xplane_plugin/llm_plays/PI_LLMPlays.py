"""
PI_LLMPlays.py - X-Plane side of "LLM plays X-Plane" (turn-based flying, in the spirit of
geohot/pi_plays_pokemon).

The sim stays PAUSED while the model thinks. The model's harness (on the Mac, llm_plays/) calls:

    GET  /health
    GET  /state                      instruments as JSON
    GET  /look                       instruments + a fresh cockpit screenshot (PNG, base64)
    POST /act   {controls..., commands: [...], seconds: N}
                                     apply controls, unpause, fly N sim-seconds, pause, return state
    POST /pause | /unpause
    POST /place {lat, lon, ele_m, heading, speed_ms}      put the aircraft somewhere (mission setup)
    POST /aircraft {path}            load an aircraft (relative to the X-Plane folder)
    POST /engine_on                  setup helper: battery, magnetos, fuel, engine running
    POST /tune_ils {lat, lon, course_true}   setup: tune NAV1 to the runway's ILS from X-Plane's nav database
    POST /save {name} | /load {name} situation files (campaign persistence)

Controls (all optional, persist until changed): pitch, roll, yaw (-1..1, yoke/pedals, +pitch = pull
back / nose up), trim (-1..1 elevator trim, + = nose up), throttle, mixture (0..1), flaps (0..1 handle), brakes (0..1 toe brakes),
parking_brake (0..1). commands = X-Plane command paths fired once (e.g. sim/starters/engage_starter_1).

Binds 127.0.0.1 only: reach it through an SSH tunnel (scripts/box.sh tunnel). Every XPLM call runs on
the main thread: HTTP threads queue jobs, the flight loop executes them.
"""

import base64
import glob
import json
import os
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from XPPython3 import xp

PORT = int(os.environ.get("LLM_PLAYS_PORT", "5066"))

READ = {
    "lat": ("sim/flightmodel/position/latitude", "d"),
    "lon": ("sim/flightmodel/position/longitude", "d"),
    "elevation_m": ("sim/flightmodel/position/elevation", "d"),
    "agl_m": ("sim/flightmodel/position/y_agl", "f"),
    "ias_kt": ("sim/flightmodel/position/indicated_airspeed", "f"),
    "gs_ms": ("sim/flightmodel/position/groundspeed", "f"),
    "vs_fpm": ("sim/flightmodel/position/vh_ind_fpm", "f"),
    "heading_true": ("sim/flightmodel/position/psi", "f"),
    "heading_mag": ("sim/flightmodel/position/mag_psi", "f"),
    "track_true": ("sim/flightmodel/position/hpath", "f"),
    "pitch_deg": ("sim/flightmodel/position/theta", "f"),
    "bank_deg": ("sim/flightmodel/position/phi", "f"),
    "alt_indicated_ft": ("sim/cockpit2/gauges/indicators/altitude_ft_pilot", "f"),
    "on_ground": ("sim/flightmodel/failures/onground_any", "i"),
    "crashed": ("sim/flightmodel2/misc/has_crashed", "i"),
    "paused": ("sim/time/paused", "i"),
    "sim_time_s": ("sim/time/total_flight_time_sec", "f"),
    "throttle": ("sim/cockpit2/engine/actuators/throttle_ratio_all", "f"),
    "mixture": ("sim/cockpit2/engine/actuators/mixture_ratio_all", "f"),
    "flaps_handle": ("sim/cockpit2/controls/flap_handle_request_ratio", "f"),
    "trim": ("sim/cockpit2/controls/elevator_trim", "f"),
    "flaps_deployed": ("sim/flightmodel2/controls/flap1_deploy_ratio", "f"),
    "parking_brake": ("sim/cockpit2/controls/parking_brake_ratio", "f"),
    "stall_warning": ("sim/cockpit2/annunciators/stall_warning", "i"),
    "stall_flag": ("sim/flightmodel/failures/stallwarning", "i"),          # what drives the horn
    "stall_ratio": ("sim/cockpit2/annunciators/stall_warning_ratio", "f"),
    "roll_rate_dps": ("sim/flightmodel/position/P", "f"),
    "pitch_rate_dps": ("sim/flightmodel/position/Q", "f"),
    "yaw_rate_dps": ("sim/flightmodel/position/R", "f"),
    "fuel_kg": ("sim/flightmodel/weight/m_fuel_total", "f"),
    "wind_from_deg": ("sim/weather/aircraft/wind_now_direction_degt", "f"),
    "wind_ms": ("sim/weather/aircraft/wind_now_speed_msc", "f"),
    "com1_khz": ("sim/cockpit2/radios/actuators/com1_frequency_hz_833", "i"),
    # what the pilot's gauges actually show (lag, instrument errors), for the model-facing readout
    "ias_ind": ("sim/cockpit2/gauges/indicators/airspeed_kts_pilot", "f"),
    "vs_ind": ("sim/cockpit2/gauges/indicators/vvi_fpm_pilot", "f"),
    "hdg_ind": ("sim/cockpit2/gauges/indicators/heading_electric_deg_mag_pilot", "f"),
    "pitch_ind": ("sim/cockpit2/gauges/indicators/pitch_AHARS_deg_pilot", "f"),
    "roll_ind": ("sim/cockpit2/gauges/indicators/roll_AHARS_deg_pilot", "f"),
    "turn_ind": ("sim/cockpit2/gauges/indicators/turn_rate_heading_deg_pilot", "f"),
    "slip_ind": ("sim/cockpit2/gauges/indicators/slip_deg", "f"),
    # NAV1 (ILS) as the pilot sees it
    "nav1_hdef": ("sim/cockpit2/radios/indicators/nav1_hdef_dots_pilot", "f"),
    "nav1_vdef": ("sim/cockpit2/radios/indicators/nav1_vdef_dots_pilot", "f"),
    "nav1_has_loc": ("sim/cockpit2/radios/indicators/nav1_display_horizontal", "i"),
    "nav1_has_gs": ("sim/cockpit2/radios/indicators/nav1_display_vertical", "i"),
    "nav1_has_dme": ("sim/cockpit2/radios/indicators/nav1_has_dme", "i"),
    "nav1_dme_nm": ("sim/cockpit2/radios/indicators/nav1_dme_distance_nm", "f"),
    "nav1_freq": ("sim/cockpit2/radios/actuators/nav1_frequency_hz", "i"),
    "marker_outer": ("sim/cockpit2/radios/indicators/outer_marker_lit", "i"),
    "marker_middle": ("sim/cockpit2/radios/indicators/middle_marker_lit", "i"),
    "marker_inner": ("sim/cockpit2/radios/indicators/inner_marker_lit", "i"),
}
READ_ARRAY = {
    "engine_rpm": ("sim/cockpit2/engine/indicators/engine_speed_rpm", "fa"),
    "engine_running": ("sim/flightmodel/engine/ENGN_running", "ia"),
}
WRITE = {
    "pitch": "sim/joystick/yoke_pitch_ratio",
    "roll": "sim/joystick/yoke_roll_ratio",
    "yaw": "sim/joystick/yoke_heading_ratio",
    "throttle": "sim/cockpit2/engine/actuators/throttle_ratio_all",
    "mixture": "sim/cockpit2/engine/actuators/mixture_ratio_all",
    "flaps": "sim/cockpit2/controls/flap_handle_request_ratio",
    "trim": "sim/cockpit2/controls/elevator_trim",
    "parking_brake": "sim/cockpit2/controls/parking_brake_ratio",
    "brakes": ("sim/cockpit2/controls/left_brake_ratio", "sim/cockpit2/controls/right_brake_ratio"),
}
LIMITS = {"pitch": (-1, 1), "roll": (-1, 1), "yaw": (-1, 1), "trim": (-1, 1), "throttle": (0, 1), "mixture": (0, 1),
          "flaps": (0, 1), "parking_brake": (0, 1), "brakes": (0, 1)}


class Job:
    def __init__(self, fn, *args):
        self.fn, self.args = fn, args
        self.done = threading.Event()
        self.result = None
        self.error = None


class PythonInterface:
    def __init__(self):
        self.Name = "LLM Plays X-Plane"
        self.Sig = "llmplays.xppython3"
        self.Desc = "Turn-based control API for an LLM pilot (sim paused while the model thinks)"
        self.jobs: "queue.Queue[Job]" = queue.Queue()
        self.refs = {}
        self.held = {}              # controls to re-apply every frame while flying
        self.flying = None          # active /act: {"until": sim_time, "job": Job, "start": ...}
        self._prev_air = None       # (on_ground, vs_fpm) of the previous frame, for touchdown capture
        self.events = []            # touchdowns etc. since the last /state or /act result
        self.server = None
        self._thread = None
        self.loop = None

    # -- lifecycle -----------------------------------------------------------
    def XPluginStart(self):
        return self.Name, self.Sig, self.Desc

    def XPluginEnable(self):
        for name, (path, _kind) in {**READ, **READ_ARRAY}.items():
            self.refs[name] = xp.findDataRef(path)
        for name, path in WRITE.items():
            paths = path if isinstance(path, tuple) else (path,)
            self.refs["w_" + name] = [xp.findDataRef(p) for p in paths]
        self.refs["acf_file"] = xp.findDataRef("sim/aircraft/view/acf_ICAO")
        self.refs["override_joystick"] = xp.findDataRef("sim/operation/override/override_joystick")
        self.refs["override_toe_brakes"] = xp.findDataRef("sim/operation/override/override_toe_brakes")
        self.loop = xp.createFlightLoop(self._flight_loop)
        xp.scheduleFlightLoop(self.loop, -1)
        ThreadingHTTPServer.allow_reuse_address = True
        self.server = ThreadingHTTPServer(("127.0.0.1", PORT), self._handler())
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()
        xp.log(f"LLMPlays: API on 127.0.0.1:{PORT}")
        return 1

    def XPluginDisable(self):
        xp.log("LLMPlays: XPluginDisable")
        if self.server:
            self.server.shutdown()
            self.server = None
        if self.loop:
            xp.destroyFlightLoop(self.loop)
            self.loop = None
        self._release_controls()

    def XPluginStop(self):
        pass

    def _serve(self):
        try:
            self.server.serve_forever()
        except BaseException as e:  # noqa: BLE001
            xp.log(f"LLMPlays: HTTP server stopped: {type(e).__name__}: {e}")
        xp.log("LLMPlays: HTTP server thread exited")

    def XPluginReceiveMessage(self, from_who, message, param):
        if message in (101, 102, 103):   # PLANE_CRASHED / PLANE_LOADED / AIRPORT_LOADED
            xp.log(f"LLMPlays: message {message} param {param}; server alive: "
                   f"{bool(self.server)} thread alive: {self._thread.is_alive() if self._thread else None}")

    # -- main thread ---------------------------------------------------------
    def _flight_loop(self, since, elapsed, counter, refcon):
        try:
            if self.held:
                self._apply(self.held)
            self._watch_touchdown()
            if self.flying:
                st = self._read_state()
                f = self.flying
                self._sample(f, st, since)
                if st["sim_time_s"] >= f["until"] or st["crashed"] or time.time() > f["wall_deadline"]:
                    xp.commandOnce(xp.findCommand("sim/operation/pause_on"))
                    self.flying = None
                    st = self._read_state()
                    st["flown_s"] = round(st["sim_time_s"] - f["start"], 2)
                    st["during"] = self._envelope(f, st)
                    st["track"] = f.get("track", [])
                    f["job"].result = st
                    f["job"].done.set()
            while True:
                try:
                    job = self.jobs.get_nowait()
                except queue.Empty:
                    break
                try:
                    job.result = job.fn(job, *job.args)
                except Exception as e:  # noqa: BLE001
                    job.error = f"{type(e).__name__}: {e}"
                if job.result is not _PENDING:
                    job.done.set()
        except Exception as e:  # noqa: BLE001
            xp.log(f"LLMPlays: flight loop error: {e}")
        return -1

    @staticmethod
    def _stall_horn(st: dict) -> bool:
        """The horn: X-Plane's stall flag / ratio, or ~10% above stall speed for the flap setting (C172:
        40 kt flaps 30, 48 kt clean) when the aircraft's own annunciator isn't wired."""
        if st.get("on_ground"):
            return False
        if st.get("stall_flag") or st.get("stall_warning") or (st.get("stall_ratio") or 0) > 0.5:
            return True
        vs_kt = 48 - 8 * (st.get("flaps_deployed") or 0)
        return 0 < (st.get("ias_kt") or 0) < vs_kt * 1.1

    def _sample(self, f: dict, st: dict, dt: float):
        """Per-frame envelope while flying: what a pilot would have seen/felt during the act."""
        e = f.setdefault("env", {"ias": [1e9, -1e9], "agl": [1e9, -1e9], "alt": [1e9, -1e9], "pitch": [1e9, -1e9],
                                 "bank": [1e9, -1e9], "vs": [1e9, -1e9], "horn_s": 0.0,
                                 "start": {k: st.get(k) for k in ("ias_kt", "alt_msl_ft", "heading_mag")}})
        for key, name in (("ias", "ias_kt"), ("agl", "agl_ft"), ("alt", "alt_indicated_ft"), ("pitch", "pitch_deg"), ("bank", "bank_deg"),
                          ("vs", "vs_fpm")):
            v = st.get(name)
            if v is not None:
                e[key][0] = min(e[key][0], v)
                e[key][1] = max(e[key][1], v)
        if not st.get("paused") and self._stall_horn(st):
            e["horn_s"] += max(0.0, min(float(dt), 0.5))
        # replay track at ~10 Hz: [t, lat, lon, alt_msl_ft, agl_ft, ias, gs, vs, hdg, pitch, bank, thr, flaps,
        #                          yoke_pitch, yoke_roll, on_ground]
        t = st.get("sim_time_s", 0.0)
        tr = f.setdefault("track", [])
        if not tr or t - tr[-1][0] >= 0.1:
            r = lambda v, n=1: round(float(v or 0), n)  # noqa: E731
            tr.append([r(t, 2), round(st.get("lat", 0), 6), round(st.get("lon", 0), 6), r(st.get("alt_msl_ft")),
                       r(st.get("agl_ft")), r(st.get("ias_kt")), r(st.get("gs_kt")), r(st.get("vs_fpm"), 0),
                       r(st.get("heading_true")), r(st.get("pitch_deg")), r(st.get("bank_deg")),
                       r(st.get("throttle"), 2), r(st.get("flaps_deployed"), 2),
                       r(self.held.get("pitch"), 3), r(self.held.get("roll"), 3), int(st.get("on_ground") or 0)])

    @staticmethod
    def _envelope(f: dict, st: dict) -> dict:
        e = f.get("env")
        if not e:
            return {}
        secs = max(st["sim_time_s"] - f["start"], 0.01)
        s0 = e["start"]
        dh = ((st.get("heading_mag", 0) - (s0.get("heading_mag") or 0) + 180) % 360) - 180
        r = lambda v: round(v, 1)  # noqa: E731
        return {"seconds": r(secs), "ias_min": r(e["ias"][0]), "ias_max": r(e["ias"][1]),
                "agl_min": r(e["agl"][0]), "alt_min": r(e["alt"][0]), "alt_max": r(e["alt"][1]),
                "pitch_min": r(e["pitch"][0]), "pitch_max": r(e["pitch"][1]),
                "bank_min": r(e["bank"][0]), "bank_max": r(e["bank"][1]), "vs_min": r(e["vs"][0]),
                "vs_max": r(e["vs"][1]), "stall_horn_s": r(e["horn_s"]),
                "ias_trend_kt_s": r((st.get("ias_kt", 0) - (s0.get("ias_kt") or 0)) / secs),
                "turn_rate_dps": r(dh / secs)}

    def _watch_touchdown(self):
        """Frame-exact touchdown capture: sink rate / bank / speed at the moment the wheels touch."""
        g = lambda n, k="f": (xp.getDatai if k == "i" else xp.getDataf)(self.refs[n])  # noqa: E731
        on_ground = g("on_ground", "i")
        vs = g("vs_fpm")
        if self._prev_air is not None:
            was_ground, prev_vs = self._prev_air
            if not was_ground and on_ground:
                self.events.append({"type": "touchdown", "sink_fpm": round(prev_vs, 1),
                                    "bank_deg": round(g("bank_deg"), 2), "pitch_deg": round(g("pitch_deg"), 2),
                                    "ias_kt": round(g("ias_kt"), 1), "lat": xp.getDatad(self.refs["lat"]),
                                    "lon": xp.getDatad(self.refs["lon"]), "sim_time_s": round(g("sim_time_s"), 2)})
        self._prev_air = (on_ground, vs)

    def _read_state(self) -> dict:
        out = {}
        for name, (_p, kind) in READ.items():
            ref = self.refs.get(name)
            if not ref:
                continue
            out[name] = xp.getDatad(ref) if kind == "d" else xp.getDatai(ref) if kind == "i" else xp.getDataf(ref)
        for name, (_p, kind) in READ_ARRAY.items():
            ref = self.refs.get(name)
            if not ref:
                continue
            vals = []
            (xp.getDatavi if kind == "ia" else xp.getDatavf)(ref, vals, 0, 1)
            out[name] = vals[0] if vals else None
        out["gs_kt"] = out.get("gs_ms", 0) * 1.94384
        out["agl_ft"] = out.get("agl_m", 0) * 3.28084
        out["alt_msl_ft"] = out.get("elevation_m", 0) * 3.28084
        out["wind_kt"] = out.get("wind_ms", 0) * 1.94384
        try:
            out["aircraft_file"] = xp.getDatas(self.refs["acf_file"]) if self.refs.get("acf_file") else None
        except Exception:  # noqa: BLE001
            out["aircraft_file"] = None
        out["stall_horn"] = self._stall_horn(out)
        out["held_controls"] = dict(self.held)
        out["events"], self.events = self.events, []
        return out

    def _apply(self, controls: dict):
        xp.setDatai(self.refs["override_joystick"], 1)
        if "brakes" in controls:
            xp.setDatai(self.refs["override_toe_brakes"], 1)
        for name, val in controls.items():
            for ref in self.refs.get("w_" + name, []):
                if ref:
                    xp.setDataf(ref, float(val))

    def _release_controls(self):
        for key in ("override_joystick", "override_toe_brakes"):
            if self.refs.get(key):
                xp.setDatai(self.refs[key], 0)

    # -- jobs (run on the main thread) --------------------------------------
    def job_state(self, job):
        return self._read_state()

    def job_act(self, job, body):
        controls = {}
        for k, (lo, hi) in LIMITS.items():
            if k in body and body[k] is not None:
                controls[k] = max(lo, min(hi, float(body[k])))
        self.held.update(controls)
        self._apply(self.held)
        fired = []
        for c in body.get("commands") or []:
            ref = xp.findCommand(str(c))
            if ref:
                xp.commandOnce(ref)
                fired.append(c)
        seconds = max(0.0, min(float(body.get("seconds", 2.0)), 60.0))
        if seconds == 0:
            st = self._read_state()
            st["commands_fired"] = fired
            return st
        st = self._read_state()
        self.flying = {"until": st["sim_time_s"] + seconds, "start": st["sim_time_s"], "job": job,
                       "wall_deadline": time.time() + seconds * 4 + 30}
        xp.commandOnce(xp.findCommand("sim/operation/pause_off"))
        return _PENDING

    def job_pause(self, job, on: bool):
        xp.commandOnce(xp.findCommand("sim/operation/pause_on" if on else "sim/operation/pause_off"))
        return {"paused": on}

    def job_screenshot(self, job):
        xp.commandOnce(xp.findCommand("sim/operation/screenshot"))
        return {"requested_at": time.time()}

    def job_place(self, job, body):
        xp.placeUserAtLocation(float(body["lat"]), float(body["lon"]), float(body["ele_m"]),
                               float(body.get("heading", 0)), float(body.get("speed_ms", 0)))
        self.held = {}
        return {"placed": True}

    def job_engine_on(self, job):
        """Setup helper (not a model tool): battery, both magnetos, fuel on all tanks, engine running."""
        f = xp.findDataRef
        for path, val in (("sim/cockpit2/electrical/battery_on", 1), ("sim/cockpit2/engine/actuators/ignition_on", 3),
                          ("sim/cockpit2/engine/actuators/ignition_key", 3), ("sim/flightmodel/engine/ENGN_running", 1)):
            ref = f(path)
            if ref:
                xp.setDatavi(ref, [val], 0, 1)
        ref = f("sim/cockpit2/fuel/fuel_tank_selector")
        if ref:
            xp.setDatai(ref, 4)
        ref = f("sim/cockpit2/engine/actuators/mixture_ratio_all")
        if ref:
            xp.setDataf(ref, 1.0)
        return {"engine_on": True}

    def job_tune_ils(self, job, body):
        """Setup: find the runway's localizer in X-Plane's nav database and tune NAV1 to it.
        body: lat/lon of the threshold, course_true of the runway."""
        import math
        lat, lon, crs = float(body["lat"]), float(body["lon"]), float(body["course_true"])
        best = None
        for nav_type in (xp.Nav_ILS, xp.Nav_Localizer):
            ref = xp.findFirstNavAidOfType(nav_type)
            last = xp.findLastNavAidOfType(nav_type)
            while ref != xp.NAV_NOT_FOUND:
                info = xp.getNavAidInfo(ref)
                dn = (info.latitude - lat) * 60.0
                de = (info.longitude - lon) * 60.0 * math.cos(math.radians(lat))
                dist = math.hypot(dn, de)
                dh = abs((info.heading - crs + 180) % 360 - 180)
                if dist < 5.0 and dh < 15 and (best is None or dist < best[0]):
                    best = (dist, info, nav_type)
                if ref == last:
                    break
                ref = xp.getNextNavAid(ref)
        if not best:
            return {"tuned": False, "error": "no localizer for this runway in X-Plane's nav database"}
        _, info, nav_type = best
        xp.setDatai(xp.findDataRef("sim/cockpit2/radios/actuators/nav1_frequency_hz"), int(info.frequency))
        return {"tuned": True, "id": info.navAidID, "name": info.name, "freq": info.frequency / 100.0,
                "course_true": round(info.heading, 1), "has_gs": nav_type == xp.Nav_ILS}

    def job_set(self, job, body):
        """Setup/debug helper (not a model tool): write datarefs {path: number | [numbers]}."""
        done = {}
        for path, val in body.items():
            ref = xp.findDataRef(path)
            if not ref:
                done[path] = "not found"
                continue
            t = xp.getDataRefTypes(ref)
            if isinstance(val, list):
                (xp.setDatavi if t & 16 else xp.setDatavf)(ref, [int(v) if t & 16 else float(v) for v in val], 0, len(val))
            elif t & 1:
                xp.setDatai(ref, int(val))
            elif t & 8:
                xp.setDatad(ref, float(val))
            else:
                xp.setDataf(ref, float(val))
            done[path] = val
        return done

    def job_get(self, job, paths):
        """Setup/debug helper: read datarefs (arrays: first 4 values)."""
        out = {}
        for path in paths:
            ref = xp.findDataRef(path)
            if not ref:
                out[path] = "not found"
                continue
            t = xp.getDataRefTypes(ref)
            if t & 16:
                v = []
                xp.getDatavi(ref, v, 0, 4)
                out[path] = v
            elif t & 8 and not t & 32:
                v = []
                xp.getDatavf(ref, v, 0, 4)
                out[path] = [round(x, 3) for x in v]
            elif t & 32:
                out[path] = xp.getDatas(ref)
            elif t & 1:
                out[path] = xp.getDatai(ref)
            elif t & 4:
                out[path] = xp.getDatad(ref)
            else:
                out[path] = round(xp.getDataf(ref), 3)
        return out

    def job_aircraft(self, job, path):
        xp.setUsersAircraft(path)
        self.held = {}
        return {"aircraft": path}

    def job_save(self, job, name):
        path = _sit_path(name)
        xp.saveDataFile(xp.DataFile_Situation, path)
        return {"saved": path}

    def job_load(self, job, name):
        path = _sit_path(name)
        if not os.path.exists(os.path.join(xp.getSystemPath(), path)):
            return {"loaded": False, "error": f"no situation {path}"}
        xp.loadDataFile(xp.DataFile_Situation, path)
        self.held = {}
        return {"loaded": path}

    # -- HTTP --------------------------------------------------------------
    def run(self, fn, *args, timeout=120):
        job = Job(fn, *args)
        self.jobs.put(job)
        if not job.done.wait(timeout):
            raise TimeoutError("sim did not finish the job (paused on a menu? crashed dialog?)")
        if job.error:
            raise RuntimeError(job.error)
        return job.result

    def screenshot_png(self) -> bytes:
        before = _newest_shot()
        self.run(PythonInterface.job_screenshot.__get__(self))
        deadline = time.time() + 10
        while time.time() < deadline:
            shot = _newest_shot()
            if shot and shot != before and os.path.getsize(shot) > 0:
                size = -1                # wait until X-Plane has finished writing (size stable)
                while size != os.path.getsize(shot):
                    size = os.path.getsize(shot)
                    time.sleep(0.25)
                with open(shot, "rb") as fh:
                    data = fh.read()
                try:
                    os.remove(shot)      # don't fill the disk
                except OSError:
                    pass
                return data
            time.sleep(0.1)
        raise TimeoutError("screenshot not written")

    def _handler(self):
        plugin = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code, obj):
                data = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _body(self):
                n = int(self.headers.get("Content-Length") or 0)
                return json.loads(self.rfile.read(n) or b"{}") if n else {}

            def do_GET(self):
                try:
                    if self.path.startswith("/health"):
                        return self._send(200, {"ok": True, "plugin": plugin.Name})
                    if self.path.startswith("/state"):
                        return self._send(200, plugin.run(PythonInterface.job_state.__get__(plugin)))
                    if self.path.startswith("/look"):
                        png = plugin.screenshot_png()
                        st = plugin.run(PythonInterface.job_state.__get__(plugin))
                        return self._send(200, {"state": st, "png_b64": base64.b64encode(png).decode()})
                    return self._send(404, {"error": "unknown path"})
                except Exception as e:  # noqa: BLE001
                    return self._send(500, {"error": f"{type(e).__name__}: {e}"})

            def do_POST(self):
                try:
                    body = self._body()
                    p = self.path
                    if p.startswith("/act"):
                        return self._send(200, plugin.run(PythonInterface.job_act.__get__(plugin), body,
                                                          timeout=float(body.get("seconds", 2)) * 4 + 60))
                    if p.startswith("/pause"):
                        return self._send(200, plugin.run(PythonInterface.job_pause.__get__(plugin), True))
                    if p.startswith("/unpause"):
                        return self._send(200, plugin.run(PythonInterface.job_pause.__get__(plugin), False))
                    if p.startswith("/place"):
                        return self._send(200, plugin.run(PythonInterface.job_place.__get__(plugin), body))
                    if p.startswith("/tune_ils"):
                        return self._send(200, plugin.run(PythonInterface.job_tune_ils.__get__(plugin), body))
                    if p.startswith("/get"):
                        return self._send(200, plugin.run(PythonInterface.job_get.__get__(plugin), body.get("paths", [])))
                    if p.startswith("/set"):
                        return self._send(200, plugin.run(PythonInterface.job_set.__get__(plugin), body))
                    if p.startswith("/engine_on"):
                        return self._send(200, plugin.run(PythonInterface.job_engine_on.__get__(plugin)))
                    if p.startswith("/aircraft"):
                        return self._send(200, plugin.run(PythonInterface.job_aircraft.__get__(plugin), body["path"]))
                    if p.startswith("/save"):
                        return self._send(200, plugin.run(PythonInterface.job_save.__get__(plugin), body["name"]))
                    if p.startswith("/load"):
                        return self._send(200, plugin.run(PythonInterface.job_load.__get__(plugin), body["name"]))
                    return self._send(404, {"error": "unknown path"})
                except Exception as e:  # noqa: BLE001
                    return self._send(500, {"error": f"{type(e).__name__}: {e}"})
        return H


_PENDING = object()


def _sit_path(name: str) -> str:
    safe = "".join(c for c in name if c.isalnum() or c in "-_") or "campaign"
    return os.path.join("Output", "situations", f"llmplays_{safe}.sit")


def _newest_shot():
    root = xp.getSystemPath()
    files = []
    for d in (os.path.join(root, "Output", "screenshots"), root):
        files += glob.glob(os.path.join(d, "*.png"))
    return max(files, key=os.path.getmtime) if files else None
