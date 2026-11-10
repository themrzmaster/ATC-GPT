"""
Client for the X-Plane side (PI_LLMPlays.py), reached through `scripts/box.sh tunnel`.

Everything is turn-based: the sim is paused except inside act(seconds).
"""

import base64
import io
import os
from typing import Optional

import requests

DEFAULT_URL = os.environ.get("LLM_PLAYS_URL", "http://127.0.0.1:5066")
C172_PATH = "Aircraft/Laminar Research/Cessna 172 SP/Cessna_172SP.acf"


class SimError(RuntimeError):
    pass


class XPlaneSim:
    def __init__(self, url: str = DEFAULT_URL):
        self.url = url.rstrip("/")

    def _get(self, path: str, timeout: float = 60):
        r = requests.get(self.url + path, timeout=timeout)
        return self._check(r)

    def _post(self, path: str, body: Optional[dict] = None, timeout: float = 120):
        r = requests.post(self.url + path, json=body or {}, timeout=timeout)
        return self._check(r)

    @staticmethod
    def _check(r):
        try:
            data = r.json()
        except ValueError as e:
            raise SimError(f"HTTP {r.status_code}: {r.text[:200]}") from e
        if r.status_code != 200 or (isinstance(data, dict) and data.get("error")):
            raise SimError(data.get("error") if isinstance(data, dict) else str(data))
        return data

    # -- API -------------------------------------------------------------------
    def health(self) -> bool:
        try:
            return bool(self._get("/health", timeout=5).get("ok"))
        except Exception:  # noqa: BLE001
            return False

    def state(self) -> dict:
        return self._get("/state")

    def look(self, max_width: int = 1280, quality: int = 80) -> tuple[dict, bytes]:
        """(state, jpeg bytes) - screenshot downscaled for the model."""
        from PIL import Image
        err = None
        for _ in range(3):   # X-Plane can still be writing the PNG: retry a fresh screenshot
            d = self._get("/look", timeout=90)
            try:
                im = Image.open(io.BytesIO(base64.b64decode(d["png_b64"])))
                im.load()
                im = im.convert("RGB")
                break
            except OSError as e:
                err = e
        else:
            raise SimError(f"screenshot unreadable after 3 tries: {err}")
        if im.width > max_width:
            im = im.resize((max_width, round(im.height * max_width / im.width)))
        from llm_plays.overlays import hide_sim_overlays
        im, _ = hide_sim_overlays(im)          # the pilot never sees X-Plane's notifications / PAUSED toolbar
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=quality)
        return d["state"], buf.getvalue()

    def act(self, seconds: float, commands: Optional[list] = None, **controls) -> dict:
        body = {k: v for k, v in controls.items() if v is not None}
        body["seconds"] = seconds
        if commands:
            body["commands"] = list(commands)
        return self._post("/act", body, timeout=seconds * 4 + 90)

    def wait_ready(self, settle_s: float = 20, timeout_s: float = 240):
        """After an aircraft load: wait for /state to answer, then let the sim settle."""
        import time
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            try:
                self._get("/state", timeout=10)
                time.sleep(settle_s)
                self._get("/state", timeout=10)
                return
            except Exception:  # noqa: BLE001
                time.sleep(3)
        raise SimError("sim not ready after aircraft load")

    def pause(self):
        return self._post("/pause")

    def place(self, lat: float, lon: float, ele_m: float, heading: float, speed_ms: float):
        return self._post("/place", {"lat": lat, "lon": lon, "ele_m": ele_m, "heading": heading,
                                     "speed_ms": speed_ms})

    def set(self, values: dict):
        """Setup helper: write datarefs {path: value | [values]}."""
        return self._post("/set", values)

    def get(self, paths: list) -> dict:
        return self._post("/get", {"paths": list(paths)})

    def start_engine(self, seconds: float, **controls) -> dict:
        """
        X-Plane auto-start, fixed for the C172: auto-start moves the fuel selector to the CENTER tank,
        which the 172 doesn't have (fuel is in left/right), so the engine cranks forever. Put it back on
        ALL after the sequence begins, then let it run.
        """
        self.act(0.5, commands=["sim/operation/auto_start"], **controls)
        self.set({"sim/cockpit2/fuel/fuel_tank_selector": 4})
        return self.act(seconds)

    def load_aircraft(self, path: str = C172_PATH):
        return self._post("/aircraft", {"path": path}, timeout=180)

    def save(self, name: str):
        return self._post("/save", {"name": name})

    def load(self, name: str) -> bool:
        """Load a saved situation; False when it doesn't exist."""
        try:
            return bool(self._post("/load", {"name": name}, timeout=180).get("loaded"))
        except SimError as e:
            if "no situation" in str(e):
                return False
            raise
