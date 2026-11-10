#!/usr/bin/env python3
"""Live view of campaigns: python -m llm_plays.viewer  ->  http://127.0.0.1:8341"""

import html
import json
import sys
from pathlib import Path

from flask import Flask, abort, send_file

REPLAY = (Path(__file__).with_name("replay.html")).read_text()

ROOT = Path(__file__).resolve().parent.parent
CAMPAIGNS = ROOT / "campaigns"
app = Flask(__name__)

PAGE = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>LLM plays X-Plane</title><meta http-equiv="refresh" content="3">
<style>
:root{{--bg:#0e1116;--fg:#e6e6e6;--muted:#8b949e;--card:#161b22;--acc:#58a6ff;--ok:#3fb950;--bad:#f85149}}
body{{background:var(--bg);color:var(--fg);font:14px/1.45 ui-monospace,Menlo,monospace;margin:0;padding:16px}}
a{{color:var(--acc)}} img{{max-width:100%;border-radius:6px}} .grid{{display:grid;grid-template-columns:minmax(0,3fr) minmax(0,2fr);gap:16px}}
.card{{background:var(--card);border-radius:8px;padding:12px;overflow:auto}} pre{{white-space:pre-wrap;margin:0}}
.ev{{border-bottom:1px solid #30363d;padding:6px 0}} .k{{color:var(--muted)}} .ok{{color:var(--ok)}} .bad{{color:var(--bad)}}
@media(max-width:800px){{.grid{{grid-template-columns:1fr}}}}
</style></head><body>{body}</body></html>"""


def _events(d: Path, n: int = 40):
    p = d / "events.jsonl"
    if not p.exists():
        return []
    return [json.loads(x) for x in p.read_text().splitlines()[-n:]]


@app.route("/")
def index():
    rows = []
    for d in sorted(CAMPAIGNS.glob("*/"), key=lambda p: p.stat().st_mtime, reverse=True):
        meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else {}
        prog = json.loads((d / "progress.json").read_text()) if (d / "progress.json").exists() else {}
        rows.append(f"<div class=ev><a href='/c/{d.name}'>{html.escape(d.name)}</a> · <a href='/c/{d.name}/replay'>replay</a> "
                    f"<span class=k>{meta.get('model','')} @ {meta.get('reasoning','?')} · score {prog.get('score','-')} · "
                    f"milestones {prog.get('milestones','-')}</span></div>")
    return PAGE.format(body="<h2>LLM plays X-Plane</h2><div class=card>" + ("".join(rows) or "no campaigns") + "</div>")


@app.route("/c/<name>")
def campaign(name):
    d = CAMPAIGNS / name
    if not d.is_dir():
        abort(404)
    prog = json.loads((d / "progress.json").read_text()) if (d / "progress.json").exists() else {}
    evs = []
    for e in reversed(_events(d)):
        k = e.get("kind")
        if k == "say":
            evs.append(f"<div class=ev><span class=k>{e['turn']} thinks</span><pre>{html.escape(e['text'])}</pre></div>")
        elif k == "act":
            a = {x: y for x, y in e["args"].items() if y is not None}
            ms = f" <span class=ok>★ {', '.join(e['milestones'])}</span>" if e.get("milestones") else ""
            evs.append(f"<div class=ev><span class=k>{e['turn']} act</span> {html.escape(json.dumps(a))}{ms}"
                       f"<pre class=k>{html.escape(e['instruments'].splitlines()[1])}</pre></div>")
        elif k == "report":
            evs.append(f"<div class=ev><span class=k>{e['turn']} report [{e['status']}]</span> {html.escape(e['message'])}</div>")
        elif k in ("error", "end"):
            cls = "bad" if k == "error" else "ok"
            evs.append(f"<div class='ev {cls}'>{k}: {html.escape(json.dumps({x: y for x, y in e.items() if x not in ('t', 'kind', 'notes')})[:400])}</div>")
    body = (f"<h2><a href='/'>LLM plays X-Plane</a> / {html.escape(name)}</h2>"
            f"<p><a href='/c/{name}/replay'>▶ replay this flight</a></p>"
            f"<div class=grid><div class=card><img src='/c/{name}/latest.jpg?{len(evs)}'>"
            f"<pre>{html.escape(json.dumps(prog, indent=1))}</pre></div>"
            f"<div class=card>{''.join(evs) or 'no events yet'}</div></div>")
    return PAGE.format(body=body)


def _replay_data(d: Path) -> dict:
    """Everything the replay page needs: samples (10 Hz track, or per-act snapshots for older flights),
    runway geometry, the model's decisions and the screenshots, all on the sim-time axis."""
    import re
    sys.path.insert(0, str(ROOT))
    from llm_plays.geometry import runway_from_db
    from llm_plays.missions import MISSIONS
    meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else {}
    events = [json.loads(x) for x in (d / "events.jsonl").read_text().splitlines()] if (d / "events.jsonl").exists() else []
    tpat = re.compile(r"sim time ([\d.]+)s")

    def t_of(e):
        m = tpat.search(e.get("instruments", ""))
        return float(m.group(1)) if m else None

    samples, source = [], "track"
    if (d / "trajectory.jsonl").exists():
        samples = [json.loads(x) for x in (d / "trajectory.jsonl").read_text().splitlines()]
    else:
        source, held = "per-act snapshots (flown before 10 Hz recording)", {}
        for e in events:
            if e.get("kind") != "act" or t_of(e) is None:
                continue
            held.update({k: v for k, v in e["args"].items() if v is not None})
            st = e.get("state") or {}
            samples.append({"t": t_of(e), "lat": st.get("lat"), "lon": st.get("lon"), "alt": st.get("alt_msl_ft"),
                            "agl": st.get("agl_ft"), "ias": st.get("ias_kt"), "gs": st.get("gs_kt"),
                            "vs": st.get("vs_fpm"), "hdg": st.get("heading_true"), "pitch": st.get("pitch_deg"),
                            "bank": st.get("bank_deg"), "thr": held.get("throttle"), "flaps": held.get("flaps"),
                            "yoke_pitch": held.get("pitch"), "yoke_roll": held.get("roll"),
                            "on_ground": st.get("on_ground"), "turn": e["turn"]})
    acts, shots, says, last_t = [], [], {}, 0.0
    for e in events:
        k = e.get("kind")
        if k == "say":
            says[e["turn"]] = e["text"]
        elif k == "act":
            t = t_of(e)
            last_t = t if t is not None else last_t
            acts.append({"turn": e["turn"], "t": last_t, "args": {a: v for a, v in e["args"].items() if v is not None},
                         "say": says.get(e["turn"], ""), "milestones": e.get("milestones") or []})
        elif k == "look":
            t = t_of(e)
            shots.append({"turn": e["turn"], "t": t if t is not None else last_t, "file": e.get("shot")})
    for f in sorted((d / "shots").glob("*.jpg")):          # spectator / short-final views, by turn
        turn = int(f.stem)
        if not any(x["file"] == f.name for x in shots):
            t = next((a["t"] for a in acts if a["turn"] == turn), None)
            if t is not None:
                shots.append({"turn": turn, "t": t, "file": f.name})
    shots.sort(key=lambda x: x["t"])
    runway = None
    m = MISSIONS.get(meta.get("mission", ""))
    if m:
        rw = runway_from_db(*m["destination"])
        far = rw.from_local(rw.length_ft, 0)
        runway = {"ident": rw.ident, "airport": rw.airport, "lat": rw.threshold_lat, "lon": rw.threshold_lon,
                  "far_lat": far[0], "far_lon": far[1], "heading": rw.heading_true, "length_ft": rw.length_ft,
                  "width_ft": rw.width_ft, "elev_ft": rw.elevation_ft}
    result = json.loads((d / "result.json").read_text()) if (d / "result.json").exists() else {}
    return {"name": d.name, "meta": meta, "source": source, "samples": samples, "acts": acts, "shots": shots,
            "runway": runway, "result": {k: result.get(k) for k in ("result", "score", "milestones", "failed", "touchdown")}}


@app.route("/c/<name>/replay.json")
def replay_json(name):
    d = CAMPAIGNS / name
    if not d.is_dir():
        abort(404)
    return app.response_class(json.dumps(_replay_data(d)), mimetype="application/json")


@app.route("/c/<name>/shots/<fname>")
def shot(name, fname):
    p = CAMPAIGNS / name / "shots" / fname
    if not p.exists():
        abort(404)
    return send_file(p, mimetype="image/jpeg", max_age=3600)


@app.route("/c/<name>/replay")
def replay(name):
    if not (CAMPAIGNS / name).is_dir():
        abort(404)
    return REPLAY.replace("__NAME__", html.escape(name))


@app.route("/c/<name>/latest.jpg")
def latest(name):
    p = CAMPAIGNS / name / "latest.jpg"
    if not p.exists():
        abort(404)
    return send_file(p, mimetype="image/jpeg", max_age=0)


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8341
    app.run(host="127.0.0.1", port=port, debug=False)
