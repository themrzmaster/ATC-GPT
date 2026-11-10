"""
ATC Trigger System - Rule-based triggers for ATC-initiated calls.

Monitors aircraft state and generates calls the controller would make without being
asked: frequency handoffs along the normal chain

    Delivery -> Ground -> Tower -> Departure -> (Center) -> Approach -> Tower -> Ground

Rules use the airport database for geometry (distance / AGL) and the session's
clearance log (populated by clearance_parser) for "did tower actually clear this
takeoff?". Messages are generated in the session language.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional

import config
from airport_data_service import haversine_nm
from session_manager import ATCSession, ATCPhase, ClearanceType, PHASE_SERVICE_NAME

logger = logging.getLogger("ATCTriggers")


class TriggerType(Enum):
    HANDOFF = "handoff"
    TRAFFIC = "traffic"
    ALTITUDE = "altitude"
    HEADING = "heading"
    SPEED = "speed"
    APPROACH = "approach"
    LANDING = "landing"
    TAXI = "taxi"
    HOLD = "hold"
    IDENT = "ident"
    RADIO_CHECK = "radio_check"
    RUNWAY_INCURSION = "runway_incursion"
    INITIATIVE = "initiative"        # LLM decided to transmit during silence


@dataclass
class ATCCall:
    trigger_type: TriggerType
    message: str
    priority: int = 1  # 1=normal, 2=high, 3=urgent
    new_phase: Optional[ATCPhase] = None
    new_frequency_type: Optional[str] = None  # GND, TWR, DEP, APP
    new_frequency_mhz: Optional[float] = None
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        return {
            "trigger_type": self.trigger_type.value,
            "message": self.message,
            "priority": self.priority,
            "new_phase": self.new_phase.value if self.new_phase else None,
            "new_frequency_type": self.new_frequency_type,
            "new_frequency_mhz": self.new_frequency_mhz,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class TriggerState:
    last_handoff_time: Optional[datetime] = None
    was_on_ground: bool = True
    last_altitude: float = 0
    last_distance_nm: Optional[float] = None
    fired: set = field(default_factory=set)   # rule names already fired (one-shot rules)


# ---------------------------------------------------------------------------
# Phraseology for ATC-initiated calls
# ---------------------------------------------------------------------------

def _say_freq(mhz: float, language: str) -> str:
    """121.7 -> 'one two one decimal seven' / 'um dois um decimal sete'."""
    digits_en = "zero one two three four five six seven eight nine".split()
    digits_pt = "zero um dois três quatro cinco meia sete oito nove".split()
    d = digits_pt if language == "pt-br" else digits_en
    s = f"{mhz:.3f}".rstrip("0").rstrip(".")
    whole, _, frac = s.partition(".")
    words = [d[int(c)] for c in whole]
    if frac:
        words.append("decimal")
        words += [d[int(c)] for c in frac]
    return " ".join(words)


def handoff_message(session: ATCSession, to_phase: ATCPhase, extra: str = "") -> tuple[str, Optional[float]]:
    freq = session.get_handoff_frequency(to_phase)
    lang = session.radio_language()
    name = PHASE_SERVICE_NAME[lang][to_phase]
    if freq:
        spoken = _say_freq(freq.frequency_mhz, lang)
        if lang == "pt-br":
            msg = f"{session.callsign}, {extra}chame {name} em {spoken}."
        else:
            msg = f"{session.callsign}, {extra}contact {name} on {spoken}."
        return msg.replace("  ", " "), freq.frequency_mhz
    # No published frequency for that position: keep them but announce the role change
    if lang == "pt-br":
        msg = f"{session.callsign}, {extra}permaneça nesta frequência, contato radar."
    else:
        msg = f"{session.callsign}, {extra}remain this frequency, radar contact."
    return msg.replace("  ", " "), None


class ATCTriggerEngine:
    """Evaluates rules on every state update and returns at most one ATCCall."""

    def __init__(self):
        self.trigger_states: dict[str, TriggerState] = {}

        # Thresholds
        self.departure_handoff_agl_ft = 800      # switch to departure once climbing through this
        self.approach_handoff_nm = 25.0          # inbound and within this -> approach
        self.tower_handoff_nm = 8.0              # inbound, low, within this -> tower
        self.tower_handoff_max_agl_ft = 3500
        self.touchdown_max_nm = 1.5              # touchdown must be on/near the field for "vacate, contact Ground"
        self.min_handoff_interval = timedelta(seconds=30)

    # -- state -------------------------------------------------------------

    def _sid(self, session: ATCSession) -> str:
        return f"{session.callsign}_{session.airport_icao}"

    def get_trigger_state(self, session: ATCSession) -> TriggerState:
        sid = self._sid(session)
        if sid not in self.trigger_states:
            self.trigger_states[sid] = TriggerState(
                was_on_ground=session.aircraft_state.on_ground,
                last_altitude=session.aircraft_state.altitude_ft,
            )
        return self.trigger_states[sid]

    def reset_session(self, session: ATCSession):
        self.trigger_states.pop(self._sid(session), None)

    def _can_handoff(self, ts: TriggerState) -> bool:
        return ts.last_handoff_time is None or datetime.now() - ts.last_handoff_time > self.min_handoff_interval

    # -- evaluate ----------------------------------------------------------

    def evaluate(self, session: ATCSession) -> Optional[ATCCall]:
        state = session.aircraft_state
        ts = self.get_trigger_state(session)

        call = (
            self._check_departure_handoff(session, ts)
            or self._check_destination_handoff(session, ts)
            or self._check_approach_handoff(session, ts)
            or self._check_tower_handoff(session, ts)
            or self._check_ground_handoff(session, ts)
        )

        ts.was_on_ground = state.on_ground
        ts.last_altitude = state.altitude_ft
        ts.last_distance_nm = session.distance_to_airport_nm()

        if call and call.new_phase and call.new_frequency_mhz is None:
            # Handoff to a position with no published frequency ("remain this frequency, radar contact").
            # With the plugin's radio-follow the session snaps back to the COM1 position and the rule fired
            # again every 30 s (live 2026-11-07, Cabo Frio). Only without COM1 (playground) is it a real
            # role change, and then only once.
            key = f"nofreq_{call.new_phase.value}"
            if getattr(session, "_com1_seen", False) or key in ts.fired:
                return None
            ts.fired.add(key)

        if call:
            logger.info(f"Trigger fired: {call.trigger_type.value} - {call.message}")
            ts.last_handoff_time = datetime.now()
        return call

    # -- rules -------------------------------------------------------------

    def _check_departure_handoff(self, session: ATCSession, ts: TriggerState) -> Optional[ATCCall]:
        """Tower -> Departure once airborne after a takeoff clearance."""
        if session.phase != ATCPhase.TOWER or not self._can_handoff(ts):
            return None
        if session.flight_plan.is_local_pattern():
            return None  # circuits stay with Tower
        state = session.aircraft_state
        if state.on_ground or not session.has_clearance(ClearanceType.TAKEOFF):
            return None
        agl = session.altitude_agl_ft()
        climbing = state.vertical_speed > 200 or (agl is not None and agl > self.departure_handoff_agl_ft)
        if agl is None:
            # No airport elevation: fall back to "was on ground, now airborne and climbing"
            if not (ts.was_on_ground is False and climbing):
                return None
        elif agl < self.departure_handoff_agl_ft:
            return None
        if not climbing:
            return None

        target = ATCPhase.DEPARTURE
        if session.get_handoff_frequency(ATCPhase.DEPARTURE) is None and \
                session.get_handoff_frequency(ATCPhase.APPROACH) is not None:
            target = ATCPhase.APPROACH  # Brazil: APP works departures too
        msg, mhz = handoff_message(session, target)
        return ATCCall(TriggerType.HANDOFF, msg, new_phase=target,
                       new_frequency_type={"departure": "DEP", "approach": "APP"}[target.value],
                       new_frequency_mhz=mhz)

    def _check_approach_handoff(self, session: ATCSession, ts: TriggerState) -> Optional[ATCCall]:
        """Center/Departure -> Approach when inbound and within approach_handoff_nm."""
        if session.phase not in (ATCPhase.CENTER, ATCPhase.DEPARTURE) or not self._can_handoff(ts):
            return None
        state = session.aircraft_state
        if state.on_ground:
            return None
        d = session.distance_to_airport_nm()
        if d is None or d > self.approach_handoff_nm:
            return None
        # Inbound = landing HERE and closing on the field. A departure levelling off inside the ring on
        # its way to another airport is outbound: it must not be handed back to this airport's Approach.
        dest = session.flight_plan.destination
        if dest and dest != session.airport_icao:
            return None
        if ts.last_distance_nm is not None and d >= ts.last_distance_nm:
            return None
        # If we just departed, only once we've been out past the ring (avoid bouncing DEP -> APP).
        if session.phase == ATCPhase.DEPARTURE:
            if "departed_far" not in ts.fired:
                if d > self.approach_handoff_nm * 0.8:
                    ts.fired.add("departed_far")
                return None
            if state.vertical_speed > 300:
                return None
        if session.get_handoff_frequency(ATCPhase.APPROACH) is None and session.phase == ATCPhase.DEPARTURE:
            return None
        msg, mhz = handoff_message(session, ATCPhase.APPROACH)
        return ATCCall(TriggerType.HANDOFF, msg, new_phase=ATCPhase.APPROACH,
                       new_frequency_type="APP", new_frequency_mhz=mhz)

    def _check_destination_handoff(self, session: ATCSession, ts: TriggerState) -> Optional[ATCCall]:
        """
        En route to another airport (session still at the departure's Departure/Approach): inside
        ARRIVAL_TRANSFER_NM of the destination, hand off to the destination's published Approach (or Tower).
        The pilot dials it and the server follows the radio to that airport. A destination with no such
        unit, or one on this same frequency, is a silent transfer done by the server instead.
        """
        dest = session.flight_plan.destination
        if not dest or dest == session.airport_icao or "dest_handoff" in ts.fired or not self._can_handoff(ts):
            return None
        if session.phase not in (ATCPhase.DEPARTURE, ATCPhase.APPROACH, ATCPhase.CENTER):
            return None
        st = session.aircraft_state
        aps = getattr(session, "_airport_service", None)
        ap = aps.get(dest) if aps else None
        if st.on_ground or not (st.latitude or st.longitude) or not ap:
            return None
        if haversine_nm(st.latitude, st.longitude, ap.latitude, ap.longitude) > config.ARRIVAL_TRANSFER_NM:
            return None
        freqs = session._freq_service.get_all_atc(dest)
        ftype = "APP" if "APP" in freqs else "TWR" if "TWR" in freqs else None
        if not ftype:
            return None
        f = freqs[ftype]
        if session.current_frequency_mhz and abs(f.frequency_mhz - session.current_frequency_mhz) < 0.01:
            return None
        ts.fired.add("dest_handoff")
        lang = session.radio_language()
        name = PHASE_SERVICE_NAME[lang][ATCPhase.APPROACH if ftype == "APP" else ATCPhase.TOWER]
        city = ap.municipality or ap.name
        spoken = _say_freq(f.frequency_mhz, lang)
        msg = (f"{session.callsign}, chame {name} {city} em {spoken}." if lang == "pt-br"
               else f"{session.callsign}, contact {city} {name} on {spoken}.")
        return ATCCall(TriggerType.HANDOFF, msg, new_frequency_type=ftype, new_frequency_mhz=f.frequency_mhz)

    def _check_tower_handoff(self, session: ATCSession, ts: TriggerState) -> Optional[ATCCall]:
        """Approach -> Tower when cleared for approach, close in and low."""
        if session.phase != ATCPhase.APPROACH or not self._can_handoff(ts):
            return None
        if session.area_control():
            return None   # no tower here: the area controller keeps the IFR arrival to landing
        state = session.aircraft_state
        if state.on_ground or not session.has_clearance(ClearanceType.APPROACH):
            return None
        d = session.distance_to_airport_nm()
        agl = session.altitude_agl_ft()
        if d is not None and d > self.tower_handoff_nm:
            return None
        if agl is not None and agl > self.tower_handoff_max_agl_ft:
            return None
        if d is None and agl is None and state.altitude_ft > 3000:
            return None
        msg, mhz = handoff_message(session, ATCPhase.TOWER)
        return ATCCall(TriggerType.HANDOFF, msg, new_phase=ATCPhase.TOWER,
                       new_frequency_type="TWR", new_frequency_mhz=mhz)

    def _check_ground_handoff(self, session: ATCSession, ts: TriggerState) -> Optional[ATCCall]:
        """Tower -> Ground after touchdown (landing clearance on record), once slowed."""
        if session.phase != ATCPhase.TOWER:
            return None
        state = session.aircraft_state
        if not state.on_ground or not session.has_clearance(ClearanceType.LANDING):
            return None
        d = session.distance_to_airport_nm()
        if d is not None and d > self.touchdown_max_nm:
            return None   # on the ground far from the field (crash / off-field landing): not a rollout
        # Record the touchdown BEFORE the cooldown gate: the airborne->ground edge is seen once, and a
        # touchdown inside the 30 s cooldown (late Tower handoff) used to lose the Ground handoff forever.
        if "landed" not in ts.fired:
            if not ts.was_on_ground:
                ts.fired.add("landed")
            else:
                return None
        if not self._can_handoff(ts) or state.groundspeed > 60:
            return None  # still rolling out
        extra = "livre da pista, " if session.radio_language() == "pt-br" else "vacate next taxiway, "
        msg, mhz = handoff_message(session, ATCPhase.GROUND, extra=extra)
        ts.fired.discard("landed")
        return ATCCall(TriggerType.HANDOFF, msg, new_phase=ATCPhase.GROUND,
                       new_frequency_type="GND", new_frequency_mhz=mhz)


# Singleton instance
trigger_engine = ATCTriggerEngine()
