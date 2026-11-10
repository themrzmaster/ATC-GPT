"""
Session Manager - Manages ATC session state and conversation history.

Tracks:
- Aircraft callsign, type and live state (from X-Plane datarefs or the playground)
- Flight plan (rules, departure, destination, route, cruise) - what a real controller
  has on the strip before you ever key the mic
- Current airport (name, elevation, runways) and frequency
- ATC clearances and assigned altitude/heading/speed/squawk
- Conversation history for LLM context
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

from frequency_service import FrequencyService, Frequency
from airport_data_service import AirportDataService, Airport, haversine_nm, bearing_deg


class ATCPhase(Enum):
    """Current ATC position the pilot is talking to."""
    CLEARANCE = "clearance"     # Clearance delivery (IFR clearance)
    GROUND = "ground"           # Pushback, taxi
    TOWER = "tower"             # Takeoff, landing, pattern
    DEPARTURE = "departure"     # Climb out, initial vectors
    APPROACH = "approach"       # Vectors, approach clearance
    CENTER = "center"           # En-route


PHASE_TO_FREQ_TYPE = {
    ATCPhase.CLEARANCE: "CLD",
    ATCPhase.GROUND: "GND",
    ATCPhase.TOWER: "TWR",
    ATCPhase.DEPARTURE: "DEP",
    ATCPhase.APPROACH: "APP",
    ATCPhase.CENTER: "CTR",
}

EMERGENCY_SQUAWKS = {
    7700: ("EMERGENCY (general)",
           "If the pilot has NOT declared it on the radio, do not say 'roger emergency': confirm it - "
           "'confirm squawking seven seven zero zero, say nature of emergency' - then give priority."),
    7600: ("RADIO FAILURE (lost communications) - the pilot may hear you but cannot reply",
           "Never 'roger' anything (the pilot said nothing). Transmit blind: 'if you hear this transmission, "
           "ident' / 'squawk ident', then give the clearance you expect them to fly (they will follow the "
           "lost-comms procedure). Hand them to Tower as usual."),
    7500: ("UNLAWFUL INTERFERENCE (hijack)",
           "Do not mention it openly. Confirm discreetly: 'confirm squawking seven five zero zero'."),
}

PHASE_SERVICE_NAME = {
    "en": {
        ATCPhase.CLEARANCE: "Delivery", ATCPhase.GROUND: "Ground", ATCPhase.TOWER: "Tower",
        ATCPhase.DEPARTURE: "Departure", ATCPhase.APPROACH: "Approach", ATCPhase.CENTER: "Center",
    },
    "pt-br": {
        ATCPhase.CLEARANCE: "Tráfego", ATCPhase.GROUND: "Solo", ATCPhase.TOWER: "Torre",
        ATCPhase.DEPARTURE: "Controle", ATCPhase.APPROACH: "Controle", ATCPhase.CENTER: "Centro",
    },
}


def phase_from_text(text: str) -> Optional[ATCPhase]:
    """Map a frequency label like 'Ground', 'Solo', 'TWR', 'Approach' to a phase."""
    t = (text or "").lower()
    if any(k in t for k in ("clearance", "delivery", "cld", "tráfego", "trafego")):
        return ATCPhase.CLEARANCE
    if any(k in t for k in ("ground", "gnd", "solo")):
        return ATCPhase.GROUND
    if any(k in t for k in ("tower", "twr", "torre")):
        return ATCPhase.TOWER
    if any(k in t for k in ("departure", "dep")):
        return ATCPhase.DEPARTURE
    if any(k in t for k in ("approach", "app", "controle")):
        return ATCPhase.APPROACH
    if any(k in t for k in ("center", "centre", "ctr", "centro")):
        return ATCPhase.CENTER
    return None


class ClearanceType(Enum):
    """Types of ATC clearances."""
    IFR = "ifr"
    PUSHBACK = "pushback"
    TAXI = "taxi"
    LINEUP = "lineup"
    TAKEOFF = "takeoff"
    LANDING = "landing"
    APPROACH = "approach"
    ALTITUDE = "altitude"
    HEADING = "heading"
    SPEED = "speed"
    FREQUENCY = "frequency"
    SQUAWK = "squawk"
    HOLD = "hold"
    GO_AROUND = "go_around"


@dataclass
class FlightPlan:
    """What the controller knows about you before you call."""
    aircraft_type: str = ""          # e.g. C172, A320
    flight_rules: str = "VFR"        # VFR | IFR
    departure: str = ""              # ICAO
    destination: str = ""            # ICAO
    route: str = ""                  # e.g. "DCT" or "ORONU1A ORONU UZ29 ..."
    cruise_altitude_ft: Optional[int] = None
    remarks: str = ""

    def is_empty(self) -> bool:
        # departure alone doesn't count: the session fills it with the tuned airport
        return not (self.aircraft_type or self.destination or self.route)

    def is_local_pattern(self) -> bool:
        """Circuit / touch-and-go work: stays with Tower, never handed to Departure/Approach."""
        text = f"{self.route} {self.remarks}".lower()
        keywords = ("pattern", "circuit", "touch and go", "touch-and-go", "closed traffic", "local",
                    "circuito", "toque e arremetida", "tráfego local", "trafego local", "tgl", "t&g")
        if any(k in text for k in keywords):
            return True
        return bool(self.departure and self.destination and self.departure == self.destination
                    and not self.route.upper().replace("DCT", "").strip())

    def to_dict(self) -> dict:
        return {
            "aircraft_type": self.aircraft_type,
            "flight_rules": self.flight_rules,
            "departure": self.departure,
            "destination": self.destination,
            "route": self.route,
            "cruise_altitude_ft": self.cruise_altitude_ft,
            "remarks": self.remarks,
        }

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "FlightPlan":
        d = d or {}
        cr = d.get("cruise_altitude_ft")
        return cls(
            aircraft_type=(d.get("aircraft_type") or "").upper(),
            flight_rules=(d.get("flight_rules") or "VFR").upper(),
            departure=(d.get("departure") or "").upper(),
            destination=(d.get("destination") or "").upper(),
            route=d.get("route") or "",
            cruise_altitude_ft=int(cr) if cr not in (None, "") else None,
            remarks=d.get("remarks") or "",
        )

    def update_from(self, d: Optional[dict]) -> bool:
        """Merge a plan read from the aircraft FMS (only non-empty fields). True if anything changed."""
        if not d:
            return False
        before = self.to_dict()
        for key in ("aircraft_type", "flight_rules", "departure", "destination"):
            v = (d.get(key) or "").strip().upper()
            if v:
                setattr(self, key, v)
        if (d.get("route") or "").strip():
            self.route = d["route"].strip()
        if d.get("cruise_altitude_ft"):
            self.cruise_altitude_ft = int(d["cruise_altitude_ft"])
        return self.to_dict() != before

    def summary(self) -> str:
        if self.is_empty():
            return "No flight plan on file (treat as VFR / unknown intentions - ask type and intentions)"
        parts = [f"{self.flight_rules}"]
        if self.aircraft_type:
            parts.append(f"type {self.aircraft_type}")
        if self.departure or self.destination:
            parts.append(f"{self.departure or '?'} -> {self.destination or '?'}")
        if self.route:
            parts.append(f"route {self.route}")
        if self.cruise_altitude_ft:
            parts.append(f"cruise {self.cruise_altitude_ft}ft")
        if self.remarks:
            parts.append(f"rmk {self.remarks}")
        return ", ".join(parts)


@dataclass
class AircraftState:
    """Current aircraft state from X-Plane datarefs."""
    callsign: str = "N12345"
    latitude: float = 0.0
    longitude: float = 0.0
    altitude_ft: float = 0.0       # MSL feet
    heading: float = 0.0           # True heading degrees
    indicated_airspeed: float = 0.0  # Knots
    groundspeed: float = 0.0       # Knots
    vertical_speed: float = 0.0    # FPM
    squawk: int = 1200
    com1_freq_hz: int = 121500000  # 121.5 MHz
    on_ground: bool = True
    updated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        return {
            "callsign": self.callsign,
            "position": {
                "lat": round(self.latitude, 6),
                "lon": round(self.longitude, 6),
                "alt_ft": round(self.altitude_ft),
            },
            "heading": round(self.heading),
            "ias_kts": round(self.indicated_airspeed),
            "gs_kts": round(self.groundspeed),
            "vs_fpm": round(self.vertical_speed),
            "squawk": self.squawk,
            "com1_freq": self.com1_freq_hz / 1000000,  # Convert to MHz
            "on_ground": self.on_ground,
        }

    def position_string(self) -> str:
        return (f"{abs(self.latitude):.4f}{'N' if self.latitude >= 0 else 'S'}, "
                f"{abs(self.longitude):.4f}{'E' if self.longitude >= 0 else 'W'} at {self.altitude_ft:.0f}ft")


@dataclass
class Clearance:
    """A single ATC clearance or instruction."""
    type: ClearanceType
    details: str
    timestamp: datetime = field(default_factory=datetime.now)
    acknowledged: bool = False
    value: Optional[int] = None   # parsed altitude/heading/squawk when applicable
    runway: Optional[str] = None  # parsed runway when applicable

    def to_dict(self) -> dict:
        return {
            "type": self.type.value,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
            "acknowledged": self.acknowledged,
            "value": self.value,
            "runway": self.runway,
        }


@dataclass
class ConversationMessage:
    role: str  # "pilot" or "atc"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content, "timestamp": self.timestamp.isoformat()}


class ATCSession:
    """One aircraft at one airport, talking to one ATC position at a time."""

    def __init__(
        self,
        airport_icao: str,
        callsign: str = "N12345",
        frequency_service: Optional[FrequencyService] = None,
        airport_data_service: Optional[AirportDataService] = None,
        language: str = "en",
        flight_plan: Optional[FlightPlan] = None,
    ):
        self.airport_icao = airport_icao.upper()
        self.callsign = callsign
        self.language = language if language in PHASE_SERVICE_NAME else "en"

        self._freq_service = frequency_service or FrequencyService()
        self._airport_service = airport_data_service
        self.airport: Optional[Airport] = (
            self._airport_service.get(self.airport_icao) if self._airport_service else None
        )

        self.flight_plan = flight_plan or FlightPlan()
        if not self.flight_plan.departure:
            self.flight_plan.departure = self.airport_icao

        # Current state
        self.phase = ATCPhase.GROUND
        self.current_frequency = "Ground"
        self.current_frequency_mhz: Optional[float] = None
        self.aircraft_state = AircraftState(callsign=callsign)

        self.airport_frequencies = self._freq_service.get_all_atc(airport_icao)
        self._update_frequency_for_phase()

        # Clearances and instructions
        self.clearances: list[Clearance] = []
        self.assigned_altitude: Optional[int] = None
        self.assigned_heading: Optional[int] = None
        self.assigned_speed: Optional[int] = None
        self.assigned_squawk: Optional[int] = None
        self.assigned_runway: Optional[str] = None

        self.conversation: list[ConversationMessage] = []
        self.created_at = datetime.now()
        self.last_transmission = datetime.now()

        # Facts already looked up this session (weather, chart lists...) so the model does not
        # burn a tool round-trip on every transmission. key -> (text, datetime)
        self.known_facts: dict[str, tuple[str, datetime]] = {}

        # Handoff issued by the controller ("contact tower on ...") but not yet acted on.
        # Applied on the pilot's next transmission unless the radio (COM1) says otherwise.
        self.pending_handoff: Optional[ATCPhase] = None
        self.pending_handoff_khz: Optional[int] = None
        # Pilot request the controller answered with "stand by"/"aguarde": answered unprompted later.
        self.standby_request: Optional[str] = None
        self.standby_at: Optional[datetime] = None
        # Controller calls the pilot hasn't answered yet (radio checks stop after a few).
        self.unanswered_initiatives: int = 0
        # Last time lat/lon changed: an airborne aircraft frozen in place = sim paused.
        self.position_changed_at: datetime = datetime.now()
        # Frequency an arrival controller took the flight over on (destination without its own unit)
        self.worked_freq_mhz: Optional[float] = None
        self._last_com1_phase: Optional[ATCPhase] = None
        self._com1_seen: bool = False

    # ------------------------------------------------------------------
    # State updates
    # ------------------------------------------------------------------

    def update_aircraft_state(
        self,
        latitude: float = None,
        longitude: float = None,
        altitude_ft: float = None,
        heading: float = None,
        indicated_airspeed: float = None,
        groundspeed: float = None,
        vertical_speed: float = None,
        squawk: int = None,
        com1_freq_hz: int = None,
        on_ground: bool = None,
    ):
        state = self.aircraft_state
        if (latitude is not None and latitude != state.latitude) or \
                (longitude is not None and longitude != state.longitude):
            self.position_changed_at = datetime.now()
        for name, val in (
            ("latitude", latitude), ("longitude", longitude), ("altitude_ft", altitude_ft),
            ("heading", heading), ("indicated_airspeed", indicated_airspeed),
            ("groundspeed", groundspeed), ("vertical_speed", vertical_speed),
            ("squawk", squawk), ("com1_freq_hz", com1_freq_hz), ("on_ground", on_ground),
        ):
            if val is not None:
                setattr(state, name, val)
        state.updated_at = datetime.now()

    def add_pilot_transmission(self, text: str):
        self.conversation.append(ConversationMessage(role="pilot", content=text))
        self.last_transmission = datetime.now()
        self.unanswered_initiatives = 0

    def sim_frozen(self, after_s: float = 10.0) -> bool:
        """Airborne (or rolling) yet the position hasn't changed: the sim is paused."""
        st = self.aircraft_state
        moving = (not st.on_ground) or st.groundspeed > 5
        return moving and (datetime.now() - self.position_changed_at).total_seconds() >= after_s

    def add_atc_transmission(self, text: str):
        self.conversation.append(ConversationMessage(role="atc", content=text))

    def add_clearance(
        self,
        clearance_type: ClearanceType,
        details: str,
        value: Optional[int] = None,
        runway: Optional[str] = None,
        position: Optional[str] = None,
        frequency_khz: Optional[int] = None,
    ) -> Clearance:
        clearance = Clearance(type=clearance_type, details=details, value=value, runway=runway)
        self.clearances.append(clearance)

        if clearance_type == ClearanceType.FREQUENCY:
            target = None
            if frequency_khz:
                target = self.phase_for_frequency(frequency_khz * 1000)
            if target is None and position:
                try:
                    target = ATCPhase(position)
                except ValueError:
                    target = None
            if target and target != self.phase:
                self.pending_handoff = target
                self.pending_handoff_khz = frequency_khz

        if value is not None:
            if clearance_type == ClearanceType.ALTITUDE:
                self.assigned_altitude = value
            elif clearance_type == ClearanceType.HEADING:
                self.assigned_heading = value
            elif clearance_type == ClearanceType.SQUAWK:
                self.assigned_squawk = value
            elif clearance_type == ClearanceType.SPEED:
                self.assigned_speed = value
        if runway and clearance_type in (
            ClearanceType.TAXI, ClearanceType.LINEUP, ClearanceType.TAKEOFF,
            ClearanceType.LANDING, ClearanceType.APPROACH,
        ):
            self.assigned_runway = runway
        return clearance

    def has_clearance(self, clearance_type: ClearanceType) -> bool:
        return any(c.type == clearance_type for c in self.clearances)

    # ------------------------------------------------------------------
    # Frequencies / phases
    # ------------------------------------------------------------------

    def _update_frequency_for_phase(self):
        freq_type = PHASE_TO_FREQ_TYPE.get(self.phase, "GND")
        freq = self.airport_frequencies.get(freq_type)
        if freq:
            self.current_frequency = freq.description or freq_type
            self.current_frequency_mhz = freq.frequency_mhz
        else:
            self.current_frequency = PHASE_SERVICE_NAME[self.language].get(self.phase, "Ground")
            self.current_frequency_mhz = None

    def set_phase(self, phase: ATCPhase):
        self.phase = phase
        self._update_frequency_for_phase()

    def change_frequency(self, new_frequency: str, new_phase: Optional[ATCPhase] = None):
        self.current_frequency = new_frequency
        if new_phase:
            self.set_phase(new_phase)

    def get_expected_frequency_hz(self) -> Optional[int]:
        if self.current_frequency_mhz:
            return int(round(self.current_frequency_mhz * 1_000_000))
        return None

    def validate_pilot_frequency(self, pilot_freq_hz: int, tolerance_khz: float = 25.0) -> tuple[bool, str]:
        expected_hz = self.get_expected_frequency_hz()
        if expected_hz is None:
            return True, ""
        pilot_mhz = pilot_freq_hz / 1_000_000
        expected_mhz = expected_hz / 1_000_000
        if abs(pilot_freq_hz - expected_hz) / 1000 <= tolerance_khz:
            return True, ""
        matched = self._freq_service.match_frequency(self.airport_icao, pilot_mhz)
        if matched:
            service_type, freq = matched
            return False, (f"Pilot is transmitting on {service_type} ({freq.frequency_formatted()}), "
                           f"not {self.current_frequency} ({expected_mhz:.3f})")
        return False, f"Pilot is on {pilot_mhz:.3f}; {self.current_frequency} is {expected_mhz:.3f} MHz"

    def phase_for_frequency(self, freq_hz: int, tolerance_khz: float = 25.0) -> Optional[ATCPhase]:
        """Which ATC position at this airport is published on freq_hz (None if unknown)."""
        if not freq_hz:
            return None
        best: tuple[float, Optional[ATCPhase]] = (tolerance_khz + 1, None)
        for phase, ftype in PHASE_TO_FREQ_TYPE.items():
            f = self.airport_frequencies.get(ftype)
            if not f:
                continue
            diff_khz = abs(f.frequency_hz() - freq_hz) / 1000
            if diff_khz <= tolerance_khz and diff_khz < best[0]:
                best = (diff_khz, phase)
        return best[1]

    def sync_phase_to_frequency(self, freq_hz: Optional[int]) -> Optional[ATCPhase]:
        """
        Follow the pilot's radio: if COM1 is on another published position of this airport,
        switch the session to that position. Returns the new phase when it changed.
        This is what makes the X-Plane plugin work without any UI - you tune Tower, you get Tower.
        """
        if not freq_hz:
            return None
        self._com1_seen = True
        phase = self.phase_for_frequency(freq_hz)
        self._last_com1_phase = phase
        if phase is None or phase == self.phase:
            return None
        self.set_phase(phase)
        self.pending_handoff = None
        return phase

    _READBACK_FILLER = {
        "contact", "contacting", "chamo", "chamarei", "chamando", "chame", "on", "em", "na", "no", "frequency",
        "frequência", "frequencia", "roger", "ciente", "wilco", "good", "day", "bom", "dia", "tarde", "noite",
        "obrigado", "obrigada", "thanks", "thank", "you", "to", "over", "switching", "mudando", "de", "a", "o",
        "decimal", "point", "ponto", "and", "e", "monitor", "monitoring", "escuta", "mantendo", "for", "para",
    }

    def handoff_readback_kind(self, pilot_text: str) -> Optional[str]:
        """
        After "contact Ground on 121.7", a transmission containing that frequency is the pilot's
        readback on the OLD frequency (the radio hasn't been switched yet):
          "short" - just the frequency ("One two one decimal seven, PT-ABC"): nobody answers it.
          "full"  - frequency plus a clearance readback: the old controller still checks it.
        None when this isn't a readback of the pending handoff.
        """
        if not self.pending_handoff or not self.pending_handoff_khz:
            return None
        from clearance_parser import frequencies_in, strip_callsign, DIGIT_WORDS, POSITION_WORDS
        if not any(abs(f - self.pending_handoff_khz) <= 10 for f in frequencies_in(pilot_text, self.callsign)):
            return None
        import re
        words = [w for w in re.findall(r"[A-Za-zÀ-ÿ]+", strip_callsign(pilot_text, self.callsign).lower())
                 if w not in DIGIT_WORDS and w not in POSITION_WORDS and w not in self._READBACK_FILLER
                 and w not in (self.airport.name.lower().split() if self.airport else [])]
        return "short" if len(words) <= 3 else "full"

    def is_handoff_readback(self, pilot_text: str) -> bool:
        return self.handoff_readback_kind(pilot_text) == "short"

    def follow_addressed_station(self, pilot_text: str) -> Optional[ATCPhase]:
        """
        No radio reported (playground / e2e): "Guarulhos Torre, PT-ABC, ..." means the pilot has
        tuned Tower, so Tower answers - even if nobody told them to switch. With a real radio
        (COM1 seen) the frequency is authoritative and this does nothing.
        """
        if self._com1_seen:
            return None
        from clearance_parser import strip_callsign
        head = strip_callsign(pilot_text, self.callsign).split(",")[0]
        head = " ".join(head.split()[:4])
        phase = phase_from_text(head)
        if phase is None:
            return None
        if phase == ATCPhase.DEPARTURE and not self.get_handoff_frequency(phase) \
                and self.get_handoff_frequency(ATCPhase.APPROACH):
            phase = ATCPhase.APPROACH          # Brazil: Controle works departures
        if phase == self.phase or not self.get_handoff_frequency(phase):
            return None
        self.set_phase(phase)
        self.pending_handoff = self.pending_handoff_khz = None
        return phase

    def apply_pending_handoff(self) -> Optional[ATCPhase]:
        """
        Called when the pilot transmits. If the controller told them to contact another
        position and the radio isn't telling us otherwise, assume they did (the next call
        happens on the new frequency). With a real radio (COM1 reported and recognised), the
        radio is authoritative and this is a no-op.
        """
        if not self.pending_handoff:
            return None
        if self._com1_seen and self._last_com1_phase is not None:
            if self._last_com1_phase == self.pending_handoff:
                self.pending_handoff = None
            return None  # radio says where the pilot is; sync_phase_to_frequency handles it
        target, self.pending_handoff, self.pending_handoff_khz = self.pending_handoff, None, None
        if target != self.phase:
            self.set_phase(target)
            return target
        return None

    def has_atc_service(self) -> bool:
        """False for uncontrolled fields: nothing but CTAF/UNICOM/OPS (or nothing) published."""
        return any(t in self.airport_frequencies for t in ("CLD", "GND", "TWR", "DEP", "APP"))

    def area_control(self) -> bool:
        """
        IFR flight in the air at a field with no Tower/Approach (e.g. Brazil's AFIS-only aerodromes such
        as Cabo Frio): the area radar controller keeps working it on the pilot's frequency - descent,
        approach clearance, landing information - instead of "no ATC service, self-announce".
        """
        return (not self.has_atc_service() and self.flight_plan.flight_rules == "IFR"
                and not self.aircraft_state.on_ground)

    def info_frequency(self):
        """AFIS / CTAF / UNICOM station published here (None if none)."""
        getter = getattr(self._freq_service, "get_info_frequency", None)
        return getter(self.airport_icao) if getter else None

    def get_handoff_frequency(self, to_phase: ATCPhase) -> Optional[Frequency]:
        freq_type = PHASE_TO_FREQ_TYPE.get(to_phase)
        return self.airport_frequencies.get(freq_type) if freq_type else None

    def radio_language(self) -> str:
        """Session language, except English when the last pilot call was English (foreign crew in Brazil)."""
        if self.language == "pt-br":
            last = next((m.content for m in reversed(self.conversation) if m.role == "pilot"), "")
            from atc_llm_service import pilot_speaks_english
            if last and pilot_speaks_english(last):
                return "en"
        return self.language

    def service_name(self, phase: Optional[ATCPhase] = None) -> str:
        return PHASE_SERVICE_NAME[self.language][phase or self.phase]

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------

    def distance_to_airport_nm(self) -> Optional[float]:
        if not self.airport or (self.aircraft_state.latitude == 0 and self.aircraft_state.longitude == 0):
            return None
        return haversine_nm(self.aircraft_state.latitude, self.aircraft_state.longitude,
                            self.airport.latitude, self.airport.longitude)

    def bearing_from_airport(self) -> Optional[float]:
        if not self.airport or (self.aircraft_state.latitude == 0 and self.aircraft_state.longitude == 0):
            return None
        return bearing_deg(self.airport.latitude, self.airport.longitude,
                           self.aircraft_state.latitude, self.aircraft_state.longitude)

    def altitude_agl_ft(self) -> Optional[float]:
        if self.airport and self.airport.elevation_ft is not None:
            return self.aircraft_state.altitude_ft - self.airport.elevation_ft
        return None

    # ------------------------------------------------------------------
    # LLM context
    # ------------------------------------------------------------------

    def get_situation_summary(self) -> str:
        state = self.aircraft_state
        lines = []
        if self.airport:
            lines.append(f"Airport: {self.airport.summary_line()}")
            lines.append(f"Valid runway identifiers here: {', '.join(self.airport.runway_idents()) or 'unknown'}")
        else:
            lines.append(f"Airport: {self.airport_icao} (no airport database record - runways unknown)")

        freqs = ", ".join(f"{t} {f.frequency_formatted()}" for t, f in self.airport_frequencies.items())
        if freqs:
            lines.append(f"Published frequencies: {freqs}")
        dest = self.flight_plan.destination
        if dest and dest != self.airport_icao:
            dfreqs = self._freq_service.get_all_atc(dest)
            if dfreqs:
                lines.append(f"Destination {dest} published frequencies (use these, never invent): "
                             + ", ".join(f"{t} {f.frequency_formatted()}" for t, f in dfreqs.items()))
        info = self.info_frequency()
        info_txt = f" (only a non-ATC station: {info.type} '{info.description}' {info.frequency_formatted()})" \
            if info else ""
        if self.area_control():
            lines.append(
                f"NO TOWER OR APPROACH UNIT is published at {self.airport_icao}{info_txt}. This IFR flight is still "
                "under radar control: YOU are the area radar controller ('Controle' / 'Control') working it on the "
                "pilot's current frequency; the pilot may call you by the airport name. Never tell this IFR flight "
                "that there is no ATC service and never send it to another frequency (no other station is "
                "simulated). Give descent, direct/vectors and the instrument approach clearance with an altitude. "
                "There is no tower: once it is established on final give wind and runway and say \"pouso a critério "
                "do piloto\" / \"landing at your discretion\" (never 'cleared to land'), and keep it on your frequency.")
        elif not self.has_atc_service():
            ctaf = f"{info.frequency_formatted()} ({info.description or info.type})" if info else \
                "123.45 (auto-informação, Brazil) / the CTAF on the chart"
            lines.append("NO ATC SERVICE at this aerodrome (no tower/ground/delivery frequency published). Calls "
                         f"here are pilots self-announcing to other traffic on {ctaf}. "
                         "You are NOT a controller here: never issue taxi, takeoff, landing or hold instructions. "
                         "If addressed directly, say once that there is no ATC service and to self-announce; "
                         "otherwise reply exactly: NONE")

        lines += [
            f"UTC time: {datetime.utcnow():%H%M}",
            f"Aircraft: {self.callsign}",
            f"Flight plan: {self.flight_plan.summary()}",
            f"You are: {self.current_frequency} ({self.service_name()}), phase '{self.phase.value}'"
            + (f" on {self.current_frequency_mhz:.3f}" if self.current_frequency_mhz else ""),
            f"Position: {state.position_string()}",
        ]
        d = self.distance_to_airport_nm()
        if d is not None:
            b = self.bearing_from_airport()
            agl = self.altitude_agl_ft()
            lines.append(f"Distance from airport: {d:.1f} nm, bearing {b:.0f}° from field"
                         + (f", {agl:.0f} ft AGL" if agl is not None else ""))
        lines += [
            f"Heading: {state.heading:.0f}°",
            f"Speed: {state.indicated_airspeed:.0f} kts IAS, {state.groundspeed:.0f} kts GS",
            f"Vertical: {state.vertical_speed:+.0f} fpm",
            f"Squawk: {state.squawk:04d}",
            f"On Ground: {state.on_ground}",
        ]
        alert = EMERGENCY_SQUAWKS.get(state.squawk)
        if alert:
            meaning, how = alert
            lines.append(f"RADAR ALERT: transponder {state.squawk:04d} = {meaning}. Deal with it on your next "
                         f"transmission, before anything routine. {how}")

        if self.assigned_runway:
            lines.append(f"Assigned runway: {self.assigned_runway}")
        if self.assigned_altitude:
            lines.append(f"Assigned altitude: {self.assigned_altitude} ft")
        if self.assigned_heading is not None:
            lines.append(f"Assigned heading: {self.assigned_heading:03d}°")
        if self.assigned_speed:
            lines.append(f"Assigned speed: {self.assigned_speed} kts")
        if self.assigned_squawk:
            lines.append(f"Assigned squawk: {self.assigned_squawk:04d}")

        facts = self.known_facts_summary()
        if facts:
            lines.append(facts)

        if self.clearances:
            lines.append("Clearances already issued this session:")
            for c in self.clearances[-6:]:
                lines.append(f"  - {c.type.value}: {c.details[:100]}")
        else:
            lines.append("Clearances already issued this session: none")

        return "\n".join(lines)

    def remember_fact(self, key: str, text: str):
        self.known_facts[key] = (text, datetime.now())

    def fact(self, key: str, max_age_s: Optional[float] = None) -> Optional[str]:
        item = self.known_facts.get(key)
        if not item:
            return None
        text, ts = item
        if max_age_s is not None and (datetime.now() - ts).total_seconds() > max_age_s:
            return None
        return text

    def known_facts_summary(self) -> str:
        if not self.known_facts:
            return ""
        lines = ["Facts already looked up this session (do not call the tool again unless they may have changed):"]
        for key, (text, ts) in self.known_facts.items():
            age = int((datetime.now() - ts).total_seconds() // 60)
            lines.append(f"  - {key} ({age} min ago): {text}")
        return "\n".join(lines)

    def get_context_for_llm(self, max_messages: int = 20) -> dict:
        return {
            "airport": self.airport_icao,
            "callsign": self.callsign,
            "phase": self.phase.value,
            "frequency": self.current_frequency,
            "flight_plan": self.flight_plan.to_dict(),
            "aircraft_state": self.aircraft_state.to_dict(),
            "active_clearances": [c.to_dict() for c in self.clearances if not c.acknowledged],
            "assigned": self._assigned_dict(),
            "conversation_history": [m.to_dict() for m in self.conversation[-max_messages:]],
        }

    def _assigned_dict(self) -> dict:
        return {
            "altitude": self.assigned_altitude,
            "heading": self.assigned_heading,
            "speed": self.assigned_speed,
            "squawk": self.assigned_squawk,
            "runway": self.assigned_runway,
        }

    def to_dict(self) -> dict:
        return {
            "airport_icao": self.airport_icao,
            "airport": self.airport.to_dict() if self.airport else None,
            "callsign": self.callsign,
            "language": self.language,
            "phase": self.phase.value,
            "frequency": self.current_frequency,
            "frequency_mhz": self.current_frequency_mhz,
            "pending_handoff": self.pending_handoff.value if self.pending_handoff else None,
            "flight_plan": self.flight_plan.to_dict(),
            "aircraft_state": self.aircraft_state.to_dict(),
            "distance_nm": self.distance_to_airport_nm(),
            "clearances": [c.to_dict() for c in self.clearances],
            "assigned": self._assigned_dict(),
            "conversation": [m.to_dict() for m in self.conversation],
            "created_at": self.created_at.isoformat(),
            "last_transmission": self.last_transmission.isoformat(),
        }


class SessionManager:
    """Manages ATC sessions (single active session today)."""

    def __init__(self):
        self.sessions: dict[str, ATCSession] = {}
        self.active_session_id: Optional[str] = None

    def create_session(
        self,
        airport_icao: str,
        callsign: str = "N12345",
        frequency_service: Optional[FrequencyService] = None,
        airport_data_service: Optional[AirportDataService] = None,
        language: str = "en",
        flight_plan: Optional[FlightPlan] = None,
    ) -> ATCSession:
        session_id = f"{callsign}_{airport_icao.upper()}"
        session = ATCSession(
            airport_icao=airport_icao,
            callsign=callsign,
            frequency_service=frequency_service,
            airport_data_service=airport_data_service,
            language=language,
            flight_plan=flight_plan,
        )
        self.sessions[session_id] = session
        self.active_session_id = session_id
        return session

    def get_active_session(self) -> Optional[ATCSession]:
        if self.active_session_id:
            return self.sessions.get(self.active_session_id)
        return None

    def get_session(self, callsign: str, airport_icao: str) -> Optional[ATCSession]:
        return self.sessions.get(f"{callsign}_{airport_icao.upper()}")

    def end_session(self, callsign: str, airport_icao: str):
        session_id = f"{callsign}_{airport_icao.upper()}"
        self.sessions.pop(session_id, None)
        if self.active_session_id == session_id:
            self.active_session_id = None


if __name__ == "__main__":
    import json
    session = ATCSession(airport_icao="KJFK", callsign="N12345",
                         airport_data_service=AirportDataService(),
                         flight_plan=FlightPlan(aircraft_type="C172", flight_rules="VFR", destination="KHPN"))
    session.update_aircraft_state(latitude=40.6413, longitude=-73.7781, altitude_ft=1500, heading=270,
                                  indicated_airspeed=120, groundspeed=130, vertical_speed=500, squawk=1200,
                                  on_ground=False)
    session.add_pilot_transmission("Kennedy Ground, November one two three four five, ready to taxi with Alpha")
    session.add_atc_transmission("November one two three four five, taxi to runway three one left via Alpha, Bravo")
    session.add_clearance(ClearanceType.TAXI, "Runway 31L via A, B", runway="31L")
    print("=== Situation Summary ===")
    print(session.get_situation_summary())
    print("\n=== LLM Context ===")
    print(json.dumps(session.get_context_for_llm(), indent=2))
