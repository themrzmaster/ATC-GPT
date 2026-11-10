"""
Position rules - what each ATC position may say, checked in code after the model replies.

The prompt already tells the model "Ground doesn't clear takeoff, Approach doesn't clear to land",
but the 2026-09-28 sweep showed every model (gpt-5.6-luna and gpt-6-luna alike) breaking it now
and then: Approach clearing a 7700 aircraft to land, Tower handing circuit traffic to Approach,
Departure telling the pilot to contact Departure. This module turns those rules into a deterministic
check; the LLM service regenerates once with the reason, and falls back to a handoff to the right
position if the model still gets it wrong.
"""

from dataclasses import dataclass
from typing import Optional

from clearance_parser import parse_clearances
from session_manager import ATCPhase, ATCSession, ClearanceType

C = ClearanceType

# Clearances a position must never issue.
FORBIDDEN: dict[ATCPhase, set[ClearanceType]] = {
    ATCPhase.CLEARANCE: {C.TAXI, C.LINEUP, C.TAKEOFF, C.LANDING, C.APPROACH, C.GO_AROUND},
    ATCPhase.GROUND: {C.LINEUP, C.TAKEOFF, C.LANDING, C.APPROACH},
    ATCPhase.TOWER: set(),
    ATCPhase.DEPARTURE: {C.TAXI, C.LINEUP, C.TAKEOFF, C.LANDING, C.PUSHBACK},
    ATCPhase.APPROACH: {C.TAXI, C.LINEUP, C.TAKEOFF, C.LANDING, C.PUSHBACK},
    ATCPhase.CENTER: {C.TAXI, C.LINEUP, C.TAKEOFF, C.LANDING, C.PUSHBACK, C.APPROACH},
}

# Who issues it instead (for the correction note and the deterministic fallback).
OWNER: dict[ClearanceType, ATCPhase] = {
    C.TAXI: ATCPhase.GROUND, C.PUSHBACK: ATCPhase.GROUND,
    C.LINEUP: ATCPhase.TOWER, C.TAKEOFF: ATCPhase.TOWER, C.LANDING: ATCPhase.TOWER, C.GO_AROUND: ATCPhase.TOWER,
    C.APPROACH: ATCPhase.APPROACH,
}


@dataclass
class Violation:
    reason: str
    owner: Optional[ATCPhase] = None      # position that should handle it (fallback handoff target)


def check(session: ATCSession, text: str) -> list[Violation]:
    """Rule violations in a transmission the controller at session.phase is about to make."""
    out: list[Violation] = []
    phase = session.phase
    for pc in parse_clearances(text, session.callsign):
        if pc.type in FORBIDDEN.get(phase, set()):
            owner = OWNER.get(pc.type)
            if owner == ATCPhase.APPROACH and not session.get_handoff_frequency(ATCPhase.APPROACH):
                owner = None
            out.append(Violation(f"{session.service_name(phase)} may not issue a {pc.type.value} clearance"
                                 + (f" (that is {session.service_name(owner)}'s job)" if owner else ""), owner))
        if pc.type == C.FREQUENCY and pc.position:
            try:
                target = ATCPhase(pc.position)
            except ValueError:
                continue
            same_freq = pc.frequency_khz is None or (
                session.current_frequency_mhz is not None
                and abs(pc.frequency_khz - session.current_frequency_mhz * 1000) <= 10)
            if target == phase and same_freq:
                out.append(Violation(f"you ARE {session.service_name(phase)}: never tell the pilot to contact "
                                     f"your own position"))
            if phase == ATCPhase.TOWER and target in (ATCPhase.APPROACH, ATCPhase.DEPARTURE) \
                    and session.flight_plan.is_local_pattern() and not session.aircraft_state.on_ground \
                    and not _going_around(session):
                out.append(Violation("circuit / touch-and-go traffic stays with Tower: do not hand it to "
                                     f"{session.service_name(target)}"))
    return out


def _going_around(session: ATCSession) -> bool:
    return any(c.type == C.GO_AROUND for c in session.clearances[-3:])


def correction_note(violations: list[Violation]) -> str:
    return ("Your draft reply breaks position rules: " + "; ".join(v.reason for v in violations)
            + ". Rewrite it as the controller of your own position only. If another position must issue that "
              "clearance, give the correct instruction for yours (e.g. hold short / continue approach) and hand "
              "the aircraft to that position with its frequency.")


def fallback(session: ATCSession, violations: list[Violation]) -> Optional[str]:
    """Deterministic last resort: hand the aircraft to the position that owns the clearance."""
    from atc_triggers import handoff_message
    owner = next((v.owner for v in violations if v.owner and v.owner != session.phase
                  and session.get_handoff_frequency(v.owner)), None)
    if owner is None:
        return None
    msg, _ = handoff_message(session, owner)
    return msg
