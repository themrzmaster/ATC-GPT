"""
PI_ATCTestTools.py - TEST ONLY helper for driving Virtual ATC scenarios remotely (not needed to fly).

Command  atctest/place  reads atctest_place.txt next to this file:
    lat lon elevation_m heading_true speed_m_s
and puts the user aircraft there (XPLMPlaceUserAtLocation: in flight, engines running when speed > 0),
so a remote tester can put the aircraft on climb-out / final and watch the ATC triggers fire.

Command  atctest/ap_hold  engages the autopilot servos with heading + altitude hold (keeps the
aircraft flying straight and level for a minute while the ATC reacts). Best effort per aircraft.

scripts/box.sh deploy-tools copies it; scripts/box.sh place <lat> <lon> <ele_ft> <hdg> <kt> uses it.
"""

import os

from XPPython3 import xp

HERE = os.path.dirname(os.path.abspath(__file__))


class PythonInterface:
    def __init__(self):
        self.Name = "ATC Test Tools"
        self.Sig = "virtualatc.testtools"
        self.Desc = "Test-only: place the aircraft for Virtual ATC scenario runs"
        self.cmds = []

    def XPluginStart(self):
        return self.Name, self.Sig, self.Desc

    def XPluginEnable(self):
        for path, desc, fn in (("atctest/place", "ATC test: place aircraft from atctest_place.txt", self._place),
                               ("atctest/ap_hold", "ATC test: autopilot heading + altitude hold", self._ap_hold)):
            ref = xp.createCommand(path, desc)
            xp.registerCommandHandler(ref, fn, 1, None)
            self.cmds.append((ref, fn))
        xp.log("ATCTestTools: enabled")
        return 1

    def XPluginDisable(self):
        for ref, fn in self.cmds:
            xp.unregisterCommandHandler(ref, fn, 1, None)
        self.cmds = []

    def XPluginStop(self):
        pass

    def XPluginReceiveMessage(self, *a):
        pass

    def _place(self, cmd, phase, refcon):
        if phase != xp.CommandBegin:
            return 0
        try:
            with open(os.path.join(HERE, "atctest_place.txt")) as fh:
                lat, lon, ele, hdg, spd = (float(v) for v in fh.read().split()[:5])
            xp.placeUserAtLocation(lat, lon, ele, hdg, spd)
            xp.log(f"ATCTestTools: placed at {lat:.5f},{lon:.5f} ele {ele:.0f} m hdg {hdg:.0f} spd {spd:.0f} m/s")
        except Exception as e:  # noqa: BLE001
            xp.log(f"ATCTestTools: place failed: {e}")
        return 0

    def _ap_hold(self, cmd, phase, refcon):
        if phase != xp.CommandBegin:
            return 0
        try:
            for c in ("sim/autopilot/servos_on", "sim/autopilot/heading", "sim/autopilot/altitude_hold"):
                ref = xp.findCommand(c)
                if ref:
                    xp.commandOnce(ref)
            xp.log("ATCTestTools: autopilot heading + altitude hold requested")
        except Exception as e:  # noqa: BLE001
            xp.log(f"ATCTestTools: ap_hold failed: {e}")
        return 0
