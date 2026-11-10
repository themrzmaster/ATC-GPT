"""List every input device pyaudio sees and record 2 s from each, printing the level.

Run on the X-Plane box (at the desk, not over SSH/RDP) while talking continuously:
    python mic_test.py
The device with the highest level while you talk goes in virtualatc_config.txt as mic_device=<index>.
"""
import array
import math

import pyaudio

p = pyaudio.PyAudio()
default = p.get_default_input_device_info()["index"]
print(f"Windows default input: {default}\nTalk continuously now...\n")
for i in range(p.get_device_count()):
    info = p.get_device_info_by_index(i)
    if info["maxInputChannels"] < 1:
        continue
    name = f"{i:>3}  {info['name'][:45]:<45}"
    try:
        s = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True,
                   input_device_index=i, frames_per_buffer=1024)
        data = b"".join(s.read(1024, exception_on_overflow=False) for _ in range(31))  # ~2 s
        s.stop_stream(); s.close()
        a = array.array("h", data)
        rms = math.sqrt(sum(x * x for x in a) / len(a))
        peak = max(abs(x) for x in a)
        bar = "#" * min(40, int(rms / 100))
        print(f"{name} rms={rms:7.0f} peak={peak:6d} {bar}")
    except Exception as e:  # noqa: BLE001
        print(f"{name} cannot open at 16 kHz ({str(e)[:40]})")
p.terminate()
print("\nrms under ~50 = silence. Put the loudest index in virtualatc_config.txt as mic_device=<index>.")
