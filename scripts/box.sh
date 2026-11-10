#!/usr/bin/env bash
# Drive the Windows X-Plane box (Tailscale) for plugin testing without touching the keyboard there.
#
#   scripts/box.sh deploy            copy plugin + config + test wavs into X-Plane's PythonPlugins
#   scripts/box.sh start             launch X-Plane in the logged-in desktop session (scheduled task)
#   scripts/box.sh stop              kill X-Plane
#   scripts/box.sh log               VirtualATC lines from XPPython3Log.txt
#   scripts/box.sh com1 121600       set COM1 (kHz) via X-Plane's UDP DREF interface
#   scripts/box.sh test              fire command virtualatc/test_transmit (sends virtualatc_test.wav as the pilot)
#   scripts/box.sh cmd <command>     run any X-Plane command via UDP CMND (e.g. sim/operation/pause_toggle)
#   scripts/box.sh deploy-tools      TEST ONLY: install PI_ATCTestTools.py (restart X-Plane after)
#   scripts/box.sh place <lat> <lon> <ele_ft> <hdg> <kt>   put the aircraft there in flight (needs deploy-tools)
#   scripts/box.sh deploy-llmplays   install PI_LLMPlays.py ("LLM plays X-Plane" turn-based API; restart X-Plane after)
#   scripts/box.sh tunnel            SSH-forward the LLMPlays API to http://127.0.0.1:5066 on this Mac
#   scripts/box.sh ps <ps1 file|->   arbitrary PowerShell
#
# The ATC server must be reachable from the box: run it on the Mac with
#   ATC_SERVER_HOST=0.0.0.0 ATC_SERVER_PORT=5055 make serve     (5000 is taken by macOS AirPlay)
# and point xplane_plugin/virtualatc_config.txt at http://<mac tailscale ip>:5055.
set -euo pipefail

# Personal settings (BOX_HOST, BOX_USER, ...) go in scripts/box.local.sh, which git ignores.
_here="$(cd "$(dirname "$0")" && pwd)"
[ -f "$_here/box.local.sh" ] && . "$_here/box.local.sh"
BOX_HOST="${BOX_HOST:?set BOX_HOST to the X-Plane box address (e.g. its Tailscale IP)}"
BOX_USER="${BOX_USER:?set BOX_USER to the Windows user on the box}"
BOX_KEY="${BOX_KEY:-$HOME/.ssh/id_ed25519}"
BOX_PYTHON="${BOX_PYTHON:-C:\\Users\\$BOX_USER\\AppData\\Local\\Programs\\Python\\Python312\\python.exe}"   # Python with pyaudio, used for py_compile
BOX_KNOWN_HOSTS="${BOX_KNOWN_HOSTS:-$HOME/.ssh/known_hosts_atc_windows_box}"   # was in /tmp: wiped on reboot
XP_DIR='C:/X-Plane 12'
PLUGIN_DIR="$XP_DIR/Resources/plugins/PythonPlugins"
TASK="VirtualATCXPlane"
HERE="$(cd "$(dirname "$0")/.." && pwd)"

if [ ! -f "$BOX_KNOWN_HOSTS" ]; then
  echo "pinned known_hosts missing: $BOX_KNOWN_HOSTS (recreate with ssh-keyscan after checking the fingerprint)" >&2
  exit 2
fi
SSH_OPTS=(-i "$BOX_KEY" -o "UserKnownHostsFile=$BOX_KNOWN_HOSTS" -o StrictHostKeyChecking=yes -o BatchMode=yes -o ConnectTimeout=20)

ps_run() {  # PowerShell script from stdin, encoded like every other tool that talks to this box
  local enc; enc="$(iconv -f UTF-8 -t UTF-16LE | base64)"
  ssh "${SSH_OPTS[@]}" "$BOX_USER@$BOX_HOST" "powershell -NoProfile -NonInteractive -OutputFormat Text -EncodedCommand $enc" 2>&1 \
    | LC_ALL=C grep -v '^#< CLIXML' | LC_ALL=C sed -e 's/<[^>]*>//g' -e 's/_x000D__x000A_/\n/g' | LC_ALL=C grep -v -e '^[[:space:]]*$' -e 'System.Management.Automation' || true
}

udp_cmd() { cat <<PS | ps_run
\$udp = New-Object System.Net.Sockets.UdpClient
\$b = [System.Text.Encoding]::ASCII.GetBytes("CMND") + 0 + [System.Text.Encoding]::ASCII.GetBytes("$1") + 0
\$udp.Send(\$b, \$b.Length, '127.0.0.1', 49000) | Out-Null; \$udp.Close(); "sent $1"
PS
}

case "${1:-}" in
  deploy)
    for f in PI_VirtualATC.py virtualatc_config.txt virtualatc_test.wav virtualatc_test_en.wav; do
      [ -f "$HERE/xplane_plugin/$f" ] || continue
      scp -q "${SSH_OPTS[@]}" "$HERE/xplane_plugin/$f" "$BOX_USER@$BOX_HOST:$PLUGIN_DIR/$f" && echo "copied $f"
    done
    cat <<PS | ps_run
& '$BOX_PYTHON' -m py_compile 'C:\X-Plane 12\Resources\plugins\PythonPlugins\PI_VirtualATC.py'; "py_compile exit: \$LASTEXITCODE (0 = ok). Restart X-Plane to load it (no reload command in XPPython3 4.2)."
PS
    ;;
  start)
    cat <<PS | ps_run
if (-not (Get-ScheduledTask -TaskName $TASK -ErrorAction SilentlyContinue)) {
  \$a = New-ScheduledTaskAction -Execute 'C:\X-Plane 12\X-Plane.exe' -WorkingDirectory 'C:\X-Plane 12'
  \$p = New-ScheduledTaskPrincipal -UserId "\$env:COMPUTERNAME\\$BOX_USER" -LogonType Interactive -RunLevel Limited
  \$s = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit ([TimeSpan]::Zero)
  Register-ScheduledTask -TaskName $TASK -Action \$a -Principal \$p -Settings \$s -Force | Out-Null; "task registered"
}
Start-ScheduledTask -TaskName $TASK; Start-Sleep 5
"X-Plane running: " + [bool](Get-Process -Name 'X-Plane*' -ErrorAction SilentlyContinue) + "  (plugin tunes ~60 s after start; scripts/box.sh log)"
PS
    ;;
  stop)
    cat <<PS | ps_run
Stop-Process -Name 'X-Plane' -Force -ErrorAction SilentlyContinue; Start-Sleep 2; "X-Plane running: " + [bool](Get-Process -Name 'X-Plane*' -ErrorAction SilentlyContinue)
PS
    ;;
  log)
    cat <<PS | ps_run
Select-String -Path 'C:\X-Plane 12\XPPython3Log.txt' -Pattern 'VirtualATC|Traceback|Error' | ForEach-Object { \$_.Line } | Select-Object -Last ${2:-25}
PS
    ;;
  com1)
    khz="${2:?kHz, e.g. 121600}"
    cat <<PS | ps_run
\$udp = New-Object System.Net.Sockets.UdpClient
\$name = 'sim/cockpit2/radios/actuators/com1_frequency_hz_833'
\$nb = [System.Text.Encoding]::ASCII.GetBytes(\$name) + (New-Object byte[] (500 - \$name.Length))
\$pkt = [System.Text.Encoding]::ASCII.GetBytes("DREF") + 0 + [BitConverter]::GetBytes([single]$khz) + \$nb
\$udp.Send(\$pkt, \$pkt.Length, '127.0.0.1', 49000) | Out-Null; \$udp.Close(); "COM1 -> $khz kHz"
PS
    ;;
  deploy-tools)   # TEST ONLY: PI_ATCTestTools.py (atctest/place, atctest/ap_hold); restart X-Plane after
    scp -q "${SSH_OPTS[@]}" "$HERE/xplane_plugin/test_tools/PI_ATCTestTools.py" "$BOX_USER@$BOX_HOST:$PLUGIN_DIR/PI_ATCTestTools.py" \
      && echo "copied PI_ATCTestTools.py"
    ;;
  deploy-llmplays)   # "LLM plays X-Plane" API plugin (restart X-Plane after)
    scp -q "${SSH_OPTS[@]}" "$HERE/xplane_plugin/llm_plays/PI_LLMPlays.py" "$BOX_USER@$BOX_HOST:$PLUGIN_DIR/PI_LLMPlays.py" \
      && echo "copied PI_LLMPlays.py"
    cat <<PS | ps_run
& '$BOX_PYTHON' -m py_compile '$PLUGIN_DIR\PI_LLMPlays.py'; "py_compile exit: \$LASTEXITCODE"
PS
    ;;
  tunnel)            # forward the plugin API (127.0.0.1:5066 on the box) to localhost:5066 here
    if lsof -nP -iTCP:5066 -sTCP:LISTEN >/dev/null 2>&1; then echo "tunnel already up on :5066"; exit 0; fi
    ssh "${SSH_OPTS[@]}" -f -N -o ExitOnForwardFailure=yes -L 5066:127.0.0.1:5066 "$BOX_USER@$BOX_HOST" && echo "tunnel up: http://127.0.0.1:5066"
    ;;
  place)          # place <lat> <lon> <ele_ft> <hdg_true> <kt>  (needs deploy-tools)
    ele_m=$(python3 -c "print(round(${4:?ele_ft} * 0.3048, 1))"); spd=$(python3 -c "print(round(${6:?kt} * 0.514444, 1))")
    echo "${2:?lat} ${3:?lon} $ele_m ${5:?hdg} $spd" | ssh "${SSH_OPTS[@]}" "$BOX_USER@$BOX_HOST" \
      "powershell -NoProfile -Command \"\$input | Set-Content -Encoding ascii '$PLUGIN_DIR/atctest_place.txt'\""
    udp_cmd "atctest/place"
    ;;
  test) udp_cmd "virtualatc/test_transmit" ;;
  cmd)  udp_cmd "${2:?x-plane command path}" ;;
  ps)   if [ "${2:-}" = "-" ] || [ -z "${2:-}" ]; then ps_run; else ps_run < "$2"; fi ;;
  *) sed -n '2,15p' "$0"; exit 1 ;;
esac
