#!/usr/bin/env bash
# Piper Cowork launcher.
#
# Copy start.env.example to .env and adjust paths for your machine.
# This script intentionally kills only processes it starts.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ -f "$SCRIPT_DIR/.env" ]]; then
    # shellcheck disable=SC1091
    source "$SCRIPT_DIR/.env"
fi

PROCESS_GROUPS=()
PROCESS_GROUP_NAMES=()
CLEANED_UP=0

bool_enabled() {
    case "${1:-}" in
        1|true|TRUE|yes|YES|on|ON) return 0 ;;
        *) return 1 ;;
    esac
}

cleanup() {
    local code=$?
    if [[ "$CLEANED_UP" == "1" ]]; then
        exit "$code"
    fi
    CLEANED_UP=1
    echo ""
    echo "[start.sh] Stopping Piper Cowork..."

    local index pgid name
    for index in "${!PROCESS_GROUPS[@]}"; do
        pgid="${PROCESS_GROUPS[$index]}"
        name="${PROCESS_GROUP_NAMES[$index]}"
        if kill -0 -- "-$pgid" 2>/dev/null; then
            if [[ "$name" == "Piper ROS" ]]; then
                echo "[start.sh] SIGINT process group $pgid ($name)"
                kill -INT -- "-$pgid" 2>/dev/null || true
            else
                echo "[start.sh] SIGTERM process group $pgid ($name)"
                kill -TERM -- "-$pgid" 2>/dev/null || true
            fi
        fi
    done

    # Give rclpy/Flask enough time to close DDS, cameras and CAN cleanly.
    for _ in 1 2 3 4 5 6; do
        local alive=0
        for pgid in "${PROCESS_GROUPS[@]}"; do
            if kill -0 -- "-$pgid" 2>/dev/null; then
                alive=1
                break
            fi
        done
        [[ "$alive" == "0" ]] && break
        sleep 0.5
    done

    for index in "${!PROCESS_GROUPS[@]}"; do
        pgid="${PROCESS_GROUPS[$index]}"
        name="${PROCESS_GROUP_NAMES[$index]}"
        if kill -0 -- "-$pgid" 2>/dev/null; then
            echo "[start.sh] SIGTERM process group $pgid ($name)"
            kill -TERM -- "-$pgid" 2>/dev/null || true
        fi
    done
    sleep 0.5
    for index in "${!PROCESS_GROUPS[@]}"; do
        pgid="${PROCESS_GROUPS[$index]}"
        name="${PROCESS_GROUP_NAMES[$index]}"
        if kill -0 -- "-$pgid" 2>/dev/null; then
            echo "[start.sh] SIGKILL process group $pgid ($name)"
            kill -KILL -- "-$pgid" 2>/dev/null || true
        fi
    done

    echo "[start.sh] Done."
    exit "$code"
}

trap cleanup SIGINT SIGTERM EXIT

if ! command -v setsid >/dev/null 2>&1; then
    echo "[start.sh] ERROR: setsid is required for reliable child-process cleanup."
    exit 1
fi

# ── User-configurable defaults ────────────────────────────────────────

CAN_PORT="${CAN_PORT:-can1}"
CAN_BITRATE="${CAN_BITRATE:-1000000}"

# Auto-detect the gs_usb CAN adapter USB address from sysfs unless overridden in .env
_detect_can_usb() {
    local dev
    for dev in /sys/bus/usb/devices/*/; do
        if [[ "$(readlink "$dev/driver" 2>/dev/null | xargs basename 2>/dev/null)" == "gs_usb" ]]; then
            basename "$dev"
            return 0
        fi
    done
    return 1
}
if [[ -z "${PIPER_CAN_USB:-}" ]]; then
    if _detected=$(_detect_can_usb); then
        PIPER_CAN_USB="$_detected"
        echo "[start.sh] Auto-detected CAN USB address: $PIPER_CAN_USB"
    else
        echo "[start.sh] WARNING: gs_usb CAN adapter not found; can_activate.sh will try without address"
        PIPER_CAN_USB=""
    fi
fi
unset -f _detect_can_usb

PIPER_WS_DIR="${PIPER_WS_DIR:-$HOME/robotarm/piper_ws/src/Piper_ros}"
PIPER_ROS_SETUP_FILES="${PIPER_ROS_SETUP_FILES:-$HOME/robotarm/install/setup.bash:$HOME/agilex_ws/install/setup.bash}"
PIPER_ROS_PACKAGE="${PIPER_ROS_PACKAGE:-piper}"
PIPER_ROS_EXECUTABLE="${PIPER_ROS_EXECUTABLE:-piper_single_ctrl}"

PIPER_START_ROS="${PIPER_START_ROS:-1}"
PIPER_ACTIVATE_CAN="${PIPER_ACTIVATE_CAN:-1}"
# Cowork performs an explicit, feedback-verified enable handshake before
# motion.  Startup auto-enable can race CAN discovery and used to terminate
# the driver after five seconds, so keep it disabled by default.
PIPER_AUTO_ENABLE="${PIPER_AUTO_ENABLE:-false}"
PIPER_GRIPPER_EXIST="${PIPER_GRIPPER_EXIST:-true}"
PIPER_BUILD_FRONTEND="${PIPER_BUILD_FRONTEND:-0}"
PIPER_LOCAL_DDS="${PIPER_LOCAL_DDS:-1}"

FLASK_HOST="${FLASK_HOST:-0.0.0.0}"
FLASK_PORT="${FLASK_PORT:-5002}"
HTTPS_PORT="${HTTPS_PORT:-5003}"
export CAN_PORT CAN_BITRATE FLASK_HOST FLASK_PORT HTTPS_PORT

# A single CAN interface must have exactly one Piper controller.  Keep the
# lock FD inherited by child processes so an orphaned driver also blocks a
# second launcher, and reject drivers started outside this script explicitly.
if bool_enabled "$PIPER_START_ROS"; then
    PIPER_LOCK_ROOT="${XDG_RUNTIME_DIR:-/tmp}"
    PIPER_LOCK_FILE="${PIPER_LOCK_ROOT}/piper_cowork_${CAN_PORT}_$(id -u).lock"
    exec {PIPER_LOCK_FD}>"$PIPER_LOCK_FILE"
    if ! flock -n "$PIPER_LOCK_FD"; then
        echo "[start.sh] ERROR: another Piper Cowork instance owns ${CAN_PORT}."
        echo "           Stop the existing start.sh before launching another instance."
        exit 1
    fi

    mapfile -t stale_piper_pids < <(
        pgrep -u "$(id -u)" -f "[p]iper_single_ctrl.*can_port:=${CAN_PORT}" || true
    )
    if (( ${#stale_piper_pids[@]} > 0 )); then
        echo "[start.sh] ERROR: existing Piper driver(s) already use ${CAN_PORT}: ${stale_piper_pids[*]}"
        echo "           Stop those drivers before restarting Piper Cowork."
        exit 1
    fi
    echo "[start.sh] Acquired single-driver lock: $PIPER_LOCK_FILE"
fi

if [[ -x "$SCRIPT_DIR/.venv/bin/python" ]]; then
    PYTHON_BIN="${PYTHON_BIN:-$SCRIPT_DIR/.venv/bin/python}"
else
    PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

echo "[start.sh] Piper Cowork root: $SCRIPT_DIR"
echo "[start.sh] Backend: http://localhost:${FLASK_PORT}"

# ── Optional frontend build ───────────────────────────────────────────

if bool_enabled "$PIPER_BUILD_FRONTEND"; then
    if command -v npm >/dev/null 2>&1; then
        echo "[start.sh] Building frontend..."
        (cd "$SCRIPT_DIR/frontend" && npm install && npm run build)
    else
        echo "[start.sh] WARNING: npm not found; skipping frontend build"
    fi
elif [[ ! -d "$SCRIPT_DIR/frontend/build" ]]; then
    echo "[start.sh] WARNING: frontend/build not found."
    echo "           Run: cd frontend && npm install && npm run build"
fi

# ── ROS2 environment ─────────────────────────────────────────────────

if bool_enabled "$PIPER_START_ROS"; then
    echo "[start.sh] Sourcing ROS2 setup files..."
    # ROS/ament/colcon setup scripts check many variables (COLCON_TRACE, AMENT_TRACE_SETUP_FILES,
    # AMENT_PYTHON_EXECUTABLE, ...) without guarding against set -u. Temporarily relax -u while
    # sourcing so we don't have to enumerate every variable they touch.
    set +u
    IFS=":" read -r -a setup_files <<< "$PIPER_ROS_SETUP_FILES"
    for setup_file in "${setup_files[@]}"; do
        if [[ -f "$setup_file" ]]; then
            echo "[start.sh]   source $setup_file"
            # shellcheck disable=SC1090
            source "$setup_file"
        else
            echo "[start.sh]   skip missing $setup_file"
        fi
    done
    set -u

    if ! command -v ros2 >/dev/null 2>&1; then
        echo "[start.sh] ERROR: ros2 command not found."
        echo "           Set PIPER_ROS_SETUP_FILES in .env or run with PIPER_START_ROS=0 for UI/backend only."
        exit 1
    fi

    if bool_enabled "$PIPER_LOCAL_DDS"; then
        # TurtleBot shells commonly export a remote Fast DDS discovery server.
        # Piper Cowork's driver and Flask ROS node run on this same host; forcing
        # loopback discovery keeps them connected even when that server is absent.
        unset ROS_DISCOVERY_SERVER
        unset FASTRTPS_DEFAULT_PROFILES_FILE
        export ROS_LOCALHOST_ONLY=1
        echo "[start.sh] ROS discovery: local host only (external discovery overrides cleared)"
    fi
fi

# ── CAN activation ───────────────────────────────────────────────────

if bool_enabled "$PIPER_START_ROS" && bool_enabled "$PIPER_ACTIVATE_CAN"; then
    echo "[start.sh] Activating CAN interface (${CAN_PORT}, ${CAN_BITRATE})..."
    if [[ -x "$PIPER_WS_DIR/can_activate.sh" || -f "$PIPER_WS_DIR/can_activate.sh" ]]; then
        (cd "$PIPER_WS_DIR" && bash can_activate.sh "$CAN_PORT" "$CAN_BITRATE" "$PIPER_CAN_USB")
    else
        echo "[start.sh] WARNING: can_activate.sh not found at $PIPER_WS_DIR; skipping CAN activation"
    fi
fi

# ── Piper ROS node ───────────────────────────────────────────────────

if bool_enabled "$PIPER_START_ROS"; then
    echo "[start.sh] Starting ${PIPER_ROS_PACKAGE}/${PIPER_ROS_EXECUTABLE}..."
    setsid ros2 run "$PIPER_ROS_PACKAGE" "$PIPER_ROS_EXECUTABLE" \
        --ros-args \
        -p can_port:="$CAN_PORT" \
        -p auto_enable:="$PIPER_AUTO_ENABLE" \
        -p gripper_exist:="$PIPER_GRIPPER_EXIST" &
    PIPER_PID=$!
    PROCESS_GROUPS+=("$PIPER_PID")
    PROCESS_GROUP_NAMES+=("Piper ROS")
    echo "[start.sh] Piper ROS PID/PGID: $PIPER_PID"
    sleep 2
fi

# ── Flask app ────────────────────────────────────────────────────────

echo "[start.sh] Starting Flask server..."
setsid "$PYTHON_BIN" app.py &
APP_PID=$!
PROCESS_GROUPS+=("$APP_PID")
PROCESS_GROUP_NAMES+=("Flask")
echo "[start.sh] Flask PID/PGID: $APP_PID"

echo ""
echo "══════════════════════════════════════════"
echo "  Piper Cowork"
echo "  UI:      http://localhost:${FLASK_PORT}"
echo "  Mobile:  https://localhost:${HTTPS_PORT}/mobile"
echo "  Ctrl+C to stop"
echo "══════════════════════════════════════════"
echo ""

wait
