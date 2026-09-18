"""
Piper Cowork — Flask + SocketIO 서버
포트: 5002

실행: python3 app.py
(start.sh에서 CAN 활성화 + ROS2 piper 노드 실행 후 이 서버 시작)
"""

import os
import sys
import signal
import threading
import time

from flask import Flask, Response, request, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO

# ── 경로 설정 ─────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from config import FLASK_HOST, FLASK_PORT, FLASK_DEBUG, HTTPS_PORT
from controller import Controller

# ── Flask 초기화 ──────────────────────────────────────────────────────
FRONTEND_BUILD = os.path.join(ROOT, "frontend", "build")
app = Flask(
    __name__,
    static_folder=FRONTEND_BUILD if os.path.exists(FRONTEND_BUILD) else None,
    static_url_path="",
)
CORS(app, resources={r"/api/*": {"origins": "*"}, r"/stream/*": {"origins": "*"}})
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# ── Controller 생성 (emit 콜백 주입) ─────────────────────────────────

def _emit(event: str, data: dict):
    socketio.emit(event, data)

controller = Controller(on_emit=_emit)
app.config["CONTROLLER"] = controller

# ── Blueprint 등록 ────────────────────────────────────────────────────
from routes.capture_routes import bp as capture_bp
from routes.robot_routes import bp as robot_bp
from routes.episode_routes import bp as episode_bp
from routes.mask_routes import bp as mask_bp
from routes.camera_routes import bp as camera_bp
from routes.urdf_routes import bp as urdf_bp
from routes.screenshot_routes import bp as screenshot_bp

app.register_blueprint(capture_bp)
app.register_blueprint(robot_bp)
app.register_blueprint(episode_bp)
app.register_blueprint(mask_bp)
app.register_blueprint(camera_bp)
app.register_blueprint(urdf_bp)
app.register_blueprint(screenshot_bp)

# ── MJPEG 카메라 스트리밍 ─────────────────────────────────────────────

MJPEG_STALL_TIMEOUT_S = 15.0  # 이 시간 동안 프레임을 한 장도 못 보내면 스레드를 접는다


def _mjpeg_generator(cam_id: str = "primary", mode: str = "rgb"):
    """cam_id: 'primary' | manifest camera id | legacy alias"""
    # jpg가 계속 None이면(카메라 다운 등) 소켓에 아무것도 안 쓰기 때문에, 이미 끊긴
    # 클라이언트를 서버가 절대 눈치채지 못해 이 제너레이터(=Werkzeug 스레드)가
    # 무한정 살아남는 스레드 누수가 있었음 — 일정 시간 프레임이 없으면 제너레이터를
    # 그냥 끝내서 스레드를 정리한다. 클라이언트(브라우저)는 자체 재시도 로직으로
    # 새 연결을 새로 맺는다.
    no_frame_since = time.monotonic()
    while True:
        mgr = controller._cameras
        if mgr is None:
            time.sleep(0.1)
            if time.monotonic() - no_frame_since > MJPEG_STALL_TIMEOUT_S:
                return
            continue
        jpg = mgr.get_jpeg_bytes(cam_id=cam_id, quality=70, mode=mode)
        if jpg is None:
            time.sleep(0.05)
            if time.monotonic() - no_frame_since > MJPEG_STALL_TIMEOUT_S:
                return
            continue
        no_frame_since = time.monotonic()
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
        )
        time.sleep(1.0 / 30)


def _mjpeg_response(cam_id: str, mode: str = "rgb"):
    response = Response(
        _mjpeg_generator(cam_id, mode),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["X-Accel-Buffering"] = "no"
    return response


@app.get("/stream/camera")
def stream_camera():
    """Primary stream: RealSense 우선, 없으면 첫 번째 웹캠."""
    return _mjpeg_response("primary")


@app.get("/stream/camera/<cam_id>")
def stream_camera_by_id(cam_id: str):
    """개별 카메라 스트림: manifest camera id 또는 legacy alias.
    ?mode=rgb|aolp|dolp|grid|stokes — polarizer(Spinnaker) 카메라 전용, 다른 카메라는 무시됨.
    """
    mgr = controller._cameras
    manifest = mgr.manifest if mgr else None
    from camera_manifest import camera_by_id, resolve_camera_id
    if cam_id.endswith("_depth"):
        base_id = cam_id[:-6]
        cam = camera_by_id(base_id, manifest)
        if not cam or "depth" not in cam.get("streams", []):
            return Response("Not found", status=404)
    elif resolve_camera_id(cam_id, manifest) is None:
        return Response("Not found", status=404)
    mode = request.args.get("mode", "rgb")
    return _mjpeg_response(cam_id, mode)


VENDOR_DIR = os.path.join(ROOT, "vendor")

@app.get("/vendor/<path:filename>")
def serve_vendor(filename):
    return send_from_directory(VENDOR_DIR, filename)


# ── SocketIO 이벤트 핸들러 ────────────────────────────────────────────

@socketio.on("connect")
def on_connect():
    # 현재 상태 전체를 클라이언트에 즉시 전송
    socketio.emit("mode_change", controller.get_status())
    controller._emit_robot_state()
    controller._emit_camera_state()
    controller._emit_logger_state()


@socketio.on("teach_start")
def on_teach_start(data):
    data = data or {}
    socketio.emit("ack", controller.start_teach(
        task=data.get("task", "unspecified"),
        operator=data.get("operator", "unknown"),
    ))


@socketio.on("teach_stop")
def on_teach_stop(_=None):
    socketio.emit("ack", controller.stop_teach())


@socketio.on("gripper_open")
def on_gripper_open(_=None):
    socketio.emit("ack", controller.gripper_open())


@socketio.on("gripper_close")
def on_gripper_close(_=None):
    socketio.emit("ack", controller.gripper_close())


@socketio.on("return_home")
def on_return_home(_=None):
    socketio.emit("ack", controller.return_home())


@socketio.on("replay_start")
def on_replay_start(data=None):
    data = data or {}
    socketio.emit("ack", controller.start_replay(
        speed_scale=data.get("speed_scale")
    ))


@socketio.on("replay_stop")
def on_replay_stop(_=None):
    socketio.emit("ack", controller.stop_replay())


@socketio.on("emergency_hold")
def on_emergency_hold(_=None):
    socketio.emit("ack", controller.hold_position())


@socketio.on("save_episode")
def on_save_episode(data=None):
    data = data or {}
    socketio.emit("ack", controller.save_episode(
        success=bool(data.get("success", True)),
        reason=str(data.get("reason", "")),
    ))


@socketio.on("discard_episode")
def on_discard_episode(_=None):
    socketio.emit("ack", controller.discard_episode())


@socketio.on("new_episode")
def on_new_episode(_=None):
    socketio.emit("ack", controller.new_episode())


@socketio.on("phone_mouse_cmd")
def on_phone_mouse_cmd(data):
    # data: { j1, j2, j4 (rad delta), seq }
    # No ack — 20Hz streaming, fire-and-forget
    controller.handle_phone_mouse_cmd(data or {})


@socketio.on("ee_twist_cmd")
def on_ee_twist_cmd(data):
    # data: { vx, vy, vz, wx, wy, wz (m/s or rad/s), dt (s) }
    # Cartesian jog via Pinocchio IK — no ack, 20Hz streaming
    controller.handle_ee_twist_cmd(data or {})

# ── Mobile control page ─────────────────────────────────────────────

MOBILE_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
<title>Piper Remote</title>
<script src="/vendor/socket.io.min.js"></script>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, sans-serif;
         background: #0f172a; color: #f1f5f9; min-height: 100vh;
         padding: 12px; }
  #mode-bar { background: #1e293b; border-radius: 12px; padding: 10px 14px;
              margin-bottom: 12px; display: flex; align-items: center; gap: 10px; }
  #mode-label { font-size: 11px; color: #94a3b8; text-transform: uppercase;
                letter-spacing: 1px; }
  #mode-badge { font-size: 14px; font-weight: 800; color: #f1f5f9; }
  #next-action { font-size: 12px; color: #64748b; margin-top: 2px; }
  #buttons { display: flex; flex-direction: column; gap: 10px; }
  .btn {
    width: 100%; padding: 20px 16px; border-radius: 14px; border: none;
    font-size: 18px; font-weight: 800; cursor: pointer;
    transition: opacity 0.1s; -webkit-tap-highlight-color: transparent;
    letter-spacing: 0.3px; touch-action: manipulation;
  }
  .btn:active { opacity: 0.75; }
  .btn:disabled { opacity: 0.35; cursor: default; }
  .btn-primary  { background: #6366f1; color: #fff; }
  .btn-success  { background: #22c55e; color: #fff; }
  .btn-danger   { background: #ef4444; color: #fff; }
  .btn-warning  { background: #f59e0b; color: #fff; }
  .btn-hold     { background: #1e293b; color: #f87171; border: 2px solid #ef4444;
                  font-size: 20px; padding: 24px 16px; }
  .btn-gray     { background: #334155; color: #cbd5e1; }
  .btn-free     { background: #1e293b; color: #34d399; border: 2px solid #34d399; }
  .btn-row      { display: flex; gap: 10px; }
  .btn-row .btn { flex: 1; }
  .freedrive-badge { background: #064e3b; color: #34d399; border-radius: 10px;
                     padding: 10px 14px; font-size: 14px; font-weight: 800;
                     text-align: center; margin-bottom: 4px; }
  #status-bar   { margin-top: 14px; background: #1e293b; border-radius: 10px;
                  padding: 8px 12px; font-size: 11px; color: #64748b; }
  #conn-dot     { display: inline-block; width: 8px; height: 8px; border-radius: 50%;
                  background: #ef4444; margin-right: 4px; }
  #conn-dot.ok  { background: #22c55e; }
  .section-label { font-size: 10px; text-transform: uppercase; letter-spacing: 1px;
                   color: #475569; margin-bottom: 4px; }
</style>
</head>
<body>
<div id="mode-bar">
  <div>
    <div style="display:flex;align-items:center;gap:6px">
      <span id="conn-dot"></span>
      <span id="mode-label">Mode</span>
    </div>
    <div id="mode-badge">—</div>
    <div id="next-action">Connecting...</div>
  </div>
</div>
<div id="buttons"></div>
<div id="status-bar">
  <span id="conn-dot-status"></span> Socket: <span id="sock-state">disconnected</span>
  &nbsp;|&nbsp; Take: <span id="take-info">—</span>
</div>

<script>
// Service Worker가 socket.io 폴링을 가로채는 것을 방지
if ('serviceWorker' in navigator) {
  navigator.serviceWorker.getRegistrations().then(regs => {
    if (regs.length > 0) {
      Promise.all(regs.map(r => r.unregister())).then(() => location.reload());
    }
  });
}

// io()는 현재 페이지 origin으로 자동 연결 (http:5002)
const socket = io();
let currentMode = '';
let availableActions = [];
let episodeId = '';
let motorsFree = false;
let disableCountdown = 0;
let disableTimerHandle = null;

// ── Calibration ──────────────────────────────────────────────────
let calibMin = Array(6).fill(Infinity);
let calibMax = Array(6).fill(-Infinity);
let calibResultLimits = null;
let calibServerValid = Array(6).fill(false);

function resetCalibTracking() {
  calibMin = Array(6).fill(Infinity);
  calibMax = Array(6).fill(-Infinity);
  calibServerValid = Array(6).fill(false);
}

function showCalibResult(limits) {
  calibResultLimits = limits;
  render(currentMode, availableActions, episodeId);
}

// ── Phone Mouse ───────────────────────────────────────────────────
let phoneMouse = false;
let pmDeadman = false;
let selectedTeachMode = 'freedrive';  // 'freedrive' | 'phone_mouse' — TEACH_READY에서 선택
let pmGyroPermission = false;
let pmIntervalHandle = null;
let pmBeta = 0, pmGamma = 0, pmAlpha = 0, pmAlphaPrev = 0;
const PM_DEAD_ZONE = 5;    // deg
const PM_SCALE = 0.001;    // deg → rad/step scale

window.addEventListener('deviceorientation', (e) => {
  pmBeta  = e.beta  || 0;
  pmGamma = e.gamma || 0;
  const alphaDelta = ((e.alpha || 0) - pmAlphaPrev + 540) % 360 - 180;
  pmAlphaPrev = e.alpha || 0;
  pmAlpha = alphaDelta;
});

function applyDeadZone(val, dz) {
  if (Math.abs(val) < dz) return 0;
  return val > 0 ? val - dz : val + dz;
}

function sendPhoneMouseCmd() {
  if (!pmDeadman) return;
  // gamma(roll) → j1(base rotation), beta(pitch) → j2(shoulder)
  // alpha(yaw rate) 는 노이즈가 심해 제외
  const j1 = applyDeadZone(pmGamma, PM_DEAD_ZONE) * PM_SCALE;
  const j2 = applyDeadZone(pmBeta,  PM_DEAD_ZONE) * PM_SCALE * -1;
  socket.emit('phone_mouse_cmd', { j1, j2, j4: 0, seq: Date.now() });
}

async function requestGyroPermission() {
  if (typeof DeviceOrientationEvent !== 'undefined' &&
      typeof DeviceOrientationEvent.requestPermission === 'function') {
    const perm = await DeviceOrientationEvent.requestPermission();
    pmGyroPermission = (perm === 'granted');
  } else {
    pmGyroPermission = true;  // Android: no permission needed
  }
  return pmGyroPermission;
}

function startDisableCountdown() {
  clearTimeout(disableTimerHandle);
  disableCountdown = 3;
  render(currentMode, availableActions, episodeId);
  function tick() {
    disableCountdown--;
    if (disableCountdown <= 0) {
      disableCountdown = 0;
      fetch('/api/robot/motors/disable', { method: 'POST' })
        .then(() => { motorsFree = true; render(currentMode, availableActions, episodeId); });
    } else {
      render(currentMode, availableActions, episodeId);
      disableTimerHandle = setTimeout(tick, 1000);
    }
  }
  disableTimerHandle = setTimeout(tick, 1000);
}

function cancelDisableCountdown() {
  clearTimeout(disableTimerHandle);
  disableTimerHandle = null;
  disableCountdown = 0;
  render(currentMode, availableActions, episodeId);
}

const MODE_COLORS = {
  TEACH_RECORDING: '#d97706',
  TEACH_STARTING: '#2563eb',
  TEACH_STOPPING: '#7c3aed',
  REPLAY_RECORDING: '#dc2626',
  REVIEW: '#6366f1',
  SAVED: '#22c55e',
  DISCARDED: '#6b7280',
  RETURN_HOME: '#0ea5e9',
};

const ACTION_DEFS = {
  connect:           { label: 'Connect System',          cls: 'btn-primary' },
  confirm_ready_pose:{ label: 'Confirm Ready Pose',      cls: 'btn-primary' },
  calibrate:         { label: 'Calibrate Joint Range',   cls: 'btn-gray' },
  stop_calibration:  { label: 'Finish Calibration',      cls: 'btn-success', large: true },
  resume_calibration:{ label: 'Resume Calibration',      cls: 'btn-free' },
  start_teach:       { label: 'Start Teaching',          cls: 'btn-success' },
  stop_teach:        { label: 'STOP TEACHING',           cls: 'btn-success', large: true },
  return_home:       { label: 'Return Home',             cls: 'btn-primary' },
  start_replay:      { label: 'Start Replay Recording',  cls: 'btn-primary' },
  stop_replay:       { label: 'STOP REPLAY',             cls: 'btn-danger',  large: true },
  save_episode:      { label: 'Save Episode (Success)',  cls: 'btn-success' },
  discard_episode:   { label: 'Discard Episode',        cls: 'btn-danger', confirm: 'Discard this episode?' },
  retake_replay:     { label: 'Re-replay (same traj)',  cls: 'btn-gray' },
  retake_teach:      { label: 'Re-teach (new recording)',cls: 'btn-gray' },
  new_episode:       { label: 'New Episode',             cls: 'btn-primary' },
  add_take:          { label: 'Add Another Take',        cls: 'btn-gray' },
};

const ACTION_APIS = {
  connect:            () => fetch('/api/connect',   { method: 'POST' }),
  confirm_ready_pose: () => fetch('/api/ready',     { method: 'POST' }),
  calibrate:          () => fetch('/api/robot/calibrate/start', { method: 'POST' }),
  stop_calibration:   () => fetch('/api/robot/calibrate/stop',  { method: 'POST' })
    .then(r => r.json()).then(data => {
      if (data.ok) showCalibResult(data.limits);
    }),
  resume_calibration: () => fetch('/api/robot/calibrate/resume', { method: 'POST' }),
  start_teach: async () => {
    if (selectedTeachMode === 'phone_mouse') {
      // Phone Mouse 모드: teach/start (→ TEACH_RECORDING 진입) 후 즉시 phone_mouse/enable (freedrive 타이머 취소)
      if (motorsFree) {
        await fetch('/api/robot/motors/enable', { method: 'POST' });
        motorsFree = false;
      }
      if (!pmGyroPermission) await requestGyroPermission();
      if (!pmGyroPermission) { alert('자이로 권한이 거부되었습니다.'); return; }
      await fetch('/api/teach/start', {
        method: 'POST', headers: {'Content-Type':'application/json'},
        body: JSON.stringify({ task: 'mobile', operator: 'mobile' })
      });
      phoneMouse = true;
      return fetch('/api/robot/phone_mouse/enable', { method: 'POST' });
    }
    // Freedrive 모드: 기존 flow (5초 카운트다운 후 모터 OFF)
    return fetch('/api/teach/start', {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ task: 'mobile', operator: 'mobile' })
    });
  },
  stop_teach:         () => fetch('/api/teach/stop', { method: 'POST' }),
  return_home:        () => fetch('/api/home', { method: 'POST' }),
  start_replay:       async () => {
    const response = await fetch('/api/replay/start', { method: 'POST' });
    const data = await response.json().catch(() => ({}));
    if (!response.ok || data.ok === false) {
      alert(data.reason || `Replay start failed (${response.status})`);
    }
    return data;
  },
  stop_replay:        () => fetch('/api/replay/stop', { method: 'POST' }),
  open_gripper:       () => fetch('/api/robot/gripper/open', { method: 'POST' }),
  close_gripper:      () => fetch('/api/robot/gripper/close',{ method: 'POST' }),
  hold_position:      () => fetch('/api/robot/hold', { method: 'POST' }),
  save_episode:       () => fetch('/api/episodes/save', {
    method: 'POST', headers: {'Content-Type':'application/json'},
    body: JSON.stringify({ success: true, reason: '' })
  }),
  discard_episode:    () => fetch('/api/episodes/discard', { method: 'POST' }),
  retake_replay:      () => fetch('/api/episodes/retake_replay', { method: 'POST' }),
  retake_teach:       () => fetch('/api/episodes/retake_teach', { method: 'POST' }),
  new_episode:        () => fetch('/api/episodes/new', { method: 'POST' }),
  add_take:           () => fetch('/api/episodes/add_take', { method: 'POST' }),
};

function render(mode, actions, episode) {
  const badgeEl = document.getElementById('mode-badge');
  badgeEl.textContent = mode;
  badgeEl.style.color = MODE_COLORS[mode] || '#f1f5f9';

  const container = document.getElementById('buttons');
  container.innerHTML = '';

  // Always show gripper + hold if in active states
  const activeStates = ['TEACH_RECORDING', 'REPLAY_READY', 'READY', 'TEACH_READY', 'REPLAY_RECORDING'];
  if (activeStates.includes(mode)) {
    const gripperRow = div('btn-row');
    gripperRow.appendChild(makeBtn('open_gripper',  'Open Gripper',  'btn-gray', actions));
    gripperRow.appendChild(makeBtn('close_gripper', 'Close Gripper', 'btn-gray', actions));
    container.appendChild(gripperRow);
  }

  // Main actions (ordered)
  const ORDER = [
    'connect', 'confirm_ready_pose',
    'calibrate', 'stop_calibration', 'resume_calibration',
    'start_teach', 'stop_teach',
    'return_home',
    'start_replay', 'stop_replay',
    'save_episode',
    'retake_replay', 'retake_teach',
    'new_episode', 'add_take',
    'discard_episode',
  ];
  ORDER.forEach(a => {
    if (!actions.includes(a)) return;
    const def = ACTION_DEFS[a];
    if (!def) return;
    const btn = makeBtn(a, def.label, def.cls, actions, def.confirm);
    if (a === 'stop_calibration') btn.disabled = !calibServerValid.every(Boolean);
    if (def.large) btn.style.padding = '28px 16px';
    container.appendChild(btn);
  });

  // ── CALIBRATING: 실시간 범위 표시 ────────────────────────────────
  if (mode === 'CALIBRATING') {
    const liveDiv = document.createElement('div');
    liveDiv.style.cssText = 'margin-top:10px;background:#1e293b;border-radius:10px;padding:10px 12px';

    const liveLabel = document.createElement('div');
    liveLabel.className = 'section-label';
    liveLabel.textContent = 'Measured Range (rad)';
    liveDiv.appendChild(liveLabel);

    const liveData = document.createElement('div');
    liveData.id = 'calib-live';
    liveData.style.cssText = 'font-family:monospace;font-size:11px;color:#7dd3fc;' +
      'margin-top:6px;line-height:1.8;white-space:pre-wrap';
    liveData.textContent = 'Move the arm freely — tracking range...';
    liveDiv.appendChild(liveData);
    container.appendChild(liveDiv);
  }

  // ── calibration 결과 표시 (한 번 완료 후) ────────────────────────
  if (calibResultLimits && ['READY','TEACH_READY'].includes(mode)) {
    const resDiv = document.createElement('div');
    resDiv.style.cssText = 'margin-top:10px;background:#052e16;border-radius:10px;padding:10px 12px';
    const resLabel = document.createElement('div');
    resLabel.className = 'section-label';
    resLabel.style.color = '#4ade80';
    resLabel.textContent = 'Saved Joint Limits';
    resDiv.appendChild(resLabel);
    const resData = document.createElement('div');
    resData.style.cssText = 'font-family:monospace;font-size:11px;color:#86efac;' +
      'margin-top:6px;line-height:1.8;white-space:pre-wrap';
    const labels = ['J1','J2','J3','J4','J5','J6'];
    resData.textContent = labels.map((l,i) => {
      const k = 'j' + (i+1);
      const v = calibResultLimits[k];
      const tag = v && v.measured ? '' : ' (default)';
      return v ? l + ': [' + v.min.toFixed(3) + ', ' + v.max.toFixed(3) + ']' + tag : '';
    }).filter(Boolean).join('\\n');
    resDiv.appendChild(resData);
    container.appendChild(resDiv);
  }

  // ── Freedrive / Phone Mouse 섹션 ────────────────────────────────
  const freedriveStates = ['READY', 'TEACH_READY', 'REPLAY_READY'];

  if (mode === 'TEACH_RECORDING') {
    // TEACH_RECORDING: TEACH_READY에서 선택한 모드만 표시 (모드 전환 없음)

    if (selectedTeachMode === 'phone_mouse') {
      // ── Phone Mouse 모드만 표시 ────────────────────────────────────
      const pmSep = document.createElement('div');
      pmSep.style.cssText = 'margin-top:12px;border-top:1px solid #1e293b;padding-top:12px';
      const pmLabel = document.createElement('div');
      pmLabel.className = 'section-label';
      pmLabel.textContent = 'Phone Mouse (Teaching)';
      pmSep.appendChild(pmLabel);

      const statusDiv = document.createElement('div');
      statusDiv.id = 'pm-status';
      statusDiv.style.cssText = 'background:#1e3a5f;border-radius:8px;padding:8px;' +
        'font-size:12px;color:#93c5fd;margin-bottom:8px;font-family:monospace';
      statusDiv.textContent = 'gamma(roll): — beta(pitch): —';
      pmSep.appendChild(statusDiv);

      const deadman = document.createElement('button');
      deadman.className = 'btn btn-danger';
      deadman.style.padding = '36px 16px';
      deadman.style.fontSize = '22px';
      deadman.textContent = 'HOLD TO MOVE';
      deadman.addEventListener('touchstart', (e) => {
        e.preventDefault();
        pmDeadman = true;
        if (!pmIntervalHandle)
          pmIntervalHandle = setInterval(sendPhoneMouseCmd, 50);
      });
      ['touchend', 'touchcancel'].forEach(ev => deadman.addEventListener(ev, () => {
        pmDeadman = false;
        clearInterval(pmIntervalHandle);
        pmIntervalHandle = null;
      }));
      pmSep.appendChild(deadman);

      if (!window._pmDisplayInterval) {
        window._pmDisplayInterval = setInterval(() => {
          const el = document.getElementById('pm-status');
          if (!el || !phoneMouse) {
            clearInterval(window._pmDisplayInterval);
            window._pmDisplayInterval = null;
            return;
          }
          el.textContent =
            'gamma(roll): ' + pmGamma.toFixed(1) + 'deg  ' +
            'beta(pitch): ' + pmBeta.toFixed(1) + 'deg';
        }, 100);
      }
      container.appendChild(pmSep);

    } else {
      // ── Freedrive 모드만 표시 ──────────────────────────────────────
      const fdSep = document.createElement('div');
      fdSep.style.cssText = 'margin-top:12px;border-top:1px solid #1e293b;padding-top:12px';
      const fdLabel = document.createElement('div');
      fdLabel.className = 'section-label';
      fdLabel.textContent = 'Freedrive (Motor Off)';
      fdSep.appendChild(fdLabel);

      if (motorsFree) {
        const badge = document.createElement('div');
        badge.className = 'freedrive-badge';
        badge.textContent = 'ARM IS FREE — guide by hand';
        fdSep.appendChild(badge);
        const reBtn = document.createElement('button');
        reBtn.className = 'btn btn-primary';
        reBtn.style.marginTop = '6px';
        reBtn.textContent = 'Re-enable Motors';
        reBtn.onclick = () => {
          fetch('/api/robot/motors/enable', { method: 'POST' })
            .then(() => { motorsFree = false; render(currentMode, availableActions, episodeId); });
        };
        fdSep.appendChild(reBtn);
      } else if (disableCountdown > 0) {
        const countBadge = document.createElement('div');
        countBadge.style.cssText = 'background:#7c2d12;color:#fcd34d;border-radius:10px;' +
          'padding:10px 14px;font-size:16px;font-weight:800;text-align:center;margin-bottom:6px';
        countBadge.textContent = 'Releasing in ' + disableCountdown + 's...';
        fdSep.appendChild(countBadge);
        const cancelBtn = document.createElement('button');
        cancelBtn.className = 'btn btn-gray';
        cancelBtn.textContent = 'Cancel';
        cancelBtn.onclick = cancelDisableCountdown;
        fdSep.appendChild(cancelBtn);
      } else {
        const freeBtn = document.createElement('button');
        freeBtn.className = 'btn btn-free';
        freeBtn.textContent = 'Release Motors (Freedrive)';
        freeBtn.onclick = startDisableCountdown;
        fdSep.appendChild(freeBtn);
      }
      container.appendChild(fdSep);
    }

  } else if (freedriveStates.includes(mode)) {
    // TEACH_READY: Teaching Mode 선택기 + 시작 자세 Freedrive
    if (mode === 'TEACH_READY') {
      const modeDiv = document.createElement('div');
      modeDiv.style.cssText = 'margin-top:12px;border-top:1px solid #1e293b;padding-top:12px';
      const modeLabel = document.createElement('div');
      modeLabel.className = 'section-label';
      modeLabel.textContent = 'Teaching Mode';
      modeDiv.appendChild(modeLabel);

      const modeRow = document.createElement('div');
      modeRow.className = 'btn-row';

      const fdModeBtn = document.createElement('button');
      fdModeBtn.className = 'btn ' + (selectedTeachMode === 'freedrive' ? 'btn-success' : 'btn-gray');
      fdModeBtn.textContent = 'Freedrive';
      fdModeBtn.onclick = () => { selectedTeachMode = 'freedrive'; render(currentMode, availableActions, episodeId); };

      const pmModeBtn = document.createElement('button');
      pmModeBtn.className = 'btn ' + (selectedTeachMode === 'phone_mouse' ? 'btn-warning' : 'btn-gray');
      pmModeBtn.textContent = 'Phone Mouse';
      pmModeBtn.onclick = () => { selectedTeachMode = 'phone_mouse'; render(currentMode, availableActions, episodeId); };

      modeRow.appendChild(fdModeBtn);
      modeRow.appendChild(pmModeBtn);
      modeDiv.appendChild(modeRow);
      container.appendChild(modeDiv);
    }

    // READY / TEACH_READY / REPLAY_READY: 시작 자세 설정용 Freedrive
    const sep = document.createElement('div');
    sep.style.cssText = 'margin-top:12px;border-top:1px solid #1e293b;padding-top:12px';
    const label = document.createElement('div');
    label.className = 'section-label';
    label.textContent = 'Start Position (Freedrive)';
    sep.appendChild(label);

    if (motorsFree) {
      const badge = document.createElement('div');
      badge.className = 'freedrive-badge';
      badge.textContent = 'ARM IS FREE — guide by hand';
      sep.appendChild(badge);
      const reBtn = document.createElement('button');
      reBtn.className = 'btn btn-primary';
      reBtn.textContent = 'Re-enable Motors';
      reBtn.style.marginTop = '6px';
      reBtn.onclick = () => {
        fetch('/api/robot/motors/enable', { method: 'POST' })
          .then(() => { motorsFree = false; render(currentMode, availableActions, episodeId); });
      };
      sep.appendChild(reBtn);
    } else if (disableCountdown > 0) {
      const countBadge = document.createElement('div');
      countBadge.style.cssText = 'background:#7c2d12;color:#fcd34d;border-radius:10px;' +
        'padding:10px 14px;font-size:16px;font-weight:800;text-align:center;margin-bottom:6px';
      countBadge.textContent = 'Releasing in ' + disableCountdown + 's...';
      sep.appendChild(countBadge);
      const cancelBtn = document.createElement('button');
      cancelBtn.className = 'btn btn-gray';
      cancelBtn.textContent = 'Cancel';
      cancelBtn.onclick = cancelDisableCountdown;
      sep.appendChild(cancelBtn);
    } else {
      const freeBtn = document.createElement('button');
      freeBtn.className = 'btn btn-free';
      freeBtn.textContent = 'Release Motors (Freedrive)';
      freeBtn.onclick = startDisableCountdown;
      sep.appendChild(freeBtn);
    }
    container.appendChild(sep);
  } else {
    // 다른 상태: Freedrive / Phone Mouse 자동 종료
    if (phoneMouse) {
      phoneMouse = false; pmDeadman = false;
      clearInterval(pmIntervalHandle); pmIntervalHandle = null;
    }
    if (disableCountdown > 0) {
      clearTimeout(disableTimerHandle); disableTimerHandle = null; disableCountdown = 0;
    }
    if (motorsFree) motorsFree = false;
  }

  // Always-visible hold button when robot is active
  if (['TEACH_RECORDING', 'RETURN_HOME', 'REPLAY_RECORDING', 'REPLAY_READY', 'TEACH_READY', 'READY'].includes(mode)) {
    const sep2 = document.createElement('div');
    sep2.style.cssText = 'margin-top:12px;border-top:1px solid #1e293b;padding-top:12px';
    const holdBtn = document.createElement('button');
    holdBtn.className = 'btn btn-hold';
    holdBtn.textContent = 'HOLD POSITION';
    holdBtn.onclick = () => fetch('/api/robot/hold', { method: 'POST' });
    sep2.appendChild(holdBtn);
    container.appendChild(sep2);
  }

  document.getElementById('take-info').textContent = episode || '—';
}

function makeBtn(action, label, cls, actions, confirmMsg) {
  const btn = document.createElement('button');
  btn.className = 'btn ' + cls;
  btn.textContent = label;
  const enabled = actions.includes(action);
  btn.disabled = !enabled;
  if (enabled) {
    btn.onclick = () => {
      if (confirmMsg && !confirm(confirmMsg)) return;
      const fn = ACTION_APIS[action];
      if (fn) fn().catch(e => console.error(e));
    };
  }
  return btn;
}

function div(cls) {
  const d = document.createElement('div');
  d.className = cls;
  return d;
}

socket.on('connect', () => {
  document.getElementById('sock-state').textContent = 'connected';
  document.getElementById('conn-dot').classList.add('ok');
});
socket.on('disconnect', () => {
  document.getElementById('sock-state').textContent = 'disconnected';
  document.getElementById('conn-dot').classList.remove('ok');
});
socket.on('mode_change', data => {
  const prev = currentMode;
  currentMode = data.mode;
  availableActions = data.available_actions || [];
  document.getElementById('next-action').textContent = data.next_action || '';
  if (prev !== 'CALIBRATING' && currentMode === 'CALIBRATING') resetCalibTracking();
  render(currentMode, availableActions, episodeId);
});
socket.on('robot_state', data => {
  if (currentMode !== 'CALIBRATING') return;
  if (!data.position) return;
  if (Array.isArray(data.calibration_valid)) calibServerValid = data.calibration_valid;
  for (let i = 0; i < 6; i++) {
    const v = data.position[i];
    if (v < calibMin[i]) calibMin[i] = v;
    if (v > calibMax[i]) calibMax[i] = v;
  }
  // live update calibration display
  const el = document.getElementById('calib-live');
  if (!el) return;
  const labels = ['J1','J2','J3','J4','J5','J6'];
  el.textContent = labels.map((l,i) => {
    const lo = calibMin[i] === Infinity  ? '—' : calibMin[i].toFixed(2);
    const hi = calibMax[i] === -Infinity ? '—' : calibMax[i].toFixed(2);
    const span = data.calibration_span && data.calibration_span[i];
    const valid = calibServerValid[i] ? ' ✓' : ' ✗';
    return l + ': [' + lo + ', ' + hi + '] span=' +
      (span == null ? '—' : span.toFixed(3)) + valid;
  }).join('  ');
  const finish = Array.from(document.querySelectorAll('button'))
    .find(b => b.textContent === 'Finish Calibration');
  if (finish) finish.disabled = !calibServerValid.every(Boolean);
});
socket.on('logger_state', data => {
  if (data.episode_id) episodeId = data.episode_id;
  document.getElementById('take-info').textContent = data.episode_id || '—';
});
</script>
</body>
</html>"""


# ── SPA fallback — / 와 /mobile 모두 React build index.html 서빙 ────

@app.get("/")
@app.get("/mobile")
@app.get("/<path:path>")
def serve_spa(path=""):
    if os.path.exists(FRONTEND_BUILD):
        index = os.path.join(FRONTEND_BUILD, "index.html")
        if os.path.exists(index):
            response = send_from_directory(FRONTEND_BUILD, "index.html")
            # index.html contains the current hashed JS filename. Caching an
            # older index keeps clients on an obsolete UI even after a build.
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
            return response
    return "<h2>Piper Cowork backend running on port 5002.<br>Build the React frontend to serve the UI.</h2>", 200

# ── 종료 처리 ─────────────────────────────────────────────────────────

_SHUTTING_DOWN = False


def _shutdown(sig, frame):
    global _SHUTTING_DOWN
    if _SHUTTING_DOWN:
        return
    _SHUTTING_DOWN = True
    print("\n[app] Shutting down...")
    controller.shutdown()
    raise SystemExit(0)


signal.signal(signal.SIGINT, _shutdown)
signal.signal(signal.SIGTERM, _shutdown)

# ── 진입점 ───────────────────────────────────────────────────────────

def _get_ssl_context():
    """
    서버 IP를 SAN에 포함한 self-signed cert 자동 생성.
    Chrome 58+ / iOS Safari는 SAN 없는 cert의 WebSocket을 차단하므로 반드시 필요.
    폰 브라우저에서 인증서 경고 → "고급" → "계속 진행" 으로 우회.
    """
    import datetime
    import ipaddress
    import socket as _socket

    cert_path = os.path.join(ROOT, "cert.pem")
    key_path  = os.path.join(ROOT, "key.pem")

    def _local_ip():
        try:
            s = _socket.socket(_socket.AF_INET, _socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "127.0.0.1"

    local_ip = _local_ip()

    # 기존 cert가 현재 IP를 SAN에 포함하는지 확인 → 다르면 재생성
    def _needs_regen():
        if not (os.path.exists(cert_path) and os.path.exists(key_path)):
            return True
        try:
            from cryptography import x509
            with open(cert_path, "rb") as f:
                cert = x509.load_pem_x509_certificate(f.read())
            san = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName)
            ips = [str(ip) for ip in san.value.get_values_for_type(x509.IPAddress)]
            return local_ip not in ips
        except Exception:
            return True

    if _needs_regen():
        try:
            from cryptography import x509
            from cryptography.x509.oid import NameOID, ExtendedKeyUsageOID
            from cryptography.hazmat.primitives import hashes, serialization
            from cryptography.hazmat.primitives.asymmetric import rsa

            key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
            name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "piper-cowork")])

            san_entries = [
                x509.DNSName("localhost"),
                x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")),
            ]
            if local_ip != "127.0.0.1":
                san_entries.append(x509.IPAddress(ipaddress.IPv4Address(local_ip)))

            cert = (
                x509.CertificateBuilder()
                .subject_name(name)
                .issuer_name(name)
                .public_key(key.public_key())
                .serial_number(x509.random_serial_number())
                .not_valid_before(datetime.datetime.utcnow())
                .not_valid_after(datetime.datetime.utcnow() + datetime.timedelta(days=3650))
                .add_extension(x509.SubjectAlternativeName(san_entries), critical=False)
                .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
                .add_extension(
                    x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]),
                    critical=False,
                )
                .sign(key, hashes.SHA256())
            )

            with open(cert_path, "wb") as f:
                f.write(cert.public_bytes(serialization.Encoding.PEM))
            with open(key_path, "wb") as f:
                f.write(key.private_bytes(
                    serialization.Encoding.PEM,
                    serialization.PrivateFormat.TraditionalOpenSSL,
                    serialization.NoEncryption(),
                ))
            print(f"[app] SSL cert generated (SAN: {local_ip}): {cert_path}")
        except Exception as e:
            print(f"[app] WARNING: SSL cert generation failed ({e}). Running HTTP.")
            return None, "http"

    return (cert_path, key_path), "https"


def _run_https_server(ssl_context):
    """
    포트 5003에서 HTTPS 전용 서버를 별도 스레드로 실행.
    모바일 /mobile 페이지만 서빙 (gyro를 위한 HTTPS 필요).
    """
    from werkzeug.serving import make_server
    import ssl as _ssl

    cert_path, key_path = ssl_context
    ctx = _ssl.SSLContext(_ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(cert_path, key_path)

    srv = make_server(FLASK_HOST, HTTPS_PORT, app, ssl_context=ctx, threaded=True)
    print(f"[app] HTTPS server (mobile) at https://{FLASK_HOST}:{HTTPS_PORT}")
    srv.serve_forever()


if __name__ == "__main__":
    ssl_context, proto = _get_ssl_context()

    # HTTPS 서버: 별도 포트(5003)에서 모바일 전용
    if ssl_context is not None:
        t = threading.Thread(target=_run_https_server, args=(ssl_context,), daemon=True)
        t.start()

    # HTTP 서버: 메인 포트(5002) — PC React proxy, Socket.IO 모두 정상 동작
    print(f"[app] Starting Piper Cowork server at http://{FLASK_HOST}:{FLASK_PORT}")
    socketio.run(
        app,
        host=FLASK_HOST,
        port=FLASK_PORT,
        debug=FLASK_DEBUG,
        use_reloader=False,
        allow_unsafe_werkzeug=True,
        ssl_context=None,  # HTTP
    )
