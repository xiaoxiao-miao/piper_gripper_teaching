import os


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default

# CAN / ROS2
CAN_PORT = os.getenv("CAN_PORT", "can1")
CAN_BITRATE = _env_int("CAN_BITRATE", 1000000)
ROS_PIPER_JOINT_STATE_TOPIC = "/joint_states_single"
ROS_PIPER_JOINT_CTRL_TOPIC = "/joint_ctrl_single"

# RealSense
REALSENSE_FPS = 10
REALSENSE_RESOLUTION = (1280, 720)
REALSENSE_ENABLE_DEPTH = True
COLOR_SAVE_JPEG_QUALITY = 85   # JPEG 품질 (0~100). PNG 대비 ~5배 압축

# Teach / Replay
TEACH_LOG_HZ = 50          # joint state 기록 주기 (Hz)
REPLAY_COMMAND_HZ = 50     # 속도와 무관한 고정 JointCtrl streaming 주기
REPLAY_DEFAULT_SPEED = 0.3  # 첫 replay 속도 배율
REPLAY_SMOOTHING_WINDOW = 11  # Savitzky-Golay window (홀수)
REPLAY_SMOOTHING_POLYORDER = 3

# Safety
MAX_JOINT_VELOCITY_RAD_S = 1.5   # replay 궤적 검사 임계값
MAX_WAYPOINT_DELTA_RAD = 0.5     # 연속 waypoint 간 최대 각도 차이
HOME_APPROACH_SPEED = 0.2        # home → trajectory[0] 이동 속도 배율
HOME_VELOCITY = 20               # safety-return 속도 (MotionCtrl_2 %)
HOME_WAIT_S = 8.0                # home 명령 후 완료 대기 시간 (초)
MOTOR_HANDSHAKE_TIMEOUT_S = 3.0  # /enable_srv 응답 + CAN feedback 대기
JOINT_FEEDBACK_MAX_AGE_S = 0.10  # 자동 이동 시작/실행 중 허용할 feedback age
# 이동 시작 전에는 위의 엄격한 기준을 사용한다. 이동 중 CAN/ROS 처리로 생기는
# 짧은 공백은 명령을 멈추고 기다리되, 이 시간을 넘긴 실제 단절만 abort한다.
JOINT_FEEDBACK_ABORT_AGE_S = 0.75
MOTION_START_TIMEOUT_S = 0.50    # target 발행 후 물리 움직임 시작 확인 시간
MOTION_START_MIN_DELTA_RAD = 0.005

# ROM calibration: measured extrema are used directly.  Piper's normal start
# pose places J2/J3 close to 0 rad, so an inward angular margin makes that
# manufacturer-provided pose unreachable.  Minimum measured span validation
# and the SDK hardware limits remain enforced independently.
CALIBRATION_MIN_SPAN_RAD = 0.25
# A manually taught trajectory can extend slightly beyond an incompletely
# swept calibration extremum.  Permit about 3 degrees around calibration-only
# bounds, while the SDK hardware bounds remain absolute.
CALIBRATION_LIMIT_TOLERANCE_RAD = 0.05
# With motors disabled, gravity sags J2/J3 a small amount past their hard-stop
# boundary (this is normal, not a fault) — the resting pose can end up a
# fraction of a radian outside the limits used for a fresh target move.
# _prepare_joint_motion()'s re-arm step holds the CURRENT measured pose, which
# by definition is already physically valid, so snap it to the boundary when
# it's outside by no more than this tolerance instead of rejecting the
# re-arm.  This never loosens validation for an actual commanded target.
REARM_HOLD_LIMIT_TOLERANCE_RAD = 0.05
# Piper URDF/vendor hard stops (rad).  Used for replay preflight until the
# driver publishes its live SDK motor-limit snapshot.
PIPER_HARDWARE_JOINT_LIMITS = [
    (-2.618, 2.618),
    (0.0, 3.14),
    (-2.967, 0.0),
    (-1.745, 1.745),
    (-1.22, 1.22),
    (-2.0944, 2.0944),
]

# Gripper (joint7)
GRIPPER_OPEN_RAD = 0.07          # 그리퍼 열림 위치 (rad)
GRIPPER_CLOSE_RAD = 0.0          # 그리퍼 닫힘 위치 (rad)
GRIPPER_TEACH_START_PCT = 70.0   # Teach 시작 시 기본 그리퍼 열림 위치 (%)
GRIPPER_VELOCITY = 5.0           # 그리퍼 이동 속도 (arm MotionCtrl % — 그리퍼 자체 속도는 GRIPPER_STEPS로 제어)
GRIPPER_STEPS = 8                # 그리퍼 이동 단계 수 (클수록 느림)
GRIPPER_STEP_DELAY = 0.08        # 각 단계 간 대기 시간 (초)
# 그리퍼 응답성 체크: 명령 후 실제 피드백이 이 정도는 움직여야 "반응함"으로 판단
# (물리 케이블 단선 등으로 명령이 씹히는 걸 감지 — UI 경고용).
GRIPPER_RESPONSE_MIN_FRAC = 0.25    # 의도한 이동량의 최소 비율
GRIPPER_RESPONSE_MIN_ABS_RAD = 0.003  # 절대 최소치 (아주 작은 이동 명령 대비)

# Safe return waypoint sequence (물리 로봇에서 실측 후 기입)
# 자동 home 복귀(retake_teach / add_take)는 이 waypoint 목록이 정의된 경우에만 허용.
# None이면 자동 이동을 거부하고 사용자가 수동으로 로봇을 초기화해야 함.
#
# 각 waypoint = [q1, q2, q3, q4, q5, q6] (rad).
# 순서대로 이동하며, 각 waypoint 도착 실패 시 그 자리에서 abort (다음 waypoint 진행 안 함).
# 최소 2개 권장: [clearance_pose, ready_pose]
# 예:
# SAFE_RETURN_WAYPOINTS = [
#     [0.0,  0.6, -0.8, 0.0,  0.8, 0.0],   # clearance: EE를 충분히 위로 올림
#     [0.0,  0.3, -0.5, 0.0,  0.8, 0.0],   # ready: 작업 대기 자세
# ]
SAFE_RETURN_WAYPOINTS = None

# Dataset
DATASET_PATH = os.getenv("DATASET_PATH", os.path.join(os.path.dirname(__file__), "dataset"))
JOINT_LIMITS_FILE = os.path.join(DATASET_PATH, "joint_limits.json")

# Postprocess defaults
POSTPROCESS_DEFAULT_MASK_FILL_COLOR = "#000000"

# Flask
FLASK_PORT = _env_int("FLASK_PORT", 5002)
HTTPS_PORT = _env_int("HTTPS_PORT", 5003)
FLASK_HOST = os.getenv("FLASK_HOST", "0.0.0.0")
FLASK_DEBUG = os.getenv("FLASK_DEBUG", "0").lower() in {"1", "true", "yes", "on"}
