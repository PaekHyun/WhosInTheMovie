import sys
import os
import string
import subprocess

# pyinstaller build command :
# pyinstaller app.py ^
#  --noconsole ^
#  --collect-all torch ^
#  --collect-all ultralytics ^
#  --add-data "yolov8n-seg.pt;."


# =========================
# GUI 프로세스에서는 CUDA 차단
# =========================
if __name__ == "__main__" and len(sys.argv) == 1:
    # GUI 모드
    os.environ["CUDA_VISIBLE_DEVICES"] = ""


# =========================
# 공통: 모델 경로 헬퍼
# =========================
def get_resource_path(filename):
    if getattr(sys, "frozen", False):
        return os.path.join(os.path.dirname(sys.executable), filename)
    return os.path.join(os.path.dirname(__file__), filename)

# =========================
# 웹캠 자동 탐색
# =========================
def find_camera(max_index=5):
    import cv2
    for i in [1, 0, 2, 3, 4]:
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            cap.release()
            return i
        cap.release()
    return None

# =========================
# 영상 처리 (CUDA 전용 프로세스)
# =========================
def process_video(video_path):
    import os
    import sys
    import queue
    import threading

    # PyInstaller 환경일 때 DLL 경로를 PATH에 추가
    if getattr(sys, 'frozen', False):
        curr_dir = os.path.dirname(sys.executable)
        torch_lib_path = os.path.join(curr_dir, "_internal", "torch", "lib")
        if os.path.exists(torch_lib_path):
            os.add_dll_directory(torch_lib_path)
            os.environ["PATH"] = torch_lib_path + os.pathsep + os.environ["PATH"]

    import cv2
    import torch
    import numpy as np
    from ultralytics import YOLO
    from ffpyplayer.player import MediaPlayer

    WIDTH, HEIGHT = 1920, 1080

    model_path = get_resource_path("yolov8s-seg.pt")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("YOLO device:", device)

    model = YOLO(model_path).to(device)

    # 모델 워밍업 (첫 프레임 지연 방지)
    dummy = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    model.predict(dummy, classes=[0], imgsz=960, half=(device == "cuda"),
                  device=device, verbose=False, retina_masks=True)

    cap = cv2.VideoCapture(video_path)
    orig_fps = cap.get(cv2.CAP_PROP_FPS)

    cam_index = find_camera()
    if cam_index is None:
        print("❌ 카메라를 찾을 수 없음")
        return

    print(f"카메라 인덱스: {cam_index}")

    cam = cv2.VideoCapture(cam_index)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    # 웹캠 버퍼 최소화 → 지연 감소
    cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    player = MediaPlayer(video_path)

    cv2.namedWindow("Sync Overlay", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(
        "Sync Overlay",
        cv2.WND_PROP_FULLSCREEN,
        cv2.WINDOW_FULLSCREEN
    )

    # =========================
    # YOLO 워커 스레드 설정
    # 웹캠 프레임을 받아 추론 후 마스크를 반환
    # maxsize=1: 항상 최신 프레임만 처리
    # =========================
    input_queue  = queue.Queue(maxsize=1)   # 웹캠 프레임 → YOLO
    output_queue = queue.Queue(maxsize=1)   # 마스크 결과 → 메인 루프
    stop_event   = threading.Event()

    def yolo_worker():
        while not stop_event.is_set():
            try:
                frame_web = input_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            results = model.predict(
                frame_web,
                classes=[0],
                imgsz=960,
                half=(device == "cuda"),
                device=device,
                verbose=False,
                retina_masks=True
            )

            if results[0].masks is not None:
                masks = results[0].masks.data
                combined_mask = torch.any(masks, dim=0).byte()
                mask_np = combined_mask.cpu().numpy()
                mask_np = cv2.resize(mask_np, (WIDTH, HEIGHT),
                                     interpolation=cv2.INTER_NEAREST)
            else:
                mask_np = None

            # 결과 큐가 꽉 찼으면 버리고 최신 것만 유지
            if output_queue.full():
                try:
                    output_queue.get_nowait()
                except queue.Empty:
                    pass
            output_queue.put(mask_np)

    worker_thread = threading.Thread(target=yolo_worker, daemon=True)
    worker_thread.start()

    # 마지막으로 받은 마스크 (YOLO가 느려도 이전 마스크 재사용)
    last_mask = None

    # 비디오 프레임 순차 읽기용 타이머
    frame_interval = 1.0 / orig_fps if orig_fps > 0 else 1.0 / 30.0

    try:
        while True:
            # ── 오디오 동기 ──────────────────────────────
            audio_frame, val = player.get_frame()
            if val == "eof":
                break
            if audio_frame is None:
                continue

            # ── 비디오: seek 없이 순차 읽기 ──────────────
            # 오디오 타임스탬프 기반으로 필요한 만큼 skip
            audio_time = audio_frame[1]
            video_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

            # 비디오가 오디오보다 많이 앞서 있으면 대기 (거의 없음)
            # 비디오가 뒤처지면 따라잡을 때까지 읽기
            while video_time < audio_time - frame_interval:
                ret, frame_vid = cap.read()
                if not ret:
                    break
                video_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

            ret, frame_vid = cap.read()
            if not ret:
                break

            frame_vid = cv2.resize(frame_vid, (WIDTH, HEIGHT))

            # ── 웹캠 프레임 읽기 ─────────────────────────
            ret, frame_web = cam.read()
            if not ret:
                continue
            frame_web = cv2.flip(frame_web, 1)

            # ── YOLO 입력 큐에 최신 프레임 전달 ──────────
            # 큐가 차 있으면 오래된 것 버리고 최신 프레임 넣기
            if input_queue.full():
                try:
                    input_queue.get_nowait()
                except queue.Empty:
                    pass
            input_queue.put(frame_web.copy())

            # ── YOLO 결과 수신 (있으면 갱신, 없으면 이전 마스크 재사용) ──
            try:
                last_mask = output_queue.get_nowait()
            except queue.Empty:
                pass  # 이전 마스크 그대로 사용

            # ── 합성 ─────────────────────────────────────
            if last_mask is not None:
                mask_bool = last_mask > 0
                frame_vid[mask_bool] = frame_web[mask_bool]

            cv2.imshow("Sync Overlay", frame_vid)
            if cv2.waitKey(1) == 27:
                break

    except Exception as e:
        print("Error:", e)

    finally:
        stop_event.set()
        worker_thread.join(timeout=3)
        player.close_player()
        cap.release()
        cam.release()
        cv2.destroyAllWindows()


# =========================
# GUI
# =========================
from PyQt5.QtWidgets import (
    QApplication, QWidget, QHBoxLayout, QVBoxLayout,
    QFileSystemModel, QTreeView, QPushButton, QLabel, QComboBox
)
from PyQt5.QtCore import QDir
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QHeaderView

class FileLauncher(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MP4 처리 실행기")
        self.resize(1500, 700)

        if getattr(sys, "frozen", False):
            self.start_folder = os.path.dirname(sys.executable)
        else:
            self.start_folder = os.path.dirname(os.path.abspath(__file__))

        self.selected_file = None
        layout = QHBoxLayout(self)

        # --- 왼쪽 레이아웃 (파일 트리) ---
        left = QVBoxLayout()
        self.drive_combo = QComboBox()
        self.drive_combo.addItems(self.get_drives())
        self.drive_combo.currentIndexChanged.connect(self.on_drive_changed)
        left.addWidget(self.drive_combo)

        self.model = QFileSystemModel()
        self.model.setNameFilters(["*.mp4"])
        self.model.setNameFilterDisables(False)
        self.model.setFilter(QDir.AllDirs | QDir.Files | QDir.NoDotAndDotDot)
        self.model.setRootPath(QDir.rootPath())

        self.tree = QTreeView()
        self.tree.setModel(self.model)
        self.tree.clicked.connect(self.on_file_selected)

        header = self.tree.header()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        left.addWidget(self.tree)
        layout.addLayout(left, 2)

        # --- 오른쪽 레이아웃 (정보 및 버튼) ---
        right = QVBoxLayout()
        self.file_label = QLabel("선택된 파일 없음")
        self.file_label.setFont(QFont("Arial", 11))
        self.file_label.setWordWrap(True)
        right.addWidget(self.file_label)

        self.status_label = QLabel("")
        self.status_label.setFont(QFont("Arial", 14))
        right.addWidget(self.status_label)

        self.run_btn = QPushButton("선택한 파일 실행")
        self.run_btn.setFixedHeight(50)
        self.run_btn.clicked.connect(self.run_script)
        right.addWidget(self.run_btn)

        right.addStretch()
        layout.addLayout(right, 1)

        current_drive = os.path.splitdrive(self.start_folder)[0].upper() + "\\"
        drive_idx = self.drive_combo.findText(current_drive)
        if drive_idx >= 0:
            self.drive_combo.setCurrentIndex(drive_idx)

        idx = self.model.index(self.start_folder)
        self.tree.setCurrentIndex(idx)
        self.tree.scrollTo(idx)
        self.tree.expand(idx)

    def get_drives(self):
        return [f"{d}:\\" for d in string.ascii_uppercase if os.path.exists(f"{d}:\\")]

    def on_drive_changed(self, _):
        drive_path = self.drive_combo.currentText()
        idx = self.model.index(drive_path)
        self.tree.scrollTo(idx)
        self.tree.setCurrentIndex(idx)

    def on_file_selected(self, index):
        path = self.model.filePath(index)
        if path.lower().endswith(".mp4"):
            self.selected_file = path
            self.file_label.setText(f"선택됨:\n{path}")

    def run_script(self):
        if not self.selected_file:
            self.file_label.setText("⚠ MP4 파일을 선택하세요")
            return

        if getattr(sys, "frozen", False):
            env = os.environ.copy()
            if "CUDA_VISIBLE_DEVICES" in env:
                del env["CUDA_VISIBLE_DEVICES"]
            env["KMP_DUPLICATE_LIB_OK"] = "TRUE"

            subprocess.Popen(
                [sys.executable, "process", self.selected_file],
                env=env,
                shell=False
            )
            self.status_label.setText("프로세스 시작됨")
        else:
            self.status_label.setText("직접 실행 중...")
            process_video(self.selected_file)


# =========================
# Entry point
# =========================
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "process":
        process_video(sys.argv[2])
    else:
        app = QApplication(sys.argv)
        win = FileLauncher()
        win.show()
        sys.exit(app.exec_())