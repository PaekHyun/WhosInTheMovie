"""
WhosInTheMovie - 실시간 웹캠 인물 합성 시스템
================================================
웹캠 영상에서 사람을 실시간으로 분리하여, 미리 녹화된 영상 위에 합성합니다.
오디오 동기화, YOLO 세그멘테이션, 멀티스레드 처리를 지원합니다.

사용법:
    python whosinthemovie.py                  # GUI 모드 실행
    python whosinthemovie.py process <mp4경로> # CLI 모드 실행

PyInstaller 빌드:
    pyinstaller whosinthemovie.py ^
        --noconsole ^
        --collect-all torch ^
        --collect-all ultralytics ^
        --add-data "yolov8s-seg.pt;."
"""

import os
import sys
import string
import subprocess
import threading
import queue
import time
from dataclasses import dataclass, field
from typing import Optional

# ============================================================================
# 설정 (Configuration)
# ============================================================================

@dataclass
class AppConfig:
    """애플리케이션 전역 설정"""
    # 화면 해상도
    width: int = 1920
    height: int = 1080

    # YOLO 모델 설정
    model_filename: str = "yolov8s-seg.pt"
    yolo_classes: list = field(default_factory=lambda: [0])  # 0 = person
    yolo_imgsz: int = 640

    # 웹캠 설정
    max_camera_index: int = 5
    camera_buffer_size: int = 1

    # 스레드 큐 타임아웃 (초)
    queue_timeout: float = 0.1

    # 웹캠 초기 대기 시간 (초)
    cam_init_timeout: float = 3.0

    # 윈도우 타이틀
    window_title: str = "Sync Overlay"

    # GUI 윈도우 크기
    gui_width: int = 1500
    gui_height: int = 700


# ============================================================================
# 유틸리티 함수
# ============================================================================

def get_resource_path(filename: str) -> str:
    """
    PyInstaller 빌드 환경과 일반 Python 환경 모두에서
    리소스 파일 경로를 올바르게 반환합니다.
    """
    if getattr(sys, "frozen", False):
        return os.path.join(os.path.dirname(sys.executable), filename)
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)


def find_camera(max_index: int = 5) -> Optional[int]:
    """
    사용 가능한 웹캠 인덱스를 자동 탐색합니다.
    우선순위: 1, 0, 2, 3, 4 (외장 웹캠 우선)
    """
    import cv2

    for idx in [1, 0, 2, 3, 4]:
        if idx > max_index:
            continue
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            cap.release()
            return idx
        cap.release()
    return None


def setup_torch_dll_path() -> None:
    """
    PyInstaller 환경에서 Torch DLL 경로를 PATH에 추가합니다.
    Windows 환경에서 DLL 로딩 실패를 방지합니다.
    """
    if not getattr(sys, "frozen", False):
        return

    curr_dir = os.path.dirname(sys.executable)
    torch_lib_path = os.path.join(curr_dir, "_internal", "torch", "lib")
    if os.path.exists(torch_lib_path):
        os.add_dll_directory(torch_lib_path)
        os.environ["PATH"] = torch_lib_path + os.pathsep + os.environ.get("PATH", "")


# ============================================================================
# 웹캠 캡처 스레드
# ============================================================================

class WebcamCaptureThread:
    """
    웹캠 프레임을 별도 스레드에서 지속적으로 캡처합니다.
    항상 최신 프레임만 유지하여 버퍼 누적 딜레이를 방지합니다.
    """

    def __init__(self, cam, width: int, height: int):
        self._cam = cam
        self._width = width
        self._height = height
        self._latest_frame: Optional[object] = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()

    @property
    def latest_frame(self) -> Optional[object]:
        """가장 최근 캡처된 웹캠 프레임을 반환합니다."""
        with self._lock:
            return self._latest_frame

    @property
    def stop_event(self) -> threading.Event:
        return self._stop_event

    def start(self) -> threading.Thread:
        """웹캠 캡처 스레드를 시작합니다."""
        thread = threading.Thread(target=self._run, daemon=True)
        thread.start()
        return thread

    def stop(self) -> None:
        """캡처 스레드를 중지합니다."""
        self._stop_event.set()

    def _run(self) -> None:
        import cv2

        while not self._stop_event.is_set():
            ret, frame = self._cam.read()
            if not ret:
                continue
            # 좌우 반전 (거울 모드)
            flipped = cv2.flip(frame, 1)
            with self._lock:
                self._latest_frame = flipped


# ============================================================================
# YOLO 추론 스레드
# ============================================================================

class YOLOInferenceThread:
    """
    YOLO 세그멘테이션을 별도 스레드에서 수행합니다.
    입력 큐와 출력 큐를 통해 메인 루프와 통신하며,
    항상 최신 프레임/결과만 유지합니다.
    """

    def __init__(self, model, device: str, config: AppConfig):
        self._model = model
        self._device = device
        self._config = config
        self._input_queue: queue.Queue = queue.Queue(maxsize=1)
        self._output_queue: queue.Queue = queue.Queue(maxsize=1)
        self._stop_event = threading.Event()

    @property
    def stop_event(self) -> threading.Event:
        return self._stop_event

    def enqueue_frame(self, frame) -> None:
        """
        웹캠 프레임을 입력 큐에 넣습니다.
        큐가 꽉 찬 경우 기존 프레임을 버리고 최신 것만 유지합니다.
        """
        if self._input_queue.full():
            try:
                self._input_queue.get_nowait()
            except queue.Empty:
                pass
        self._input_queue.put(frame)

    def get_mask(self, timeout: float = 0.0) -> Optional[object]:
        """
        가장 최근 세그멘테이션 마스크를 반환합니다.
        없으면 None을 반환합니다.
        """
        try:
            return self._output_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def start(self) -> threading.Thread:
        """YOLO 추론 스레드를 시작합니다."""
        thread = threading.Thread(target=self._run, daemon=True)
        thread.start()
        return thread

    def stop(self) -> None:
        """추론 스레드를 중지합니다."""
        self._stop_event.set()

    def _run(self) -> None:
        import cv2
        import torch
        import numpy as np

        while not self._stop_event.is_set():
            try:
                frame = self._input_queue.get(timeout=self._config.queue_timeout)
            except queue.Empty:
                continue

            results = self._model.predict(
                frame,
                classes=self._config.yolo_classes,
                imgsz=self._config.yolo_imgsz,
                half=(self._device == "cuda"),
                device=self._device,
                verbose=False,
                retina_masks=True,
            )

            mask_np = None
            if results[0].masks is not None:
                masks = results[0].masks.data
                combined_mask = torch.any(masks, dim=0).byte()
                mask_np = combined_mask.cpu().numpy()
                mask_np = cv2.resize(
                    mask_np,
                    (self._config.width, self._config.height),
                    interpolation=cv2.INTER_NEAREST,
                )

            # 출력 큐가 꽉 찬 경우 기존 결과를 버리고 최신 것만 유지
            if self._output_queue.full():
                try:
                    self._output_queue.get_nowait()
                except queue.Empty:
                    pass
            self._output_queue.put(mask_np)


# ============================================================================
# 영상 처리 (메인 로직)
# ============================================================================

def process_video(video_path: str, config: Optional[AppConfig] = None) -> None:
    """
    웹캠 인물을 영상에 실시간 합성하여 출력합니다.

    Args:
        video_path: 합성할 배경 영상(MP4) 경로
        config: 애플리케이션 설정 (기본값 사용 시 생략 가능)
    """
    import cv2
    import torch
    import numpy as np
    from ultralytics import YOLO
    from ffpyplayer.player import MediaPlayer

    if config is None:
        config = AppConfig()

    # --- PyInstaller 환경 DLL 경로 설정 ---
    setup_torch_dll_path()

    # --- 디바이스 및 모델 로드 ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] YOLO device: {device}")

    model_path = get_resource_path(config.model_filename)
    model = YOLO(model_path).to(device)

    # 모델 워밍업 (첫 프레임 지연 방지)
    dummy = np.zeros((config.height, config.width, 3), dtype=np.uint8)
    model.predict(
        dummy,
        classes=config.yolo_classes,
        imgsz=config.yolo_imgsz,
        half=(device == "cuda"),
        device=device,
        verbose=False,
        retina_masks=True,
    )
    print("[INFO] 모델 워밍업 완료")

    # --- 비디오 캡처 ---
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("[ERROR] 영상 파일을 열 수 없습니다:", video_path)
        return

    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = 1.0 / orig_fps if orig_fps > 0 else 1.0 / 30.0
    print(f"[INFO] 영상: {orig_fps:.1f} FPS, 총 {total_frames} 프레임")

    # --- 웹캠 캡처 ---
    cam_index = find_camera(config.max_camera_index)
    if cam_index is None:
        print("[ERROR] 사용 가능한 웹캠을 찾을 수 없습니다")
        cap.release()
        return
    print(f"[INFO] 웹캠 인덱스: {cam_index}")

    cam = cv2.VideoCapture(cam_index)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, config.width)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, config.height)
    cam.set(cv2.CAP_PROP_BUFFERSIZE, config.camera_buffer_size)

    # --- 오디오 플레이어 ---
    player = MediaPlayer(video_path)

    # --- 전체화면 윈도우 ---
    cv2.namedWindow(config.window_title, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(
        config.window_title,
        cv2.WND_PROP_FULLSCREEN,
        cv2.WINDOW_FULLSCREEN,
    )

    # --- 백그라운드 스레드 시작 ---
    cam_thread = WebcamCaptureThread(cam, config.width, config.height)
    cam_worker = cam_thread.start()

    yolo_thread = YOLOInferenceThread(model, device, config)
    yolo_worker = yolo_thread.start()

    # 웹캠 첫 프레임 대기
    start_wait = time.time()
    while cam_thread.latest_frame is None and time.time() - start_wait < config.cam_init_timeout:
        time.sleep(0.05)
    if cam_thread.latest_frame is None:
        print("[ERROR] 웹캠 프레임을 가져올 수 없습니다")
        cam_thread.stop()
        yolo_thread.stop()
        cap.release()
        cam.release()
        player.close_player()
        cv2.destroyAllWindows()
        return

    # --- 메인 루프 ---
    last_mask = None
    last_video_time = 0.0

    try:
        while True:
            # 오디오 프레임 읽기 (오디오가 마스터 클럭)
            audio_frame, val = player.get_frame()
            if val == "eof":
                print("[INFO] 영상 재생 완료")
                break
            if audio_frame is None:
                continue

            audio_time = audio_frame[1]

            # 오디오 시간에 해당하는 비디오 프레임으로 점프
            target_frame = int(audio_time * orig_fps)
            if target_frame >= total_frames:
                continue

            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            ret, frame_vid = cap.read()
            if not ret:
                continue
            frame_vid = cv2.resize(frame_vid, (config.width, config.height))

            # 웹캠 최신 프레임 가져오기
            frame_web = cam_thread.latest_frame
            if frame_web is None:
                cv2.imshow(config.window_title, frame_vid)
                if cv2.waitKey(1) == 27:
                    break
                continue

            # YOLO에 웹캠 프레임 전달
            yolo_thread.enqueue_frame(frame_web)

            # 마스크 결과 가져오기 (없으면 이전 마스크 재사용)
            new_mask = yolo_thread.get_mask()
            if new_mask is not None:
                last_mask = new_mask

            # 마스크 합성
            if last_mask is not None:
                mask_bool = last_mask > 0
                frame_vid[mask_bool] = frame_web[mask_bool]

            # 출력
            cv2.imshow(config.window_title, frame_vid)
            if cv2.waitKey(1) == 27:  # ESC 종료
                break

    except Exception as e:
        print(f"[ERROR] {e}")
    finally:
        # --- 정리 ---
        cam_thread.stop()
        yolo_thread.stop()
        cam_worker.join(timeout=3)
        yolo_worker.join(timeout=3)
        player.close_player()
        cap.release()
        cam.release()
        cv2.destroyAllWindows()
        print("[INFO] 프로세스 종료")


# ============================================================================
# GUI (PyQt5 파일 선택 실행기)
# ============================================================================

def run_gui(config: Optional[AppConfig] = None) -> None:
    """
    PyQt5 기반 파일 선택 GUI를 실행합니다.
    MP4 파일을 선택하면 영상 처리 프로세스를 시작합니다.
    """
    from PyQt5.QtWidgets import (
        QApplication, QWidget, QHBoxLayout, QVBoxLayout,
        QFileSystemModel, QTreeView, QPushButton, QLabel, QComboBox,
        QHeaderView,
    )
    from PyQt5.QtCore import QDir
    from PyQt5.QtGui import QFont

    if config is None:
        config = AppConfig()

    class FileLauncher(QWidget):
        """MP4 파일 선택 및 실행 GUI 위젯"""

        def __init__(self):
            super().__init__()
            self._config = config
            self._selected_file: Optional[str] = None
            self._init_ui()

        def _init_ui(self) -> None:
            self.setWindowTitle("WhosInTheMovie - MP4 처리 실행기")
            self.resize(config.gui_width, config.gui_height)

            # 시작 폴더 결정
            if getattr(sys, "frozen", False):
                self._start_folder = os.path.dirname(sys.executable)
            else:
                self._start_folder = os.path.dirname(os.path.abspath(__file__))

            layout = QHBoxLayout(self)

            # --- 왼쪽: 파일 트리 ---
            left_layout = QVBoxLayout()

            self._drive_combo = QComboBox()
            self._drive_combo.addItems(self._get_drives())
            self._drive_combo.currentIndexChanged.connect(self._on_drive_changed)
            left_layout.addWidget(self._drive_combo)

            self._model = QFileSystemModel()
            self._model.setNameFilters(["*.mp4"])
            self._model.setNameFilterDisables(False)
            self._model.setFilter(QDir.AllDirs | QDir.Files | QDir.NoDotAndDotDot)
            self._model.setRootPath(QDir.rootPath())

            self._tree = QTreeView()
            self._tree.setModel(self._model)
            self._tree.clicked.connect(self._on_file_selected)

            header = self._tree.header()
            header.setSectionResizeMode(0, QHeaderView.Stretch)

            left_layout.addWidget(self._tree)
            layout.addLayout(left_layout, 2)

            # --- 오른쪽: 정보 및 실행 버튼 ---
            right_layout = QVBoxLayout()

            self._file_label = QLabel("선택된 파일 없음")
            self._file_label.setFont(QFont("Arial", 11))
            self._file_label.setWordWrap(True)
            right_layout.addWidget(self._file_label)

            self._status_label = QLabel("")
            self._status_label.setFont(QFont("Arial", 14))
            right_layout.addWidget(self._status_label)

            self._run_btn = QPushButton("🎬 선택한 파일 실행")
            self._run_btn.setFixedHeight(50)
            self._run_btn.clicked.connect(self._run_script)
            right_layout.addWidget(self._run_btn)

            right_layout.addStretch()
            layout.addLayout(right_layout, 1)

            # 초기 드라이브 및 경로 설정
            current_drive = os.path.splitdrive(self._start_folder)[0].upper() + "\\"
            drive_idx = self._drive_combo.findText(current_drive)
            if drive_idx >= 0:
                self._drive_combo.setCurrentIndex(drive_idx)

            idx = self._model.index(self._start_folder)
            self._tree.setCurrentIndex(idx)
            self._tree.scrollTo(idx)
            self._tree.expand(idx)

        def _get_drives(self) -> list:
            """사용 가능한 윈도우 드라이브 목록을 반환합니다."""
            return [
                f"{d}:\\" for d in string.ascii_uppercase
                if os.path.exists(f"{d}:\\")
            ]

        def _on_drive_changed(self, _) -> None:
            """드라이브 콤보박스 변경 시 파일 트리 루트를 업데이트합니다."""
            drive_path = self._drive_combo.currentText()
            if drive_path:
                idx = self._model.index(drive_path)
                self._tree.scrollTo(idx)
                self._tree.setCurrentIndex(idx)

        def _on_file_selected(self, index) -> None:
            """파일 트리에서 파일 선택 시 호출됩니다."""
            path = self._model.filePath(index)
            if path.lower().endswith(".mp4"):
                self._selected_file = path
                self._file_label.setText(f"선택됨:\n{path}")

        def _run_script(self) -> None:
            """선택된 MP4 파일로 영상 처리를 시작합니다."""
            if not self._selected_file:
                self._file_label.setText("⚠ MP4 파일을 선택하세요")
                return

            if getattr(sys, "frozen", False):
                # PyInstaller 빌드: CUDA 사용을 위해 별도 프로세스로 실행
                env = os.environ.copy()
                env.pop("CUDA_VISIBLE_DEVICES", None)
                env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
                subprocess.Popen(
                    [sys.executable, "process", self._selected_file],
                    env=env,
                    shell=False,
                )
                self._status_label.setText("✅ 프로세스 시작됨")
            else:
                # 개발 환경: 직접 실행
                self._status_label.setText("⏳ 실행 중...")
                process_video(self._selected_file, self._config)

    # --- GUI 실행 ---
    app = QApplication(sys.argv)
    window = FileLauncher()
    window.show()
    sys.exit(app.exec_())


# ============================================================================
# 엔트리 포인트
# ============================================================================

def main() -> None:
    """
    프로그램 진입점입니다.

    - 인자 없이 실행: GUI 모드
    - `process <mp4경로>` 인자: CLI 영상 처리 모드
    """
    # GUI 프로세스에서는 CUDA 차단 (PyInstaller 빌드 시)
    if __name__ == "__main__" and len(sys.argv) == 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    if len(sys.argv) > 1 and sys.argv[1] == "process":
        # CLI 모드: 영상 처리
        if len(sys.argv) < 3:
            print("사용법: python whosinthemovie.py process <mp4경로>")
            sys.exit(1)
        process_video(sys.argv[2])
    else:
        # GUI 모드
        run_gui()


if __name__ == "__main__":
    main()
