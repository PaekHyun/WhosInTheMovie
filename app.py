import sys
import os
import string
import subprocess

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
# 영상 처리 (CUDA 전용 프로세스)
# =========================
def process_video(video_path):
    # ❗ torch / YOLO는 반드시 여기서 import
    import os
    import sys
    
    # PyInstaller 환경일 때 DLL 경로를 PATH에 추가
    if getattr(sys, 'frozen', False):
        curr_dir = os.path.dirname(sys.executable)
        # _internal 폴더 내의 torch/lib 경로 확보
        torch_lib_path = os.path.join(curr_dir, "_internal", "torch", "lib")
        if os.path.exists(torch_lib_path):
            os.add_dll_directory(torch_lib_path)
            os.environ["PATH"] = torch_lib_path + os.pathsep + os.environ["PATH"]    
    
    
    
    import cv2
    import torch
    from ultralytics import YOLO
    from ffpyplayer.player import MediaPlayer

    WIDTH, HEIGHT = 1920, 1080

    model_path = get_resource_path("yolov8n-seg.pt")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("YOLO device:", device)

    model = YOLO(model_path).to(device)

    cap = cv2.VideoCapture(video_path)
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cam = cv2.VideoCapture(1)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

    player = MediaPlayer(video_path)

    cv2.namedWindow("Sync Overlay", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(
        "Sync Overlay",
        cv2.WND_PROP_FULLSCREEN,
        cv2.WINDOW_FULLSCREEN
    )

    try:
        while True:
            audio_frame, val = player.get_frame()
            if val == "eof":
                break
            if audio_frame is None:
                continue

            audio_time = audio_frame[1]
            target_frame = int(audio_time * orig_fps)

            if target_frame >= total_frames:
                continue

            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            ret, frame_vid = cap.read()
            if not ret:
                continue

            frame_vid = cv2.resize(frame_vid, (WIDTH, HEIGHT))

            ret, frame_web = cam.read()
            if not ret:
                continue

            results = model.predict(
                frame_web,
                classes=[0],
                imgsz=640,
                half=(device == "cuda"),
                device=device,
                verbose=False
            )

            if results[0].masks is not None:
                masks = results[0].masks.data
                combined_mask = torch.any(masks, dim=0).float()
                combined_mask = cv2.resize(
                    combined_mask.cpu().numpy(),
                    (WIDTH, HEIGHT)
                )
                mask_bool = combined_mask > 0
                frame_vid[mask_bool] = frame_web[mask_bool]

            cv2.imshow("Sync Overlay", frame_vid)

            if cv2.waitKey(1) == 27:
                break

    except Exception as e:
        print("Error:", e)

    finally:
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
        self.resize(1500, 600)

        self.start_folder = r"C:\\"
        self.selected_file = None

        layout = QHBoxLayout(self)

        # 왼쪽
        left = QVBoxLayout()

        self.drive_combo = QComboBox()
        self.drive_combo.addItems(self.get_drives())
        self.drive_combo.currentIndexChanged.connect(self.on_drive_changed)
        left.addWidget(self.drive_combo)

        self.model = QFileSystemModel()
        self.model.setNameFilters(["*.mp4"])
        self.model.setNameFilterDisables(False)
        self.model.setFilter(QDir.AllDirs | QDir.Files)

        self.tree = QTreeView()
        self.tree.setModel(self.model)
        self.tree.clicked.connect(self.on_file_selected)
        
        
        header = self.tree.header()
        header.setSectionResizeMode(0, QHeaderView.Stretch)      # 파일명 컬럼
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)

        self.tree.setColumnWidth(0, 800)  # 초기 폭 (선택)
        
        
        left.addWidget(self.tree)

        layout.addLayout(left, 2)

        # 오른쪽
        right = QVBoxLayout()

        self.file_label = QLabel("선택된 파일 없음")
        font = QFont()
        font.setPointSize(12)
        self.file_label.setFont(font)
        right.addWidget(self.file_label)

        self.status_label = QLabel("")
        status_font = QFont()
        status_font.setPointSize(14)
        self.status_label.setFont(status_font)
        right.addWidget(self.status_label)

        self.run_btn = QPushButton("선택한 파일 실행")
        self.run_btn.clicked.connect(self.run_script)
        right.addWidget(self.run_btn)

        right.addStretch()
        layout.addLayout(right, 1)

        self.tree.setRootIndex(
            self.model.setRootPath(self.start_folder)
        )

    def get_drives(self):
        return [
            f"{d}:\\"
            for d in string.ascii_uppercase
            if os.path.exists(f"{d}:\\")
        ]

    def on_drive_changed(self, _):
        drive = self.drive_combo.currentText()
        self.tree.setRootIndex(self.model.setRootPath(drive))

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
            # 🔥 자식 프로세스에서는 CUDA 허용
            env = os.environ.copy()
            if "CUDA_VISIBLE_DEVICES" in env:
                del env["CUDA_VISIBLE_DEVICES"] # pop보다 확실하게 제거
            env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
            
            subprocess.Popen(
                [sys.executable, "process", self.selected_file],
                env=env,
                shell=False
            )
            self.status_label.setText("실행됨")
        else:
            # 개발 중 (VS Code)
            self.status_label.setText("처리 중...")
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
