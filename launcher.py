import sys
import subprocess
from PyQt5.QtWidgets import (
    QApplication, QWidget, QHBoxLayout, QVBoxLayout,
    QFileSystemModel, QTreeView, QPushButton, QLabel, QComboBox
)
from PyQt5.QtCore import QDir
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QHeaderView

import string
import os

class FileLauncher(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MP4 처리 실행기")
        self.resize(1500, 600)
        
        self.start_folder = r"C:\\"  # <- 탐색기 시작 폴더 지정
        

        layout = QHBoxLayout()
        self.setLayout(layout)

        # --- 왼쪽 패널: 드라이브 선택 + 폴더 트리 ---
        left_panel = QVBoxLayout()

        # 드라이브 콤보박스
        self.drive_combo = QComboBox()
        self.drive_combo.addItems(self.get_drives())
        self.drive_combo.currentIndexChanged.connect(self.on_drive_changed)
        left_panel.addWidget(self.drive_combo)

        # 폴더/파일 트리
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
        left_panel.addWidget(self.tree)
        layout.addLayout(left_panel, 2)

        # --- 오른쪽 패널: 실행 버튼 ---
        right_panel = QVBoxLayout()
        
        
        # 선택 파일 레이블 (항상 파일명 표시)
        self.file_label = QLabel("선택된 파일 없음")
        file_font = QFont()
        file_font.setPointSize(12)  # 글씨 크기 조정
        self.file_label.setFont(file_font)
        right_panel.addWidget(self.file_label)

        # 실행 상태 레이블 (실행됨 / 실행 끝)
        self.status_label = QLabel("")
        status_font = QFont()
        status_font.setPointSize(14)  # 더 큰 글씨
        self.status_label.setFont(status_font)
        right_panel.addWidget(self.status_label)
        
        

        self.run_btn = QPushButton("선택한 파일 실행")
        self.run_btn.clicked.connect(self.run_script)
        right_panel.addWidget(self.run_btn)

        right_panel.addStretch()
        layout.addLayout(right_panel, 1)

        self.selected_file = None

        # 초기 드라이브 설정
        if os.path.exists(self.start_folder):
            drive_letter = self.start_folder[:3]  # "D:\\"
            index = self.drive_combo.findText(drive_letter)
            if index >= 0:
                self.drive_combo.setCurrentIndex(index)
            self.tree.setRootIndex(self.model.setRootPath(self.start_folder))        
        # self.on_drive_changed(0)

    def get_drives(self):
        """윈도우 드라이브 목록 가져오기"""
        drives = []
        for d in string.ascii_uppercase:
            if os.path.exists(f"{d}:\\"):
                drives.append(f"{d}:\\")
        return drives

    def on_drive_changed(self, index):
        drive_path = self.drive_combo.currentText()
        if drive_path:
            self.tree.setRootIndex(self.model.setRootPath(drive_path))

    def on_file_selected(self, index):
        path = self.model.filePath(index)
        if path.lower().endswith(".mp4"):
            self.selected_file = path
            self.file_label.setText(f"선택됨:\n{path}")

    def run_script(self):
        if not self.selected_file:
            self.file_label.setText("⚠ MP4 파일을 선택하세요")
            return
                
        subprocess.Popen(
            [sys.executable, "process_video.py", self.selected_file],
            shell=False
        )

            # 실행 상태 레이블 "실행됨"
        self.status_label.setText("실행됨")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FileLauncher()
    window.show()
    sys.exit(app.exec_())
