# 🎬 WhosInTheMovie

> 실시간으로 웹캠 영상에서 사람을 분리하여, 미리 녹화된 영상 위에 합성하는 시스템

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.8.0-orange)
![YOLO](https://img.shields.io/badge/YOLO-v8seg-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 📌 프로젝트 소개

**WhosInTheMovie**는 웹캠에 비친 사람을 실시간으로 세그멘테이션하여, PC에 저장된 영상 위에 합성·출력하는 실시간 합성 시스템입니다.

### 주요 기능

| 기능 | 설명 |
|------|------|
| 🎥 실시간 인물 분리 | YOLOv8-seg 모델로 웹캠 영상에서 사람만 실시간 분리 |
| 🎞️ 영상 합성 | 분리된 인물을 배경 영상 위에 오버레이 |
| 🔊 오디오 동기화 | ffpyplayer 기반 오디오 마스터 클럭으로 영상-음성 싱크 유지 |
| 🖥️ 전체화면 출력 | 합성 결과를 전체화면으로 실시간 출력 |
| 📂 GUI 파일 선택 | PyQt5 기반 파일 탐색기로 MP4 파일 선택 |
| 🚀 멀티스레드 처리 | 웹캠 캡처·YOLO 추론·렌더링을 각각 독립 스레드로 분리 |

---

## 🏗️ 아키텍처

```
┌─────────────────────────────────────────────────────┐
│                    메인 루프                         │
│  오디오 클럭 → 비디오 프레임 점프 → 합성 → 출력      │
└──────────┬──────────────────┬────────────────────────┘
           │                  │
    ┌──────▼──────┐   ┌──────▼──────┐
    │ WebcamThread│   │  YOLOThread  │
    │ 최신 프레임  │   │ 세그멘테이션 │
    │ 유지 (lock) │   │ 큐 기반 통신  │
    └─────────────┘   └─────────────┘
```

### 스레드 구조

| 스레드 | 역할 | 통신 방식 |
|--------|------|-----------|
| **WebcamCaptureThread** | 웹캠 프레임 지속 캡처, 최신 프레임만 유지 | Lock 기반 공유 변수 |
| **YOLOInferenceThread** | YOLO 세그멘테이션 추론 | 입력/출력 Queue (maxsize=1) |
| **메인 스레드** | 오디오 동기화, 프레임 합성, 화면 출력 | 직접 제어 |

---

## 📁 프로젝트 구조

```
WhosInTheMovie/
├── whosinthemovie.py    # 메인 실행 파일 (GUI + 영상처리 통합)
├── yolov8s-seg.pt       # YOLOv8 세그멘테이션 모델 (s-size)
├── yolov8n-seg.pt       # YOLOv8 세그멘테이션 모델 (n-size, 경량)
└── README.md
```

> **참고**: 기존 `launcher.py` + `process_video.py` + `app.py` 세 파일은
> `whosinthemovie.py` 하나로 통합·리팩토링되었습니다.

---

## ⚙️ 설치

### 1. 필수 패키지 설치

```bash
pip install torch==2.8.0 torchvision==0.19.0
pip install ultralytics
pip install opencv-python
pip install ffpyplayer
pip install PyQt5
```

### 2. YOLO 모델 다운로드

프로젝트 루트에 아래 모델 파일을 배치합니다:

| 모델 | 크기 | 용도 |
|------|------|------|
| `yolov8s-seg.pt` | ~22MB | 기본 (정밀도 우선) |
| `yolov8n-seg.pt` | ~6MB | 경량 (속도 우선) |

모델 파일은 [Ultralytics 공식 저장소](https://github.com/ultralytics/assets/releases)에서 다운로드할 수 있습니다.

---

## 🚀 사용법

### GUI 모드 (권장)

```bash
python whosinthemovie.py
```

1. 파일 탐색기에서 MP4 파일 선택
2. **"🎬 선택한 파일 실행"** 버튼 클릭
3. 전체화면으로 합성 영상 출력
4. **ESC** 키로 종료

### CLI 모드

```bash
python whosinthemovie.py process "C:\Videos\background.mp4"
```

---

## 🔧 설정

`AppConfig` 데이터클래스를 통해 설정을 변경할 수 있습니다:

```python
from whosinthemovie import AppConfig, process_video

config = AppConfig(
    width=1920,           # 출력 해상도 너비
    height=1080,          # 출력 해상도 높이
    model_filename="yolov8s-seg.pt",  # YOLO 모델 파일명
    yolo_imgsz=640,       # YOLO 입력 이미지 크기
    camera_buffer_size=1, # 웹캠 버퍼 크기
)

process_video("video.mp4", config=config)
```

| 설정 항목 | 기본값 | 설명 |
|-----------|--------|------|
| `width` | 1920 | 출력 해상도 너비 |
| `height` | 1080 | 출력 해상도 높이 |
| `model_filename` | `yolov8s-seg.pt` | YOLO 모델 파일명 |
| `yolo_classes` | `[0]` | 탐지 클래스 (0=person) |
| `yolo_imgsz` | 640 | YOLO 입력 이미지 사이즈 |
| `max_camera_index` | 5 | 웹캠 탐색 최대 인덱스 |
| `camera_buffer_size` | 1 | 웹캠 버퍼 크기 (1=최신만) |
| `queue_timeout` | 0.1 | 스레드 큐 타임아웃 (초) |
| `cam_init_timeout` | 3.0 | 웹캠 초기 대기 시간 (초) |

---

## 📦 PyInstaller 빌드

```bash
pyinstaller whosinthemovie.py ^
    --noconsole ^
    --collect-all torch ^
    --collect-all ultralytics ^
    --add-data "yolov8s-seg.pt;."
```

> ⚠️ **PyTorch 버전 주의**: PyInstaller는 최신 PyTorch와 호환되지 않을 수 있습니다.
> 빌드 시 PyTorch **2.8.0** 사용을 권장합니다.

---

## 🔄 리팩토링 변경 사항

기존 3개 파일(`launcher.py`, `process_video.py`, `app.py`)을 단일 파일로 통합하면서 다음과 같이 개선했습니다:

### 구조 개선
| 항목 | 변경 전 | 변경 후 |
|------|---------|---------|
| 파일 구성 | 3개 파일 분산 | `whosinthemovie.py` 단일 파일 |
| 설정 관리 | 하드코딩된 매직 넘버 | `AppConfig` 데이터클래스로 중앙 관리 |
| 웹캠 캡처 | 메인 루프에서 직접 `cam.read()` | `WebcamCaptureThread` 클래스로 분리 |
| YOLO 추론 | 인라인 스레드 코드 | `YOLOInferenceThread` 클래스로 분리 |

### 코드 품질 개선
- **타입 힌트**: 모든 함수에 타입 어노테이션 추가
- **에러 처리**: 웹캠 미감지, 영상 파일 미열림 등 예외 상황 처리 강화
- **리소스 정리**: `finally` 블록에서 스레드 조인·리소스 해제 보장
- **로깅**: `print` 메시지에 `[INFO]`, `[ERROR]` 접두사로 가독성 향상
- **함수 분리**: `setup_torch_dll_path()`, `find_camera()` 등 유틸리티 함수 독립화
- **Docstring**: 모든 클래스·함수에 문서화 문자열 추가

---

## 🛠️ 기술 스택

| 기술 | 용도 |
|------|------|
| **Python 3.8+** | 메인 언어 |
| **PyTorch** | 딥러닝 프레임워크 |
| **Ultralytics YOLOv8** | 실시간 인물 세그멘테이션 |
| **OpenCV** | 영상 처리 및 웹캠 캡처 |
| **ffpyplayer** | 오디오 재생 및 동기화 |
| **PyQt5** | GUI 파일 선택 인터페이스 |

---

## 📋 시스템 요구사항

| 항목 | 최소 사양 | 권장 사양 |
|------|-----------|-----------|
| OS | Windows 10 | Windows 10/11 |
| GPU | NVIDIA GTX 1060 | NVIDIA RTX 3060+ |
| RAM | 8GB | 16GB+ |
| 웹캠 | USB 웹캠 | 1080p 지원 웹캠 |
| CUDA | 11.8+ | 12.0+ |

---

## 📄 라이선스

MIT License

---

## 👤 작성자

**PaekHyun** - [GitHub](https://github.com/PaekHyun)
