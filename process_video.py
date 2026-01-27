import cv2
import torch
import sys
from ultralytics import YOLO
from ffpyplayer.player import MediaPlayer

# --- 인자로 MP4 경로 받기 ---
if len(sys.argv) < 2:
    print("사용법: python process_video.py <video_path.mp4>")
    sys.exit(1)

VIDEO_PATH = sys.argv[1]  # GUI에서 전달된 mp4 경로
WIDTH, HEIGHT = 1920, 1080

# --- YOLO 모델 ---
model = YOLO('yolov8n-seg.pt').to('cuda')

# --- 비디오 캡처 ---
cap = cv2.VideoCapture(VIDEO_PATH)
orig_fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# --- 웹캠 캡처 ---
cam = cv2.VideoCapture(1)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

# --- 오디오 플레이어 ---
player = MediaPlayer(VIDEO_PATH)

# --- 윈도우 설정 ---
cv2.namedWindow('Sync Overlay', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty('Sync Overlay', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

try:
    while True:
        # --- 오디오 프레임 읽기 (오디오가 마스터 클럭) ---
        audio_frame, val = player.get_frame()

        if val == 'eof':
            break            
            

        if audio_frame is None:
            continue

        audio_time = audio_frame[1]  # 현재 오디오 재생 시간 (초)

        # --- 오디오 시간에 해당하는 비디오 프레임 번호 계산 ---
        target_frame = int(audio_time * orig_fps)

        if target_frame >= total_frames:
            continue

        # --- 비디오 해당 위치로 점프 ---
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ret, frame_vid = cap.read()
        if not ret:
            continue

        frame_vid = cv2.resize(frame_vid, (WIDTH, HEIGHT))

        # --- 웹캠 프레임 읽기 ---
        ret, frame_web = cam.read()
        if not ret:
            continue

        # --- YOLO 세그멘테이션 ---
        results = model.predict(frame_web, classes=[0],
                                imgsz=640, half=True,
                                device='cuda', verbose=False)

        if results[0].masks is not None:
            masks = results[0].masks.data
            combined_mask = torch.any(masks, dim=0).float()
            combined_mask = cv2.resize(combined_mask.cpu().numpy(), (WIDTH, HEIGHT))

            mask_bool = combined_mask > 0
            frame_vid[mask_bool] = frame_web[mask_bool]

        # --- 출력 ---
        cv2.imshow('Sync Overlay', frame_vid)

        if cv2.waitKey(1) == 27:  # ESC
            break

except Exception as e:
    print("Error:", e)

finally:
    player.close_player()
    cap.release()
    cam.release()
    cv2.destroyAllWindows()
