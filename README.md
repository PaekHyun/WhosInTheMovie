# WhosInTheMovie
Real-time system that composites a live webcam person into a pre-recorded video and plays the merged result instantly.

It extracts a person from a live webcam feed and composites them with a pre-recorded video stored on the PC, then outputs the result in real time.
The program is executed using the launcher.py file, and app.py is a single, merged file intended for building a .exe executable.
Since PyInstaller does not work properly with the latest versions of PyTorch, it is recommended to install version 2.8.0 when creating an executable file.

실시간으로 웹캠 영상에서 사람을 분리하고, 
저장해둔 PC의 영상으로 합쳐서 실시간으로 출력해줍니다.

launcher.py 파일을 이용하여 실행하고
app.py 파일은 .exe 파일로 만들기 위해 하나의 파일로 합친 파일입니다.

pytorch 최신 버전의 경우, pyinstaller가 제대로 동작하지 않기 때문에
exe파일로 만들 경우에는 2.8.0으로 설치하길 추천드립니다.
