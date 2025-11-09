# unitree_sdk2 — Human-Follow (YOLO-Pose) + Whisper Voice Control

This README describes a reproducible setup for a Human-Follow control application using YOLO-Pose (ONNX) and whisper.cpp for speech recognition on Ubuntu 22.04. Commands are copy/paste-ready.

## Table of contents
- Requirements
- Repository layout
- YOLO-Pose (ONNX) setup
- whisper.cpp setup
- System dependencies
- Build and CMake configuration
- Run instructions
- Voice command summary
- Troubleshooting checklist

## Requirements
- OS: Ubuntu 22.04 (WSL2 supported)
- Python: 3.8+
- CMake >= 3.16
- GCC >= 9 with C++17 support
- Network: Ethernet to robot (default IP 192.168.123.161)
- Workspace root: ~/unitree_ws/src/unitree_sdk2

## Repository layout (example)
Replace ~/unitree_ws/src/unitree_sdk2 with your clone root.
```
~/unitree_ws/src/unitree_sdk2/
├── assets/
│   ├── models/
│   │   └── yolov8/
│   │       ├── yolo11n-pose.onnx
│   │       └── pose.names
│   └── whisper.cpp/
│       ├── models/
│       │   └── ggml-tiny.en.bin
│       └── build/
├── example/go2/
│   ├── human_follow_control.cpp
│   └── CMakeLists.txt
└── build/
    └── bin/
        └── human_follow_control
```

## YOLO-Pose (ONNX) setup
1. Install Python packages:
```
pip3 install --user ultralytics opencv-python onnx onnxruntime
```
2. Create model directory and export:
```
cd ~/unitree_ws/src/unitree_sdk2
mkdir -p assets/models/yolov8
python3 - <<'PY'
from ultralytics import YOLO
m = YOLO('yolo11n-pose.pt')  # will auto-download if needed
m.export(format='onnx', imgsz=640, simplify=True)
PY
# Move exported file if necessary
mv yolo11n-pose.onnx assets/models/yolov8/ || true
```
3. Create pose names:
```
cat > assets/models/yolov8/pose.names <<'EOF'
person
EOF
```
4. Verify:
```
ls -lh assets/models/yolov8/
```

## whisper.cpp setup
1. Clone and download model:
```
cd ~/unitree_ws/src/unitree_sdk2/assets
git clone https://github.com/ggerganov/whisper.cpp.git
cd whisper.cpp
bash ./models/download-ggml-model.sh tiny.en
```
2. Build whisper.cpp:
```
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```
3. Verify:
```
ls -lh build/src/libwhisper.a build/ggml/src/libggml.a
./build/bin/whisper --help || true
```

## System dependencies
Install required OS packages:
```
sudo apt update
sudo apt install -y build-essential cmake git wget curl python3-pip \
    libasound2-dev alsa-utils libopencv-dev libopencv-contrib-dev libgomp1
```
Verify OpenCV (Python):
```
python3 -c "import cv2; print(cv2.__version__)"
```

## CMake / Build configuration
Example additions to CMakeLists.txt (adjust paths to your layout):
```
find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)
find_package(ALSA REQUIRED)

set(WHISPER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/assets/whisper.cpp)
set(WHISPER_BUILD_DIR ${WHISPER_DIR}/build)

include_directories(
  ${OpenCV_INCLUDE_DIRS}
  ${WHISPER_DIR}
  ${WHISPER_DIR}/ggml
)

add_executable(human_follow_control example/go2/human_follow_control.cpp)

target_link_libraries(human_follow_control PRIVATE
  unitree_sdk2
  ${OpenCV_LIBS}
  Threads::Threads
  ${ALSA_LIBRARIES}
  ${WHISPER_BUILD_DIR}/src/libwhisper.a
  ${WHISPER_BUILD_DIR}/ggml/src/libggml.a
  pthread dl m gomp
)
```
Build the project:
```
cd ~/unitree_ws/src/unitree_sdk2
mkdir -p build && cd build
cmake ..
make -j$(nproc)
ls -lh bin/human_follow_control
```

## Run instructions
1. Connect host to robot network interface (check with `ip addr`).
2. Ping robot to confirm:
```
ping -c 3 192.168.123.161
```
3. Run application (replace eth0 with your interface if required):
```
cd ~/unitree_ws/src/unitree_sdk2/build
sudo -E ./bin/human_follow_control eth0
```
Notes:
- Use correct model paths in your application: assets/models/yolov8/yolo11n-pose.onnx and assets/whisper.cpp/models/*.bin
- whisper.cpp expects sample rate and audio device present.

## Voice commands (summary)
Typical usage pattern implemented by the example:
- Trigger mechanism: push-to-talk key (implementation dependent) then speak "robot <command>"
Common commands:
- robot follow / robot stop / robot sit / robot stand
- robot turn around / robot spin
- robot hello / robot dance / robot jump / robot flip

Refer to the application source comments for exact phrases and keyboard controls (W/A/S/D, M toggle, ESC to exit).

## Troubleshooting
- YOLO model missing:
```
ls assets/models/yolov8/yolo11n-pose.onnx
```
- whisper model missing:
```
ls assets/whisper.cpp/models/
```
- Audio device errors:
```
arecord -l
arecord -d 5 -f cd test.wav
```
- Rebuild whisper libraries:
```
cd assets/whisper.cpp/build
make clean
cmake ..
make -j$(nproc)
```
- Rebuild main project after dependencies change:
```
cd build
rm -rf *
cmake ..
make -j$(nproc)
```

## Quick checklist
- [ ] System packages installed
- [ ] YOLO-Pose ONNX exported and present at assets/models/yolov8/
- [ ] pose.names created
- [ ] whisper.cpp cloned and model downloaded
- [ ] whisper.cpp built (libwhisper.a, libggml.a present)
- [ ] CMakeLists configured with correct include/link paths
- [ ] Project built and binary exists at build/bin/human_follow_control
- [ ] Host connected to robot network and microphone is working

If additional integration details are needed (example source edits, exact argument names), consult the example/go2/human_follow_control.cpp and its CMakeLists.txt in this repository.
