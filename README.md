# BDIA-SafetyWalk

**English** | [한국어](README.ko.md)

> An on-device pedestrian traffic light detector for Android that tells visually impaired users, by voice, whether the light is red or green.

<p align="center">
  <img src="https://velog.velcdn.com/images/m32k/post/5ad53510-0bdf-4f42-861c-07fe85f22c05/image.JPG" width="45%" alt="Red light detected" />
  <img src="https://velog.velcdn.com/images/m32k/post/e371d35f-55d3-4147-b53d-eabbaba7c1fd/image.JPG" width="45%" alt="Green light detected" />
</p>

## About

Safety Walk is a smartphone app that helps visually impaired people cross crosswalks safely. Point the rear camera toward the crossing, and the app detects the pedestrian signal in real time and announces its state ("red light" / "green light") through text-to-speech.

All inference runs **on the device** with ONNX Runtime. No network connection or server is needed, so it works with low latency even where reception is poor.

The long-term goal is to support independent mobility for visually impaired people, prevent traffic accidents, and strengthen social inclusion. See the [Roadmap](#roadmap) for the planned GPS-based route guidance and public-data integration.

## Features

- **Real-time pedestrian light detection**: a custom-trained YOLOv10n model detects two classes, `greenlight` and `redlight`, from the camera stream.
- **Voice guidance (TTS)**: when a detection has confidence above 80%, the app speaks the result in Korean, e.g. "빨간불, 92% 입니다." ("Red light, 92%."). Announcements are throttled to one every 5 seconds to avoid flooding the user.
- **Tiled inference for small, distant signals**: each 1080×1920 frame is split into six overlapping 640×640 tiles. Each tile is inferred in parallel, the boxes are mapped back to full-frame coordinates, and Non-Maximum Suppression (IoU 0.3) merges duplicates. This keeps far-away signals large enough to detect instead of downscaling the whole frame.
- **Visual overlay**: bounding boxes and class/confidence labels are drawn over the camera preview (useful for low-vision users and for debugging).
- **Fully offline**: the ONNX model ships inside the APK.

## How it works

```
CameraX (rear, 1080×1920)
      │  ImageAnalysis (KEEP_ONLY_LATEST)
      ▼
Rotate frame → split into 6 tiles (640×640, overlapping)
      │
      ▼  (coroutines, parallel)
ONNX Runtime · YOLOv10n  ──►  boxes per tile (conf > 0.75)
      │
      ▼
Offset to full-frame coords → NMS (IoU 0.3)
      │
      ├──► Compose Canvas: draw boxes + labels
      └──► TextToSpeech (ko-KR): announce if conf > 80%, max once / 5 s
```

| File | Role |
| --- | --- |
| `MainActivity.kt` | Requests camera permission, sets up TTS, collects results and speaks them |
| `MainViewModel.kt` | Pre-processing, tiling, ONNX inference, post-processing, NMS |
| `YOLOView.kt` | `CameraView` (CameraX preview + analysis) and `CanvasView` (box overlay) |
| `PermissionHelper.kt` | Camera permission request and redirect to settings |
| `res/raw/yolov10n.onnx` | Model used by the app (≈9 MB) |
| `res/raw/yolov10s.onnx` | Larger alternative model (≈29 MB, not loaded by default) |
| `res/raw/classes.txt` | Class labels: `greenlight`, `redlight` |

## Model

| Item | Value |
| --- | --- |
| Architecture | YOLOv10n (Ultralytics 8.1.34) |
| Input | `images` — `[1, 3, 640, 640]`, RGB, scaled to 0–1 |
| Output | `output0` — `[1, 300, 6]` (`x1, y1, x2, y2, score, class`), NMS-free |
| Classes | `0: greenlight`, `1: redlight` |
| Dataset | [pedestrian-light_2024 (Roboflow Universe, v4)](https://universe.roboflow.com/guk/pedestrian-light_2024/dataset/4) |
| Training | Google Colab — [notebook](https://colab.research.google.com/drive/1k8rkL-8aflN1hu1m8k_q6cjrRBPRnXyQ?usp=sharing) |

To use the larger `yolov10s` model, change `R.raw.yolov10n` to `R.raw.yolov10s` in `MainViewModel.readModel()`. It is more accurate but slower and larger.

### Train your own model

1. Collect crosswalk photos and label them in Roboflow → [Creating a Custom Image Dataset with Roboflow (Korean)](https://velog.io/@m32k/Roboflow%EB%A5%BC-%ED%99%9C%EC%9A%A9%ED%95%9C-%EC%BB%A4%EC%8A%A4%ED%85%80-%EC%9D%B4%EB%AF%B8%EC%A7%80-%EB%8D%B0%EC%9D%B4%ED%84%B0-%EC%85%8B-%EB%A7%8C%EB%93%A4%EA%B8%B0)
2. Train YOLOv10 in Colab and export to ONNX → [Training YOLOv10 on a Roboflow dataset in Colab (Korean)](https://velog.io/@m32k/Colab%EC%9C%BC%EB%A1%9C-YOLOv10-%EC%9D%84-%ED%99%9C%EC%9A%A9%ED%95%98%EC%97%AC-Roboflow-%EB%8D%B0%EC%9D%B4%ED%84%B0%EC%85%8B%EC%9C%BC%EB%A1%9C-%EA%B0%9D%EC%B2%B4-%ED%83%90%EC%A7%80-%EB%AA%A8%EB%8D%B8-%ED%95%99%EC%8A%B5)
3. Replace `app/src/main/res/raw/yolov10n.onnx` and update `classes.txt` in the same class order as your `data.yaml`.

## Tech stack

- Kotlin, Jetpack Compose (Material 3)
- CameraX 1.3.3
- ONNX Runtime for Android 1.18.0
- Android TextToSpeech
- Kotlin Coroutines / StateFlow
- Training: Ultralytics YOLOv10, Roboflow, Google Colab

## Getting started

**Requirements**

- Android Studio (Hedgehog or newer recommended), JDK 17
- An Android device running Android 7.0 (API 24) or newer with a rear camera
- Korean TTS voice installed on the device (Settings → Text-to-speech)

**Build and run**

```bash
git clone https://github.com/M32k/BDIA-SafetyWalk.git
cd BDIA-SafetyWalk
./gradlew installDebug
```

Or open the project in Android Studio and press **Run**. Grant camera permission on first launch, then point the camera toward a pedestrian signal.

> A real device is recommended. Emulators usually lack a usable rear camera, and inference is slow.

## Known limitations

- The tiling code assumes a 1080×1920 camera frame. Devices that deliver a different analysis resolution may crash or get misplaced boxes.
- Only the single highest-confidence detection is announced. If several signals are visible, the app does not tell which one belongs to the user's crosswalk.
- Voice messages are Korean only.
- Detection quality depends on lighting, distance and angle. **This app is an assistive aid, not a replacement for a white cane, guide dog or the user's own judgment.**

## Roadmap

- [ ] GPS and sensor-based positioning to guide users to the nearest/optimal crosswalk
- [ ] Integrate public pedestrian signal data (signal locations and timing) from public institutions
- [ ] Remaining green time announcements
- [ ] Haptic (vibration) feedback in addition to voice
- [ ] Support arbitrary camera resolutions and both orientations
- [ ] English and other TTS languages
- [ ] Direction cues (e.g. "signal on your left")

## References

- [Creating a Custom Image Dataset with Roboflow — velog (Korean)](https://velog.io/@m32k/Roboflow%EB%A5%BC-%ED%99%9C%EC%9A%A9%ED%95%9C-%EC%BB%A4%EC%8A%A4%ED%85%80-%EC%9D%B4%EB%AF%B8%EC%A7%80-%EB%8D%B0%EC%9D%B4%ED%84%B0-%EC%85%8B-%EB%A7%8C%EB%93%A4%EA%B8%B0)
- [Training YOLOv10 on a Roboflow dataset in Colab — velog (Korean)](https://velog.io/@m32k/Colab%EC%9C%BC%EB%A1%9C-YOLOv10-%EC%9D%84-%ED%99%9C%EC%9A%A9%ED%95%98%EC%97%AC-Roboflow-%EB%8D%B0%EC%9D%B4%ED%84%B0%EC%85%8B%EC%9C%BC%EB%A1%9C-%EA%B0%9D%EC%B2%B4-%ED%83%90%EC%A7%80-%EB%AA%A8%EB%8D%B8-%ED%95%99%EC%8A%B5)
- [Colab training notebook](https://colab.research.google.com/drive/1k8rkL-8aflN1hu1m8k_q6cjrRBPRnXyQ?usp=sharing)
- [Roboflow dataset: pedestrian-light_2024](https://universe.roboflow.com/guk/pedestrian-light_2024/dataset/4)
- [YOLOv10: Real-Time End-to-End Object Detection](https://github.com/THU-MIG/yolov10)

## License

No license file has been added to this repository yet. Note that the bundled models were trained and exported with Ultralytics, whose models are distributed under **AGPL-3.0**; keep this in mind when choosing a license or redistributing the app.
