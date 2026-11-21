# BDIA-SafetyWalk

[English](README.md) | **한국어**

> 시각장애인에게 보행 신호등이 빨간불인지 초록불인지 음성으로 알려주는, 기기 내(온디바이스) 보행 신호등 인식 안드로이드 앱입니다.

<p align="center">
  <img src="https://velog.velcdn.com/images/m32k/post/5ad53510-0bdf-4f42-861c-07fe85f22c05/image.JPG" width="45%" alt="빨간불 인식" />
  <img src="https://velog.velcdn.com/images/m32k/post/e371d35f-55d3-4147-b53d-eabbaba7c1fd/image.JPG" width="45%" alt="초록불 인식" />
</p>

## 소개

Safety Walk는 시각장애인이 횡단보도를 안전하게 건널 수 있도록 돕는 스마트폰 앱입니다. 후면 카메라를 횡단보도 쪽으로 향하면 보행 신호등을 실시간으로 인식하고, 신호 상태("빨간불" / "초록불")를 TTS(음성 합성)로 알려줍니다.

모든 추론은 ONNX Runtime으로 **기기 안에서** 처리됩니다. 네트워크나 서버가 필요 없어서 통신 상태가 나쁜 곳에서도 지연 없이 동작합니다.

장기적으로는 시각장애인의 자립적인 이동을 돕고, 교통사고를 예방하며, 사회적 포용을 넓히는 것을 목표로 합니다. GPS 기반 경로 안내와 공공데이터 연동 계획은 [로드맵](#로드맵)을 참고하세요.

## 주요 기능

- **실시간 보행 신호등 인식**: 직접 학습한 YOLOv10n 모델이 카메라 영상에서 `greenlight`, `redlight` 두 클래스를 탐지합니다.
- **음성 안내(TTS)**: 신뢰도 80%를 넘는 탐지 결과를 "빨간불, 92% 입니다."처럼 한국어로 읽어줍니다. 안내가 너무 잦지 않도록 5초에 한 번만 말합니다.
- **멀리 있는 작은 신호등을 위한 타일 추론**: 1080×1920 프레임을 서로 겹치는 640×640 타일 6개로 나눠 병렬로 추론합니다. 각 타일의 박스를 전체 프레임 좌표로 되돌린 뒤 NMS(IoU 0.3)로 중복을 합칩니다. 프레임 전체를 축소하지 않기 때문에 멀리 있는 신호등도 충분한 크기로 인식할 수 있습니다.
- **화면 표시**: 카메라 미리보기 위에 바운딩 박스와 클래스·신뢰도를 그려줍니다. 저시력 사용자나 디버깅에 유용합니다.
- **완전 오프라인**: ONNX 모델이 APK 안에 포함되어 있습니다.

## 동작 방식

```
CameraX (후면, 1080×1920)
      │  ImageAnalysis (KEEP_ONLY_LATEST)
      ▼
프레임 회전 → 6개 타일로 분할 (640×640, 겹침)
      │
      ▼  (코루틴 병렬 처리)
ONNX Runtime · YOLOv10n  ──►  타일별 박스 (conf > 0.75)
      │
      ▼
전체 프레임 좌표로 변환 → NMS (IoU 0.3)
      │
      ├──► Compose Canvas: 박스 + 라벨 그리기
      └──► TextToSpeech (ko-KR): conf > 80%일 때 안내, 5초에 최대 1회
```

| 파일 | 역할 |
| --- | --- |
| `MainActivity.kt` | 카메라 권한 요청, TTS 초기화, 결과 수집 및 음성 안내 |
| `MainViewModel.kt` | 전처리, 타일 분할, ONNX 추론, 후처리, NMS |
| `YOLOView.kt` | `CameraView`(CameraX 미리보기·분석), `CanvasView`(박스 오버레이) |
| `PermissionHelper.kt` | 카메라 권한 요청 및 설정 화면 이동 |
| `res/raw/yolov10n.onnx` | 앱에서 사용하는 모델 (약 9MB) |
| `res/raw/yolov10s.onnx` | 더 큰 대체 모델 (약 29MB, 기본으로는 사용하지 않음) |
| `res/raw/classes.txt` | 클래스 라벨: `greenlight`, `redlight` |

## 모델

| 항목 | 내용 |
| --- | --- |
| 구조 | YOLOv10n (Ultralytics 8.1.34) |
| 입력 | `images` — `[1, 3, 640, 640]`, RGB, 0~1 정규화 |
| 출력 | `output0` — `[1, 300, 6]` (`x1, y1, x2, y2, score, class`), NMS-free |
| 클래스 | `0: greenlight`, `1: redlight` |
| 데이터셋 | [pedestrian-light_2024 (Roboflow Universe, v4)](https://universe.roboflow.com/guk/pedestrian-light_2024/dataset/4) |
| 학습 | Google Colab — [노트북](https://colab.research.google.com/drive/1k8rkL-8aflN1hu1m8k_q6cjrRBPRnXyQ?usp=sharing) |

### 학습 과정

모델은 아래 참고 자료의 블로그 글에 정리한 과정으로 만들었습니다.

1. **데이터셋 (Roboflow)**: 횡단보도 사진에서 신호등 불빛 부분만 박스로 라벨링했습니다(잔여 시간 표시는 제외). 전처리는 Auto-Orient와 640×640 Resize, 증강은 밝기 ±12%, 흐림 1.5px, 노이즈 0.14%(최대 3배)로 설정했습니다. 훈련/검증/테스트는 6:2:2로 나누고 YOLOv8 형식으로 내보냈습니다(YOLOv10도 같은 라벨 형식을 씁니다).
2. **학습 (Google Colab, T4 GPU)**: [THU-MIG/yolov10](https://github.com/THU-MIG/yolov10)으로 사전 학습된 `yolov10n.pt`에서 시작했습니다.

   ```bash
   yolo task=detect mode=train epochs=50 batch=16 plots=True \
     model=weights/yolov10n.pt data=<dataset>/data.yaml
   ```
3. **내보내기**: `runs/detect/train/weights/best.pt`를 ONNX로 변환한 뒤 `app/src/main/res/raw/`에 넣었습니다.

더 큰 `yolov10s` 모델을 쓰려면 `MainViewModel.readModel()`의 `R.raw.yolov10n`을 `R.raw.yolov10s`로 바꾸면 됩니다. 정확도는 더 높지만 느리고 용량이 큽니다.

### 직접 모델 학습하기

1. 횡단보도 사진을 모아 Roboflow에서 라벨링 → [Roboflow를 활용한 커스텀 이미지 데이터 셋 만들기](https://velog.io/@m32k/Roboflow%EB%A5%BC-%ED%99%9C%EC%9A%A9%ED%95%9C-%EC%BB%A4%EC%8A%A4%ED%85%80-%EC%9D%B4%EB%AF%B8%EC%A7%80-%EB%8D%B0%EC%9D%B4%ED%84%B0-%EC%85%8B-%EB%A7%8C%EB%93%A4%EA%B8%B0)
2. Colab에서 YOLOv10 학습 후 ONNX로 내보내기 → [Colab으로 YOLOv10을 활용하여 Roboflow 데이터셋으로 객체 탐지 모델 학습](https://velog.io/@m32k/Colab%EC%9C%BC%EB%A1%9C-YOLOv10-%EC%9D%84-%ED%99%9C%EC%9A%A9%ED%95%98%EC%97%AC-Roboflow-%EB%8D%B0%EC%9D%B4%ED%84%B0%EC%85%8B%EC%9C%BC%EB%A1%9C-%EA%B0%9D%EC%B2%B4-%ED%83%90%EC%A7%80-%EB%AA%A8%EB%8D%B8-%ED%95%99%EC%8A%B5)
3. `app/src/main/res/raw/yolov10n.onnx`를 교체하고, `classes.txt`를 `data.yaml`과 같은 클래스 순서로 수정합니다.

## 기술 스택

- Kotlin, Jetpack Compose (Material 3)
- CameraX 1.3.3
- ONNX Runtime for Android 1.18.0
- Android TextToSpeech
- Kotlin Coroutines / StateFlow
- 학습: Ultralytics YOLOv10, Roboflow, Google Colab

## 시작하기

**준비물**

- Android Studio (Hedgehog 이상 권장), JDK 17
- 후면 카메라가 있는 Android 7.0(API 24) 이상 기기
- 기기에 한국어 TTS 음성 설치 (설정 → 텍스트 음성 변환)

**빌드 및 실행**

```bash
git clone https://github.com/M32k/BDIA-SafetyWalk.git
cd BDIA-SafetyWalk
./gradlew installDebug
```

또는 Android Studio에서 프로젝트를 열고 **Run**을 누르면 됩니다. 처음 실행할 때 카메라 권한을 허용한 뒤 카메라를 보행 신호등 쪽으로 향하세요.

> 실제 기기에서 실행하는 것을 권장합니다. 에뮬레이터는 쓸 만한 후면 카메라가 없는 경우가 많고 추론도 느립니다.

## 알려진 한계

- 타일 분할 코드가 1080×1920 카메라 프레임을 전제로 합니다. 분석 해상도가 다른 기기에서는 앱이 종료되거나 박스 위치가 어긋날 수 있습니다.
- 신뢰도가 가장 높은 탐지 결과 하나만 안내합니다. 여러 신호등이 보일 때 어느 것이 사용자의 횡단보도 신호인지는 구분하지 않습니다.
- 음성 안내는 한국어만 지원합니다.
- 조명, 거리, 각도에 따라 인식 성능이 달라집니다. **이 앱은 보조 수단일 뿐, 흰 지팡이·안내견·사용자 본인의 판단을 대신하지 않습니다.**

## 로드맵

- [ ] GPS·센서 기반 위치 파악으로 가장 가까운(최적의) 횡단보도 안내
- [ ] 공공기관의 보행 신호등 데이터(위치·신호 주기) 연동
- [ ] 초록불 잔여 시간 안내
- [ ] 음성과 함께 진동(햅틱) 피드백 제공
- [ ] 다양한 카메라 해상도와 화면 방향 지원
- [ ] 영어 등 다른 언어 TTS 지원
- [ ] 방향 안내 (예: "왼쪽에 신호등이 있습니다")

## 참고 자료

- [Roboflow를 활용한 커스텀 이미지 데이터 셋 만들기 — velog](https://velog.io/@m32k/Roboflow%EB%A5%BC-%ED%99%9C%EC%9A%A9%ED%95%9C-%EC%BB%A4%EC%8A%A4%ED%85%80-%EC%9D%B4%EB%AF%B8%EC%A7%80-%EB%8D%B0%EC%9D%B4%ED%84%B0-%EC%85%8B-%EB%A7%8C%EB%93%A4%EA%B8%B0)
- [Colab으로 YOLOv10을 활용하여 Roboflow 데이터셋으로 객체 탐지 모델 학습 — velog](https://velog.io/@m32k/Colab%EC%9C%BC%EB%A1%9C-YOLOv10-%EC%9D%84-%ED%99%9C%EC%9A%A9%ED%95%98%EC%97%AC-Roboflow-%EB%8D%B0%EC%9D%B4%ED%84%B0%EC%85%8B%EC%9C%BC%EB%A1%9C-%EA%B0%9D%EC%B2%B4-%ED%83%90%EC%A7%80-%EB%AA%A8%EB%8D%B8-%ED%95%99%EC%8A%B5)
- [Colab 학습 노트북](https://colab.research.google.com/drive/1k8rkL-8aflN1hu1m8k_q6cjrRBPRnXyQ?usp=sharing)
- [Roboflow 데이터셋: pedestrian-light_2024](https://universe.roboflow.com/guk/pedestrian-light_2024/dataset/4)
- [YOLOv10: Real-Time End-to-End Object Detection — 저장소](https://github.com/THU-MIG/yolov10) / [논문 (arXiv)](https://arxiv.org/abs/2405.14458)
- [Ultralytics 내보내기(Export) 문서](https://docs.ultralytics.com/modes/export/)
- [ONNX Runtime](https://onnxruntime.ai)

## 감사의 말

- 안드로이드 앱은 [Aloe-droid/YOLOv10_android](https://github.com/Aloe-droid/YOLOv10_android)([velog 글](https://velog.io/@aloe/YOLOv10-%EC%95%88%EB%93%9C%EB%A1%9C%EC%9D%B4%EB%93%9C))를 바탕으로 시작했습니다. Safety Walk는 여기에 직접 학습한 신호등 모델, 한국어 음성 안내, NMS를 포함한 타일 추론, 시각장애인 사용을 고려한 동작을 더했습니다.
- THU-MIG의 [YOLOv10](https://github.com/THU-MIG/yolov10)과 그 기반인 [Ultralytics](https://github.com/ultralytics/ultralytics)
- 데이터셋 라벨링과 배포에 사용한 [Roboflow](https://roboflow.com)

## 라이선스

아직 이 저장소에는 라이선스 파일이 없습니다. 포함된 모델은 Ultralytics로 학습·변환되었고, Ultralytics 모델은 **AGPL-3.0**으로 배포된다는 점을 라이선스를 정하거나 앱을 재배포할 때 고려하세요.
