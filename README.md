# 🚘 YOLO-Vision: Real-Time Multi-Vehicle Analysis & CoreML Acceleration
> **실시간 다중 차량 동시 분석 및 CoreML 기반 추론 속도 극한 가속 파이프라인**
> Apple Silicon(MPS) 환경에서의 모델 최적화(Fine-tuning)부터 ANE(Apple Neural Engine)를 활용한 Edge Device 추론 시간 최소화까지 구현한 고정밀 비전 검사 프로젝트입니다.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-yellow.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-MPS_Support-EE4C2C.svg)
![CoreML](https://img.shields.io/badge/CoreML-Apple_Neural_Engine-black.svg)

---

## 🎬 Project Demo Showcase
> **CoreML 기반 경량화 모델의 실시간 다중 객체(Bus, Truck, Car) 탐지 및 고속 추적(Tracking) 시연**

<div align="center">
  <video src="https://github.com/gyuminkang/yolo-vision-multi-inspection-coreml/blob/main/visualization/cctv.mp4?raw=true" width="100%" autoplay loop muted playsinline></video>
</div>

---

## 📑 Table of Contents
1. [Data Preprocessing & Class Unification (`checking.ipynb`)](#1-data-preprocessing--class-unification)
2. [Image Size Fine-Tuning Benchmarking (`Size_finetune.ipynb`)](#2-image-size-fine-tuning-benchmarking)
3. [Epoch Fine-Tuning & Tracking (`Epoch_finetune.ipynb`)](#3-epoch-fine-tuning--tracking)
4. [Architecture Benchmarking & CoreML Export (`run.ipynb`)](#4-architecture-benchmarking--coreml-export)
5. [Training Metrics & Visualizations](#5-training-metrics--visualizations)
6. [Final Inference Case Study](#6-final-inference-case-study)

---

## 1. Data Preprocessing & Class Unification
**📂 Notebook:** `checking.ipynb`

모델 학습 전, 데이터셋의 무결성을 검증하고 다양한 형태로 파편화된 원본 라벨을 실무 표준 스펙에 맞춰 상위 클래스로 병합(Merge)하는 전처리 파이프라인을 구축했습니다.

* **통합 매핑 규칙(Class Mapping) 및 분포 최적화:** 총 12개의 세부 라벨로 흩어져 있던 51,577개의 객체를 3개의 핵심 클래스(`Bus`, `Truck`, `Car`)로 통합 매핑 처리 및 인덱싱 재정렬을 완료하여 모델의 분류 혼동(Confusion)을 최소화했습니다.

| Target Class (ID) | Original Sub-classes (파편화된 원본 라벨) | Object Count |
| :--- | :--- | :--- |
| **Bus (0)** | `big bus`, `bus-l-`, `bus-s-`, `small bus` | 1,625 |
| **Truck (1)** | `big truck`, `mid truck`, `small truck`, `truck-l-`, `truck-m-`, `truck-s-`, `truck-xl-` | 18,311 |
| **Car (2)** | `car` | 31,641 |
| **Total Objects** | **총 12개 하위 클래스 정제 완료** | **51,577** |

* **데이터 무결성 검증:** `train/valid/test` 내부의 모든 txt 라벨 데이터를 전수 조사하여 Bounding Box 좌표 구조의 결함 여부 및 Out-of-bounds 오류를 자동 스캔.

## 2. Image Size Fine-Tuning Benchmarking
**📂 Notebook:** `Size_finetune.ipynb`

다중 객체 탐지 장비의 하드웨어 제약 조건과 판정 정밀도 간의 최적의 타협점(Trade-off)을 도출하기 위해, 입력 해상도(`imgsz`)별 스케일링 벤치마크 실험을 수행했습니다.

* **해상도별 성능 및 추론 속도 벤치마크 (Test Set 기준):**

| Rank | Resolution (`imgsz`) | mAP@0.50 🏆 | mAP@0.5-0.95 | Precision | Recall | Inference Latency |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **1** | **640 (Optimal)** | **0.7786** | 0.4704 | 0.7439 | **0.7888** | **5.22 ms** |
| 2 | 1024 | 0.7783 | 0.4669 | 0.7557 | 0.7861 | 12.76 ms |
| 3 | 768 | 0.7776 | 0.4708 | 0.7698 | 0.7677 | 8.32 ms |
| 4 | 320 | 0.7220 | 0.4184 | 0.7338 | 0.6967 | 1.60 ms |

> **💡 Architecture Insight:** 벤치마크 분석 결과, `imgsz=640` 설정이 초고해상도(1024) 대비 **추론 속도를 약 2.4배(12.76ms → 5.22ms) 단축**시키면서도 **최고조의 판정 성능(mAP@0.50 0.7786 및 재현율 0.7888)**을 기록하였습니다. 이에 따라 리소스 대비 효율이 가장 뛰어난 640 사이즈를 최종 추론 파이프라인의 기준 해상도로 채택했습니다.

* **커스텀 메모리 관리 (MPS 최적화):** Apple Silicon 환경에서 연속적인 모델 튜닝(`model.tune`) 시 발생하는 공유 메모리 누수를 원천 차단하기 위해, `on_tune_epoch_end` 이벤트 시점마다 가비지 컬렉터(`gc.collect()`) 및 GPU 캐시(`torch.mps.empty_cache()`)를 강제 릴리즈하는 메모리 클린업 콜백 구현.

## 3. Epoch Fine-Tuning & Tracking
**📂 Notebook:** `Epoch_finetune.ipynb`

손실 함수 수렴을 극대화하기 위한 학습 최적화 및 고속 주행 환경에서의 객체 유실을 방지하는 추적(Tracking) 알고리즘 고도화를 진행했습니다.

* **Epoch 최적화 벤치마크 및 Catastrophic Forgetting 분석:**

| 학습 기간 (Epochs) | Test mAP@0.50 🏆 | 성능 변화 추이 |
| :---: | :---: | :--- |
| **10 (Optimal)** | **0.7672** | **최상단 성능 달성 (과적합 방지 성공)** |
| 50 | 0.7655 | 미세 하락 발생 |
| 150 | 0.7642 | 하락폭 증가 |
| 300 | 0.7636 | 지속적인 성능 저하 현상 확인 |

> **💡 Fine-Tuning Insight (Overfitting & Catastrophic Forgetting):**
> YOLOv8 아키텍처는 수백만 장의 이미지로 이미 강력한 일반화 성능을 갖춘 사전 학습 가중치(Pre-trained Weights)를 보유하고 있습니다. 벤치마크 결과, 에포크가 늘어날수록 오히려 테스트 성능이 감소하는 현상이 관찰되었습니다. 이는 모델에 강제적인 추가 학습을 과도하게 지시할 경우, 기존의 정교한 가중치 밸런스가 무너지는 **과적합(Overfitting)** 및 **재앙적 망각(Catastrophic Forgetting)**이 발생하기 때문입니다. 따라서 Early Stopping을 적극 활용하여 에포크를 10회 내외로 제한하는 것이 모델의 특징 추출 능력을 가장 예리하게 유지하는 최적점임을 입증했습니다.

* **ByteTrack 알고리즘 가속 최적화:** 프레임 간 객체 폐색(Occlusion) 및 일시적 유실 문제를 해결하기 위해 버퍼 파라미터 튜닝.
  * `track_buffer`: 100 ➡️ **300 프레임** 확장 (일시적으로 놓친 객체의 ID 유지력을 약 10초간 대폭 향상)
  * `track_high_thresh`: 임계값 제어를 통해 노이즈로 인한 오탐(False Positive) 판정율 최소화.

## 4. Architecture Benchmarking & CoreML Export
**📂 Notebook:** `run.ipynb`

앞서 도출한 최적의 파라미터 환경(`Epoch=10`, `imgsz=640`) 하에서, 1차적으로 모델 규모(Capacity)에 따른 아키텍처 벤치마크를 수행하고, 2차적으로 실시간 병목 현상을 타파하기 위한 하드웨어 가속(Export) 전략을 수립했습니다.

* **1차: YOLOv8 모델 규모별 성능 및 추론 속도 벤치마크**

| Rank | Model Architecture | mAP@0.50 🏆 | mAP@0.5-0.95 | Precision | Recall | Inference Latency |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **1** | **Medium (채택)** | **0.7839** | 0.4660 | 0.7609 | 0.7834 | 12.92 ms |
| 2 | Small | 0.7768 | 0.4698 | 0.7436 | **0.7886** | 5.27 ms |
| 3 | Nano | 0.7562 | 0.4413 | **0.7625** | 0.7003 | **2.43 ms** |

> **💡 Bottleneck Analysis:** 벤치마크 결과, Medium 아키텍처가 `mAP@0.50 0.7839`로 가장 압도적인 판정 정밀도를 달성하여 최종 모델로 채택되었습니다. 하지만 모델이 무거워짐에 따라 Edge Device 환경에서 실시간(Real-time) 추론 시 지연 시간(Latency)이 다소 길어지는 병목 현상이 식별되었습니다. 이를 해결하기 위해 포맷 변환을 통한 하드웨어 가속을 시도했습니다.

* **2차: 하드웨어 가속 런타임 벤치마크 (Medium 아키텍처 적용)**

| Export Format | mAP@0.50 | mAP@0.5-0.95 | Precision | Recall | Inference Latency 🚀 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **PyTorch (Baseline)** | 0.7839 | 0.4660 | 0.7609 | 0.7834 | 13.09 ms |
| ONNX | 0.7818 | 0.4637 | 0.7619 | 0.7748 | 10.29 ms |
| **CoreML (Final)** | **0.7824** | 0.4633 | **0.7629** | 0.7752 | **9.05 ms** |

> **💡 Hardware Acceleration Insight (CoreML 채택 타당성):**
> 원본 PyTorch 가중치를 산업 표준인 ONNX와 Apple 생태계 전용인 CoreML로 변환하여 교차 검증을 수행했습니다. 그 결과, **CoreML 포맷**이 원본 모델의 판정 정밀도를 완벽에 가깝게 보존(`mAP@0.50` 기준 단 0.19% 차이)하면서도, **추론 지연 시간을 13.09ms에서 9.05ms로 약 31% 대폭 단축(처리 속도 1.45배 향상)**시키는 압도적인 퍼포먼스를 입증했습니다. 이는 Apple Neural Engine(ANE) 프로세서의 연산 가속 능력을 극대화한 결과이며, 최종적으로 고정밀 Medium 아키텍처를 실시간 다중 검사 환경에 무손실로 배포하는 데 성공했습니다.

---

## 5. Training Metrics & Visualizations
에포크 최적화 과정에서 도출된 하이퍼파라미터 수렴 및 모델 평가지표 시각화 결과입니다.

<div align="center">
  <img src="visualization/Bounding_Box_Loss_Convergence.png" width="48%" alt="Box Loss">
  <img src="visualization/Classification_Loss_Convergence.png" width="48%" alt="Cls Loss">
</div>
<br>
<div align="center">
  <img src="visualization/Distribution_Focal_Loss(DFL)_Convergence.png" width="48%" alt="DFL Loss">
  <img src="visualization/Learning_Rate_Scheduling.png" width="48%" alt="LR Scheduling">
</div>
<br>
<div align="center">
  <img src="visualization/Validation_Metrics_Precision&_Recall.png" width="48%" alt="Precision Recall">
  <img src="visualization/Validation_Metrics_Mean_Average_Precision(mAP).png" width="48%" alt="mAP Score">
</div>

* **Loss Convergence:** Bounding Box, Classification, DFL(Distribution Focal Loss)의 Train/Val 곡선이 이격 없이 안정적으로 하향 수렴하며 이상적인 일반화 성능 달성.
* **Validation Metrics:** Precision과 Recall의 밸런스가 균형 있게 상승하였으며, 최종 mAP@0.50 지표가 최상위권에 안착하여 판정 모델로서의 높은 신뢰성 검증.

---

## 6. Final Inference Case Study
경량화 및 하드웨어 셋업이 완료된 CoreML 최적화 가중치를 활용하여 실제 테스트 셋 이미지 내 다중 객체를 바운딩 박스 단위로 판정한 결과입니다.

<div align="center">
  <img src="visualization/CoreML_Detection_Best_Examples_with_Bus_Truck_and_Car.png" width="100%" alt="CoreML Detection Result Analysis">
</div>

---
**👨‍💻 Author:** gyuminkang  
**📧 Contact:** [본인 이메일 주소]
