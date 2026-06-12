# 🚘 YOLO-Vision: Real-Time Multi-Vehicle Analysis & CoreML Acceleration
> **Real-Time Multi-Vehicle Simultaneous Analysis & Extreme Inference Acceleration Pipeline via CoreML**
> A high-precision vision inspection project covering model optimization (Fine-tuning) in an Apple Silicon (MPS) environment to minimizing inference latency on Edge Devices using the Apple Neural Engine (ANE).

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-yellow.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-MPS_Support-EE4C2C.svg)
![CoreML](https://img.shields.io/badge/CoreML-Apple_Neural_Engine-black.svg)

---

## 🎬 Project Demo Showcase
> **Demonstrating real-time multi-object (Bus, Truck, Car) detection and high-speed tracking, validated using live, real-world CCTV footage from the Wongok Sleep Rest Area on the Gyeongbu Expressway (KEC Suwon Branch jurisdiction, Seoul/Busan direction).**

<div align="center">
  <img src="visualization/cctv.gif" width="50%" alt="YOLOv8 CoreML Real-time Tracking Demo">
</div>

---

## 1. Data Preprocessing & Class Unification
**📂 Notebook:** `checking.ipynb`

Prior to model training, we built a preprocessing pipeline to verify dataset integrity and merge fragmented original labels into standardized macro-classes suited for industrial specifications.

* **Class Mapping & Distribution Optimization:** A total of 51,577 objects scattered across 12 sub-labels were unified and re-indexed into 3 core classes (`Bus`, `Truck`, `Car`) to minimize model classification confusion.

| Target Class (ID) | Original Sub-classes (Fragmented Labels) | Object Count |
| :--- | :--- | :--- |
| **Bus (0)** | `big bus`, `bus-l-`, `bus-s-`, `small bus` | 1,625 |
| **Truck (1)** | `big truck`, `mid truck`, `small truck`, `truck-l-`, `truck-m-`, `truck-s-`, `truck-xl-` | 18,311 |
| **Car (2)** | `car` | 31,641 |
| **Total Objects** | **Successfully Refined from 12 Sub-classes** | **51,577** |

* **Data Integrity Verification:** Conducted a full scan of all `.txt` label data within the `train/valid/test` sets to automatically detect Bounding Box coordinate defects and out-of-bounds errors.

## 2. Image Size Fine-Tuning Benchmarking
**📂 Notebook:** `Size_finetune.ipynb`

To find the optimal trade-off between the hardware constraints of multi-object detection equipment and detection precision, we conducted a scaling benchmark experiment across various input resolutions (`imgsz`).

* **Resolution Performance & Inference Latency Benchmark (Test Set):**

| Rank | Resolution (`imgsz`) | mAP@0.50 🏆 | mAP@0.5-0.95 | Precision | Recall | Inference Latency |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **1** | **640 (Optimal)** | **0.7786** | 0.4704 | 0.7439 | **0.7888** | **5.22 ms** |
| 2 | 1024 | 0.7783 | 0.4669 | 0.7557 | 0.7861 | 12.76 ms |
| 3 | 768 | 0.7776 | 0.4708 | 0.7698 | 0.7677 | 8.32 ms |
| 4 | 320 | 0.7220 | 0.4184 | 0.7338 | 0.6967 | 1.60 ms |

> **💡 Architecture Insight:** The benchmark analysis revealed that the `imgsz=640` setting reduced inference latency by approximately 2.4x (12.76ms → 5.22ms) compared to ultra-high resolution (1024) while achieving peak detection performance (mAP@0.50 0.7786 and Recall 0.7888). Consequently, the highly resource-efficient 640 size was adopted as the baseline resolution for the final inference pipeline.

* **Custom Memory Management (MPS Optimization):** To completely prevent shared memory leaks occurring during continuous model tuning (`model.tune`) in the Apple Silicon environment, a memory cleanup callback was implemented. It forces garbage collection (`gc.collect()`) and GPU cache release (`torch.mps.empty_cache()`) at the end of every `on_tune_epoch_end` event.

## 3. Epoch Fine-Tuning & Tracking
**📂 Notebook:** `Epoch_finetune.ipynb`

We advanced training optimization to maximize loss function convergence and enhanced the tracking algorithm to prevent object loss in high-speed driving environments.

* **Epoch Optimization Benchmark & Catastrophic Forgetting Analysis:**

| Training Duration (Epochs) | Test mAP@0.50 🏆 | Performance Trend |
| :---: | :---: | :--- |
| **10 (Optimal)** | **0.7672** | **Achieved Peak Performance (Prevented Overfitting)** |
| 50 | 0.7655 | Minor degradation |
| 150 | 0.7642 | Increased degradation |
| 300 | 0.7636 | Continuous performance drop observed |

> **💡 Fine-Tuning Insight (Overfitting & Catastrophic Forgetting):**
> The YOLOv8 architecture already possesses pre-trained weights with strong generalization capabilities learned from millions of images. Benchmark results showed that test performance paradoxically degraded as epochs increased. This indicates that forcing excessive additional training disrupts the delicate balance of existing weights, leading to **Overfitting** and **Catastrophic Forgetting**. Thus, actively utilizing Early Stopping to limit epochs to around 10 proved to be the optimal strategy for keeping the model's feature extraction capability sharpest.

* **ByteTrack Algorithm Acceleration:** Tuned buffer parameters to resolve inter-frame occlusion and temporary object loss.
  * `track_buffer`: Expanded from 100 ➡️ **300 frames** (significantly improving ID retention of temporarily missed objects to about 10 seconds).
  * `track_high_thresh`: Minimized False Positive rates caused by noise through strict threshold control.

## 4. Architecture Benchmarking & CoreML Export
**📂 Notebook:** `run.ipynb`

Under the previously derived optimal parameter environment (`Epoch=10`, `imgsz=640`), we first conducted an architecture benchmark based on model capacity, followed by establishing a hardware acceleration (Export) strategy to break through real-time bottlenecks.

* **Phase 1: YOLOv8 Model Capacity Performance & Latency Benchmark**

| Rank | Model Architecture | mAP@0.50 🏆 | mAP@0.5-0.95 | Precision | Recall | Inference Latency |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **1** | **Medium (Selected)** | **0.7839** | 0.4660 | 0.7609 | 0.7834 | 12.92 ms |
| 2 | Small | 0.7768 | 0.4698 | 0.7436 | **0.7886** | 5.27 ms |
| 3 | Nano | 0.7562 | 0.4413 | **0.7625** | 0.7003 | **2.43 ms** |

> **💡 Bottleneck Analysis:** The benchmark results showed the Medium architecture achieved the most dominant detection precision with `mAP@0.50 0.7839`, leading to its selection. However, due to the increased model weight, a bottleneck in inference latency (12.92 ms) was identified during real-time inference on Edge Devices. To resolve this, we attempted hardware acceleration via format conversion.

* **Phase 2: Hardware Accelerated Runtime Benchmark (Applying Medium Architecture)**

| Export Format | mAP@0.50 | mAP@0.5-0.95 | Precision | Recall | Inference Latency 🚀 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **PyTorch (Baseline)** | 0.7839 | 0.4660 | 0.7609 | 0.7834 | 13.09 ms |
| ONNX | 0.7818 | 0.4637 | 0.7619 | 0.7748 | 10.29 ms |
| **CoreML (Final)** | **0.7824** | 0.4633 | **0.7629** | 0.7752 | **9.05 ms** |

> **💡 Hardware Acceleration Insight (Justification for CoreML Adoption):**
> We cross-validated the original PyTorch weights by converting them to the industrial standard ONNX and the Apple ecosystem-exclusive CoreML. The results demonstrated that the **CoreML format** nearly perfectly preserved the original model's precision (only a 0.15% difference in `mAP@0.50`) while drastically **reducing inference latency from 13.09ms to 9.05ms (a ~31% reduction, translating to a 1.45x speedup)**. This proves the maximized computational acceleration capabilities of the Apple Neural Engine (ANE) processor, successfully enabling the deployment of the high-precision Medium architecture in a real-time multi-inspection environment without loss.

---

## 5. Training Metrics & Visualizations
Visualizations of hyperparameter convergence and model evaluation metrics derived during the epoch optimization process.

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

* **Loss Convergence:** Train/Val curves for Bounding Box, Classification, and DFL (Distribution Focal Loss) stably converge downward without divergence, achieving ideal generalization performance.
* **Validation Metrics:** Precision and Recall balanced well, and the final mAP@0.50 metric settled in the top tier, verifying high reliability as an inspection model.

---

## 6. Final Inference Case Study
Results of detecting multiple objects at the bounding box level within actual test set images using the lightweight and hardware-optimized CoreML weights.

<div align="center">
  <img src="visualization/CoreML_Detection_Best_Examples_with_Bus_Truck_and_Car.png" width="100%" alt="CoreML Detection Result Analysis">
</div>

---

## 7. References & Data Sources
* **Training Dataset (YOLOv8):** [Roboflow Universe - Vehicles (roboflow-100)](https://universe.roboflow.com/roboflow-100/vehicles-q0x2v)
  * The original dataset utilized for training the multi-vehicle detection model.
* **Real-world CCTV Inference Video:** [Korea Expressway Corporation RoadPlus](https://www.roadplus.co.kr/main/main.do)
  * Real-time CCTV footage utilized to test the model's performance in a real-world deployment scenario.
  * **Demo Location:** Gyeongbu Expressway, Korea Expressway Corporation Suwon Branch jurisdiction (Won-gok Sleep Rest Area, near Anseong JC, Seoul/Busan direction).

