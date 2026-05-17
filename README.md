# 🎬 Video Scene Classification Engine

**Course:** Deep Learning (CSTC - 308)  
**Team Members:** Akansha Patel (123110019), Nanshi (123110022), Sanchi (123110032), Sneha (123110054)

---

## 📌 Project Overview
This project implements a robust **Video Scene Classification Engine** evaluating two distinct methodologies: a baseline spatial network using **GoogLeNet Transfer Learning** with frame-based temporal aggregation, and an extended **Hybrid CNN-LSTM network** designed to explicitly capture temporal dependencies.

The core objective was to build and contrast deep learning pipelines that demonstrate:
*   **Transfer Learning Efficiency:** GoogLeNet pretrained on ImageNet.
*   **Video-to-Image Decomposition:** Uniform frame extraction pipelines.
*   **Class Imbalance Mitigation:** Weighted sampling across heterogeneous frame distributions.
*   **Real-world Robustness:** Performance evaluation across diverse, challenging scene categories.

---

## 🎞️ Dataset Architecture
The dataset is constructed in a hierarchical video-class format spanning four domains:
*   **Sports Videos:** High motion, dynamic camera transitions.
*   **News Videos:** Static shots, human presenters, studio lighting.
*   **Movies:** Mixed lighting, complex scene transitions, high cinematic variance.
*   **Violence Videos:** High motion intensity, abrupt visual changes, erratic action semantics.

### Frame Extraction Strategy
Videos are decomposed into frames using uniform sampling:
$$\text{Frame Extraction Rate} = 1 \text{ frame every } N \text{ frames (Configurable Skip Strategy)}$$

---

## ⚙️ Pipeline & Feature Strategy

### 1. Video Decomposition Layer
*   Videos are parsed into discrete frame sequences using OpenCV.
*   Temporal redundancy is reduced via uniform frame-skipping heuristics.

### 2. Spatial Feature Learning (Baseline Backbone)
*   **Architecture:** GoogLeNet (Inception v1) initialized with ImageNet weights.
*   **Input Resolution:** $224 \times 224 \times 3$ (RGB).
*   **Baseline Configuration:** Feature extractor frozen; only the final classification head is updated.

### 3. Class Imbalance Mitigation
*   Weighted Random Sampling is applied during training batches to ensure equal category selection probability and prevent dominant class bias.

### 4. Temporal Aggregation Layers (Dual Approaches)
*   **Approach A (Baseline):** Frame-wise independent classification followed by a post-hoc **Majority Voting** consensus layer.
*   **Approach B (Extended Experiment):** End-to-end sequential processing using a multi-layer **Long Short-Term Memory (LSTM)** network operating on temporal frame queues.

---

## 📊 Performance Evaluation: Baseline Model

### Classification Report (GoogLeNet + Majority Voting)

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Movies** | 1.00 | 1.00 | 1.00 | 215 |
| **News** | 1.00 | 1.00 | 1.00 | 567 |
| **Sports** | 0.93 | 0.96 | 0.95 | 1208 |
| **Violence** | 0.92 | 0.87 | 0.90 | 643 |

*   **Overall Accuracy:** 0.95 (95%)
*   **Macro Average F1-Score:** 0.96
*   **Weighted Average F1-Score:** 0.95
*   **Total Test Samples:** 2633 frames/sequences

### Confusion Matrix Analysis

| Actual \ Predicted | Movies | News | Sports | Violence |
| :--- | :---: | :---: | :---: | :---: |
| **Movies** | **215** | 0 | 0 | 0 |
| **News** | 0 | **567** | 0 | 0 |
| **Sports** | 0 | 0 | **1162** | 46 |
| **Violence** | 0 | 0 | 82 | **561** |

#### Baseline Key Insights:
1.  **Near-Perfect Separation for Structured Classes:** *Movies* and *News* achieved ideal metrics ($F1 = 1.00$) due to stable visual structures, uniform lighting conditions, and minimal intra-class visual variance.
2.  **Kinematic Overlap & Ambiguity:** The primary failure modes occurred between **Sports $\rightarrow$ Violence (46)** and **Violence $\rightarrow$ Sports (82)**. Because the baseline treats frames independently, it relies purely on static spatial representations and fails to decouple high-motion action signatures from violent contexts.

---

## 🧪 Extended Experiment: LSTM-Based Temporal Fine-Tuning

To evaluate the impact of explicit sequence modeling, an extended architecture was introduced incorporating an LSTM layer coupled with **partial backbone fine-tuning**.

[16 Sampled Frames] ──> [GoogLeNet (Selective Fine-Tuning)] ──> [Spatial Embeddings] ──> [Multi-layer LSTM] ──> [Softmax Classification]


### Setup & Implementation Details
*   **Sequence Depth:** Each video is converted into a uniform sequence of $16$ chronological frames.
*   **Partial Fine-Tuning:** Rather than leaving the backbone entirely frozen, selective higher-level convolutional blocks (**Inception Mixed Layers**) of GoogLeNet were unfrozen to allow adaptation to domain-specific features.
*   **Sequence Mapping:** The resulting spatial embeddings are passed sequentially into a multi-layer LSTM network to capture cross-frame motion dynamics before final softmax classification.

### Empirical Results & Architectural Analysis
The LSTM-enhanced variant achieved a lower validation accuracy of approximately **72.45%**. 

#### Critical Diagnosis of the Convergence Trade-Off:
*   **Optimization Instability:** Unfreezing the higher-level Inception layers introduced significant training instability during early epochs. 
*   **Domain Heterogeneity:** Because the dataset features starkly different visual profiles—ranging from completely static newsroom settings to chaotic, high-motion action sequences—partial fine-tuning caused a temporary degradation in feature representation consistency.
*   **Data Scarcity vs. Parameter Capacity:** The increased flexibility and parameter count of the CNN-LSTM pipeline demanded a larger data scale and more rigorous hyperparameter optimization (e.g., adaptive learning rate schedules, dropout tuning) to completely prevent gradient misalignment.

---

## 🧮 Architectural Comparison Matrix

| Architecture | Paradigm | Advantages | Disadvantages | Validation Accuracy |
| :--- | :--- | :--- | :--- | :---: |
| **GoogLeNet + Majority Vote** | Frozen Spatial Features + Heuristic Voting | Highly stable convergence; computationally lightweight; preserves generalized ImageNet weights. | Lacks true sequence modeling; vulnerable to frame-level motion ambiguity. | **~95.00%** |
| **GoogLeNet + LSTM** | Sequential Deep Learning + Partial Fine-Tuning | Captures true temporal correlations; adaptable to dataset-specific visual styles. | Prone to early training instability; requires complex tuning and larger datasets. | **~72.45%** |

---

## ⚠️ System Limitations
*   **Temporal Indeterminacy (Baseline):** Relies on isolated frame heuristics, resulting in semantic confusion when high-speed motions overlap.
*   **Convergence Sensitivity (Hybrid):** Fine-tuning sequential models on heterogeneous video domains yields high optimization volatility.
*   **Sampling Redundancy:** Fixed-interval sampling can skip brief, high-impact scene-changing frames or introduce redundant consecutive data.

---

## 🔮 Future Architecture Roadmap
*   **Advanced Optimization:** Introduce progressive unfreezing schedules and Layer-wise Adaptive Rate Scaling (LARS) to stabilize the LSTM pipeline.
*   **Native 3D Convolutions:** Implement C3D or I3D architectures to jointly learn spatial and temporal features within early layers.
*   **Spatiotemporal Transformers:** Transition from recurrent loops to Video Transformers (e.g., **TimeSformer** or **ViViT**) using divided space-time attention.
*   **Dynamic Frame Gating:** Replace uniform extraction with entropy-based keyframe selection.

---

## 🚀 Inference Pipeline

### Execution Flow:
[Input Video (.mp4/.avi)] ──> [Uniform Frame Parser] ──> [Feature Extraction / LSTM State] ──> [Classification Header] ──> [Output Class Label]


### Supported Outputs:
*   `MOVIES` | `NEWS` | `SPORTS` | `VIOLENCE`

---

## 📦 Model Artifacts
*   **Baseline Weights:** `scene_model_baseline.pth`

---

## 🔗 Future Work

- Replace CNN with Video Transformers  
- Add real-time webcam classification  
- Build interactive UI dashboard  
- Improve sampling strategy dynamically  
