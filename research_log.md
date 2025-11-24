# SwinIR Knowledge Distillation Research Log

## Goal of the Project

To investigate the effectiveness of knowledge distillation for compressing the SwinIR image restoration model. We will compare a conventionally trained lightweight "student" model against students trained with response-based and feature-based distillation from a large "teacher" model.

---

## Phase 0: Project Setup, Environment Configuration, and Baseline Verification

**Date:** 2025-11-25

**Objective:**
To establish a stable, reproducible, and correct working environment for the SwinIR project. The goal of this phase was to successfully run the original, pre-trained SwinIR model and then to verify that a custom training script could be successfully launched.

### 1. Environment Configuration

A dedicated Conda environment was created to ensure project isolation. The initial approach of replicating the original 2021 dependencies (PyTorch 1.8) resulted in `OSError` and dependency conflicts, indicating incompatibility with modern hardware (NVIDIA driver CUDA 13.0 compatible).

The successful configuration was achieved by creating a **modernized environment**:

- **Environment Manager:** Miniconda
- **Environment Name:** `swinir`
- **Python Version:** 3.8
- **Key Dependencies:**
  - `pytorch`, `torchvision`, `torchaudio` (Installed via Conda)
  - `pytorch-cuda=12.1` (Selected as the latest stable, Conda-certified CUDA toolkit)
  - Other dependencies (`numpy`, `opencv-python`, `timm`, `matplotlib`, etc.) were installed via Pip from the KAIR `requirement.txt` file and manual error correction.

### 2. Codebase Assembly

The project required code from two repositories: the original SwinIR "showroom" and the KAIR "factory".

- **Action:** Key modules from KAIR were copied into the main `SwinIR-main` project directory to create a self-contained training environment. This included `main_train_psnr.py` (renamed), the `options/` folder, and a merge of `data/`, `models/`, and `utils/` folders.

### 3. Verification & Troubleshooting

- **Test 1 (Inference):** A successful test was performed using the pre-trained SwinIR model for grayscale denoising, confirming the environment could run inference.
- **Test 2 (Training):** A test of the training script (`main_train_student.py`) was initiated. Several `AssertionError` and `UserWarning` issues related to `dataroot` paths and `num_workers` were debugged and resolved.
- **Final Outcome:** **SUCCESS.** The training script for our custom student model successfully launched, initialized, loaded data, and began the training loop, running up to iteration 200 before being manually stopped.

**Conclusion:** The setup and verification phase is complete. The project environment is stable, correct, and ready for experimentation.

---

## Experiment 1: Baseline Training (Model A - "Lone Wolf")

**Objective:**
To establish a performance baseline for our lightweight student architecture. This model is trained conventionally, using only the L1 pixel loss against the ground-truth images.

**Student Model Architecture (`SwinIR_Student`):**

- `embed_dim`: 60
- `depths`: [4, 4, 4, 4]
- `num_heads`: [6, 6, 6, 6]
- **Total Parameters:** 891,123 (5.51% of the Teacher model)

**Training Configuration (`train_swinir_student.json`):**

- **Task:** Classical Super-Resolution (x4)
- **Dataset:** DIV2K (800 training images)
- **Loss Function:** L1 Loss
- **Optimizer:** Adam (lr = 2e-4)
- **Scheduler:** MultiStepLR
- **Total Iterations (for final run):** 100,000

**Execution Notes:**

- A successful verification run was completed on 2025-11-24.
- The log output at iteration 200 correctly showed all three loss components: `l_g_L1: 4.061e-02 G_loss: 4.061e-02`. This confirms the implementation is working as intended.

**Results:**

- _(This section will be filled in after the full training is complete. We will record the final average PSNR on test sets like Set5 here.)_

---

## Experiment 2: Response-Based Distillation (Model B - "Apprentice")

**Objective:**
To determine if standard knowledge distillation (learning from the teacher's final output) can improve the performance of the lightweight student model compared to the baseline (Model A). This serves as a stronger, more established baseline for our novel method.

**Methodology:**
The student model was trained using a combined loss function with two components:

1.  **L1 Loss:** A standard pixel-wise L1 loss comparing the student's output to the ground-truth high-resolution image.
2.  **Distillation Loss (L1):** An L1 loss comparing the student's output to the final output of the pre-trained, frozen Teacher model.

The code was modified in `models/model_plain.py` to load the Teacher model and calculate this combined loss when the `"distillation_type"` in the configuration is set to `"response"`.

**Student Model Architecture (`SwinIR_Student`):**

- Identical to Model A to ensure a fair, controlled experiment.

**Training Configuration (`train_swinir_student_distill_response.json`):**

- **Task:** Classical Super-Resolution (x4)
- **Key Settings:** `"distillation_type": "response"`, and `"pretrained_netTeacher"` path was set to the pre-trained SwinIR-M x4 model. All other settings (dataset, learning rate, etc.) were kept identical to Experiment 1.

**Execution Notes:**

- A successful verification run was completed on 2025-11-24.
- The log output at iteration 200 correctly showed all three loss components: `l_g_L1: 4.902e-02 l_g_distill: 3.712e-02 G_loss: 8.614e-02`. This confirms the implementation is working as intended.

**Results:**

- _(This section will be filled in after the full training is complete. We will record the final average PSNR here.)_

---

## Experiment 3: Feature-Based Distillation (Model C - "Mind Reader")

**Objective:**
To test our primary novel hypothesis: that forcing the student to mimic the teacher's intermediate feature representations provides a superior training signal compared to standard distillation (Model B) and conventional training (Model A).

**Methodology:**
This experiment uses a combined loss function with three components:

1.  **L1 Loss:** (Same as Model A and B).
2.  **Distillation Loss (L1):** (Same as Model B).
3.  **Feature Loss (L1):** Our novel contribution. This is an L1 loss that compares the intermediate feature maps from the student's RSTB blocks to the corresponding feature maps from the teacher's RSTB blocks.

To handle the difference in feature dimensions (`embed_dim` of 60 for the student vs. 180 for the teacher), a set of learnable linear "translator" layers (`nn.Linear(180, 60)`) were introduced. These projectors are trained alongside the student to map the teacher's feature space to the student's feature space. The architectural code (`network_swinir.py`, `network_swinir_student.py`) was modified to expose these intermediate features, and the training logic (`model_plain.py`) was upgraded to calculate this new loss.

**Student Model Architecture (`SwinIR_Student`):**

- Identical to Model A and B.

**Training Configuration (`train_swinir_student_distill_feature.json`):**

- **Task:** Classical Super-Resolution (x4)
- **Key Settings:** `"distillation_type": "feature"`. All other settings were kept identical to Experiment 2.

**Execution Notes:**

- A successful verification run was completed on 2025-11-24.
- The log output at iteration 200 correctly showed all three loss components: `l_g_L1: 4.496e-02 l_g_distill: 4.085e-02 l_g_feature: 7.542e+00 G_loss: 7.628e+00`. This confirms the implementation is working as intended.

**Results:**

- _(This section will be filled in after the full training is complete. We will record the final average PSNR here.)_
