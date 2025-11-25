# Project Command Reference

This document contains a complete list of the useful commands for running experiments, testing models, and managing the project environment.

---

## 1. Environment Management

These commands are for setting up and inspecting the Conda environment.

### Create the Environment

Used by a new team member to build the project environment for the first time.

```bash
conda env create -f environment.yml
```

### Activate the Environment

Run this in the terminal every time you start a new work session.

```bash
conda activate swinir
```

### Inspect Installed Packages

Use this to see a complete list of all packages installed in the current environment.

```bash
conda list
```

---

## 2. Training Experiments

These commands are used to launch the training runs for our three experimental models.

The current training budget is set to **20,000 iterations**. Each run will save a model checkpoint (`.pth` file) and run a test every **2,000 iterations**. The training will automatically resume from the latest checkpoint if stopped and restarted.

### Model A: Baseline Training ("Lone Wolf")

Trains the student model using only the standard L1 loss.

```bash
python main_train_student.py --opt options/swinir/train_swinir_student.json
```

### Model B: Response-Based Distillation ("Apprentice")

Trains the student using L1 loss and guidance from the Teacher's final output.

```bash
python main_train_student.py --opt options/swinir/train_swinir_student_distill_response.json
```

### Model C: Feature-Based Distillation ("Mind Reader")

Trains the student using L1 loss, response distillation, and our novel feature-based loss.

```bash
python main_train_student.py --opt options/swinir/train_swinir_student_distill_feature.json
```

---

## 3. Testing Trained Models

These commands are used to evaluate the performance of a trained student "brain" (`.pth` file) on a test dataset.

The main script for this is our custom `main_test_student.py`, which is designed to work with our lightweight student architecture.

### How to Use

To run a test, you must replace the placeholder in the `--model_path` argument with the actual path to the `.pth` file you want to evaluate.

**Example Command Structure:**

```bash
python main_test_student.py --task classical_sr --scale 4 --model_path [PATH_TO_YOUR_MODEL_BRAIN] --folder_lq testsets/Set5/LR_bicubic/X4 --folder_gt testsets/Set5/HR
```

### Example for a Trained Model C

This example shows how to test the `20000_G.pth` brain from our Model C experiment.

```bash
python main_test_student.py --task classical_sr --scale 4 --model_path superresolution/student_C_distill_feature_x4/models/20000_G.pth --folder_lq testsets/Set5/LR_bicubic/X4 --folder_gt testsets/Set5/HR
```

You can adapt this command to test any brain from any of our experiments (Model A, B, or C) by simply changing the model path.
