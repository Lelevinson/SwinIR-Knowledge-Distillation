# SwinIR Knowledge Distillation Project

This repository contains the code for a research project exploring knowledge distillation techniques to create a lightweight, efficient version of the SwinIR model for image restoration. The goal is to train a small "student" model that learns from a large, pre-trained "teacher" model, and to investigate novel feature-based distillation methods.

## Original Project

This work is built upon the official SwinIR and KAIR repositories. All credit for the original model architecture and training framework goes to the original authors.

- **Original SwinIR Project:** [https://github.com/JingyunLiang/SwinIR](https://github.com/JingyunLiang/SwinIR)
- **Original KAIR Training Framework:** [https://github.com/cszn/KAIR](https://github.com/cszn/KAIR)

---

## 1. Local Setup Instructions

This project requires several external dependencies that are not tracked by Git. Follow these steps to set up a complete and correct local environment.

### Step 1: Clone the Repository

Clone this repository to your local machine:

```bash
git clone https://github.com/Lelevinson/SwinIR-Knowledge-Distillation.git
cd SwinIR-Knowledge-Distillation
```

### Step 2: Create the Conda Environment

This project uses a specific Conda environment. The required packages are listed in `environment.yml`. Create and activate the environment with the following commands:

```bash
# Create the environment from the file
conda env create -f environment.yml

# Activate the environment
conda activate swinir
```

### Step 3: Download and Organize Datasets

The training datasets are not included in this repository and must be downloaded manually. All datasets should be placed inside a `trainsets/` folder in the main project directory.

#### A) Super-Resolution Datasets

Our primary experiments use the **DIV2K** dataset.

1.  **Download:** [DIV2K Dataset](https://cv.snu.ac.kr/research/EDSR/DIV2K.tar).
2.  **Place:** After extracting, ensure the high-resolution images are located at the following path, as our configuration files depend on it:
    - `trainsets/DIV2K/DIV2K_train_HR/`

The **Flickr2K** dataset is also used by the original Teacher model.

1.  **Download:** [Flickr2K Dataset](https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar).
2.  **Place:** After extracting, the images should be located in their own folder, for example:
    - `trainsets/Flickr2K/Flickr2K_HR/`

#### B) Denoising & Other Datasets

Future experiments will use the following datasets. Please download and extract them into their own subfolders within `trainsets/`.

- **BSD500:**

  - **Download:** [BSD500 Dataset](http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz).
  - **Suggested Path:** `trainsets/BSD500/images/`

- **WED (Waterloo Exploration Database):**

  - **Download:** [WED Dataset](http://ivc.uwaterloo.ca/database/WaterlooExploration/exploration_database_and_code.rar).
  - **Suggested Path:** `trainsets/WED/`

- **OST (Outdoor Scene Training):**
  - **Download:** [OST Dataset](https://openmmlab.oss-cn-hangzhou.aliyuncs.com/datasets/OST_dataset.zip).
  - **Suggested Path:** `trainsets/OST/`

When we create the configuration files (`.json`) for these new tasks, we will simply update the `"dataroot_H"` path to point to these specific locations.

### Step 4: Download Pre-trained Models

The pre-trained models (the "brains") are not included in this repository due to their large size. We have prepared a separate guide with download scripts for all official SwinIR models.

- Our distillation experiments require a specific **Teacher Model**. Please ensure you have downloaded at least this model: `001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth`.
- For instructions on how to download this model and all others, please see the guide here: **[DOWNLOAD_MODELS.md](download_models.md)**.

---

## 2. Project Structure

This project is a self-contained training and testing environment. Here is a guide to the key files and folders:

- **`main_train_student.py`**: The main script for running all training experiments (Models A, B, and C).
- **`main_test_swinir.py`**: The original script for testing pre-trained models or models you have trained.
- **`options/swinir/`**: Contains the JSON "curriculum" files that configure each experiment. This is where you control the model architecture, learning rates, and our custom distillation methods.
- **`models/model_plain.py`**: The core "Professor" logic. Contains our custom code for calculating the standard, response-based, and feature-based distillation losses.
- **`models/network_swinir.py`**: The blueprint for the original "Teacher" model.
- **`models/network_swinir_student.py`**: Our dedicated, modifiable blueprint for the "Student" model, where our architectural changes are made.
- **`research_log.md`**: Our official lab notebook. All experimental plans and results should be documented here.
- **`DOWNLOAD_MODELS.md`**: A guide with PowerShell commands to download all official pre-trained SwinIR models.

---

### Ignored Folders (Not on GitHub)

The following folders are listed in the `.gitignore` file and will not be uploaded to the repository. They are either too large or specific to a local machine.

- **`trainsets/`**: This is where you should place the large training datasets (e.g., DIV2K).
- **`model_zoo/`**: This is where pre-trained "brains" (`.pth` files) for the teacher models are stored.
- **`superresolution/`** & **`results/`**: These folders are automatically created by the training and testing scripts to store their output (logs, saved student model brains, and generated images).

---

## 3. How to Run Experiments

All experiments are controlled by the JSON files in `options/swinir/`. Use the following commands to run a short verification test (run to `iter: 200` and stop with `Ctrl + C`).

To run a full training, let the script run until it reaches the desired number of iterations (e.g., 100,000) and then stop it.

### Model A (Baseline "Lone Wolf")

```bash
python main_train_student.py --opt options/swinir/train_swinir_student.json
```

### Model B (Response-Based Distillation "Apprentice")

```bash
python main_train_student.py --opt options/swir/train_swinir_student_distill_response.json
```

### Model C (Feature-Based Distillation "Mind Reader")

```bash
python main_train_student.py --opt options/swinir/train_swinir_student_distill_feature.json
```

---

## 4. Contribution Workflow (Git)

To collaborate effectively, please follow this standard Git workflow. This ensures our work stays in sync and we avoid conflicts.

1.  **Update:** Before starting any new work, always get the latest changes from the team. This downloads any new commits from GitHub.

    ```bash
    git pull
    ```

2.  **Work:** Make your code changes, run your experiments, update the research log, etc.

3.  **Save:** When you have completed a meaningful task, save a snapshot of your work locally. This is a two-step process.

    ```bash
    # Step 1: Stage your changed files
    git add .

    # Step 2: Commit them with a clear, descriptive message
    git commit -m "Your clear and descriptive message here"
    ```

4.  **Share:** Upload your saved work to GitHub for the rest of the team to see.
    ```bash
    git push
    ```

If you are working on a major new feature or a risky experiment, it is good practice to create a new branch to work on, rather than committing directly to `main`.
