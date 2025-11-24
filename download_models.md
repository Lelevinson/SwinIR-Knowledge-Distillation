# Download Scripts for Pre-trained SwinIR Models

This document provides the PowerShell commands to download all the official pre-trained SwinIR models used in the original paper.

The main training scripts in this project require a specific teacher model (`001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth`), but you may wish to download others for testing or for different distillation experiments.

## Instructions

1.  Make sure a `model_zoo/swinir/` directory exists in your project. If not, create it.
2.  Open a PowerShell terminal.
3.  Copy and paste the commands for the models you wish to download and run them.

---

### Classical Super-Resolution (SR)

```powershell
# Classical SR (Medium size, trained on DIV2K)
Invoke-WebRequest -Uri "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/001_classicalSR_DIV2K_s48w8_SwinIR-M_x2.pth" -OutFile "model_zoo\swinir\001_classicalSR_DIV2K_s48w8_SwinIR-M_x2.pth"
Invoke-WebRequest -Uri "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/001_classicalSR_DIV2K_s48w8_SwinIR-M_x3.pth" -OutFile "model_zoo\swinir\001_classicalSR_DIV2K_s48w8_SwinIR-M_x3.pth"
Invoke-WebRequest -Uri "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/001_classicalSR_DIV2K_s48w8_SwinIR-M_x4.pth" -OutFile "model_zoo\swinir\001_classicalSR_DIV2K_s48w8_SwinIR-M_x4.pth"

# Classical SR (Medium size, trained on DIV2K+Flickr2K) - Recommended Teacher
Invoke-WebRequest -Uri "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/001_classicalSR_DF2K_s64w8_SwinIR-M_x2.pth" -OutFile "model_zoo\swinir\001_classicalSR_DF2K_s64w8_SwinIR-M_x2.pth"
Invoke-WebRequest -Uri "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/001_classicalSR_DF2K_s64w8_SwinIR-M_x3.pth" -OutFile "model_zoo\swinir\001_classicalSR_DF2K_s64w8_SwinIR-M_x3.pth"
Invoke-WebRequest -Uri "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth" -OutFile "model_zoo\swinir\001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth"
```

### Lightweight SR (Small size, trained on DIV2K)

```powershell
Invoke-WebRequest -Uri "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x2.pth" -OutFile "model_zoo\swinir\002_lightweightSR_DIV2K_s64w8_SwinIR-S_x2.pth"
Invoke-WebRequest -Uri "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x3.pth" -OutFile "model_zoo\swinir\002_lightweightSR_DIV2K_s64w8_SwinIR-S_x3.pth"
Invoke-WebRequest -Uri "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x4.pth" -OutFile "model_zoo\swinir\002_lightweightSR_DIV2K_s64w8_SwinIR-S_x4.pth"
```

### Real-World Super-Resolution (SR)

```powershell
# Real-World SR (Medium and Large size)
Invoke-WebRequest -Uri "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth" -OutFile "model_zoo\swinir\003_realSR_BSRGAN_DFO_s64w8_SwinIR-M_x4_GAN.pth"
Invoke-WebRequest -Uri "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth" -OutFile "model_zoo\swinir\003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth"
```

### Grayscale Denoising (Medium size)

```powershell
Invoke-WebRequest -Uri "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/004_grayDN_DFWB_s128w8_SwinIR-M_noise15.pth" -OutFile "model_zoo\swinir\004_grayDN_DFWB_s128w8_SwinIR-M_noise15.pth"
Invoke-WebRequest -Uri "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/004_grayDN_DFWB_s128w8_SwinIR-M_noise25.pth" -OutFile "model_zoo\swinir\004_grayDN_DFWB_s128w8_SwinIR-M_noise25.pth"
Invoke-WebRequest -Uri "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/004_grayDN_DFWB_s128w8_SwinIR-M_noise50.pth" -OutFile "model_zoo\swinir\004_grayDN_DFWB_s128w8_SwinIR-M_noise50.pth"
```

### Color Denoising (Medium size)

```powershell
Invoke-WebRequest -Uri "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/005_colorDN_DFWB_s128w8_SwinIR-M_noise15.pth" -OutFile "model_zoo\swinir\005_colorDN_DFWB_s128w8_SwinIR-M_noise15.pth"
Invoke-WebRequest -Uri "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/005_colorDN_DFWB_s128w8_SwinIR-M_noise25.pth" -OutFile "model_zoo\swinir\005_colorDN_DFWB_s128w8_SwinIR-M_noise25.pth"
Invoke-WebRequest -Uri "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/005_colorDN_DFWB_s128w8_SwinIR-M_noise50.pth" -OutFile "model_zoo\swinir\005_colorDN_DFWB_s128w8_SwinIR-M_noise50.pth"
```

### Grayscale CAR (Medium size)

```powershell
Invoke-WebRequest -Uri "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/006_CAR_DFWB_s126w7_SwinIR-M_jpeg10.pth" -OutFile "model_zoo\swinir\006_CAR_DFWB_s126w7_SwinIR-M_jpeg10.pth"
Invoke-WebRequest -Uri "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/006_CAR_DFWB_s126w7_SwinIR-M_jpeg20.pth" -OutFile "model_zoo\swinir\006_CAR_DFWB_s126w7_SwinIR-M_jpeg20.pth"
Invoke-WebRequest -Uri "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/006_CAR_DFWB_s126w7_SwinIR-M_jpeg30.pth" -OutFile "model_zoo\swinir\006_CAR_DFWB_s126w7_SwinIR-M_jpeg30.pth"
Invoke-WebRequest -Uri "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/006_CAR_DFWB_s126w7_SwinIR-M_jpeg40.pth" -OutFile "model_zoo\swinir\006_CAR_DFWB_s126w7_SwinIR-M_jpeg40.pth"
```

### Color CAR (Medium size)

```powershell
Invoke-WebRequest -Uri "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/006_colorCAR_DFWB_s126w7_SwinIR-M_jpeg10.pth" -OutFile "model_zoo\swinir\006_colorCAR_DFWB_s126w7_SwinIR-M_jpeg10.pth"
Invoke-WebRequest -Uri "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/006_colorCAR_DFWB_s126w7_SwinIR-M_jpeg20.pth" -OutFile "model_zoo\swinir\006_colorCAR_DFWB_s126w7_SwinIR-M_jpeg20.pth"
Invoke-WebRequest -Uri "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/006_colorCAR_DFWB_s126w7_SwinIR-M_jpeg30.pth" -OutFile "model_zoo\swinir\006_colorCAR_DFWB_s126w7_SwinIR-M_jpeg30.pth"
Invoke-WebRequest -Uri "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/006_colorCAR_DFWB_s126w7_SwinIR-M_jpeg40.pth" -OutFile "model_zoo\swinir\006_colorCAR_DFWB_s126w7_SwinIR-M_jpeg40.pth"
```
