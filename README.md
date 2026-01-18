# Holo-Code

This repository contains the official implementation code for **Holo-Code**.

## 📂 Project Structure

- `run_holo_official.sh`: The main script to run Holo-Code watermarking.
- `run_holo_vs_gssync.sh`: Script for comparing Holo-Code with Gaussian Shading (GS) and other baselines.
- `watermark.py`: Core watermarking logic.
- `run_gaussian_shading.py`: Base diffusion generation script.
- `prc_core/`: Modules related to PRC baseline methods.

## 🚀 Getting Started

### Prerequisites

```bash
pip install -r requirements.txt
```

### Usage

To generate watermarked images using Holo-Code:

```bash
bash run_holo_official.sh
```

To run comparisons (Holo vs GS):

```bash
bash run_holo_vs_gssync.sh
```

## 📄 Output

- Generated images will be saved in the `output/` directory.
- Quantitative results (FID, Bit Accuracy) can be found in the logs.

## 🙏 Acknowledgements

This codebase relies on [Gaussian Shading](https://github.com/bsmhmmlf/Gaussian-Shading) and Stable Diffusion.
