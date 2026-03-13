# 🎭 Pix2Pix Face Regeneration System

A deep learning project that reconstructs masked human faces using a **Conditional Generative Adversarial Network (cGAN)** — specifically the Pix2Pix architecture — paired with an interactive **Gradio** web interface for real-time inference.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Demo](#demo)
- [Project Structure](#project-structure)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Installation](#installation)
- [Training](#training)
- [Inference / Running the App](#inference--running-the-app)
- [Results](#results)
- [Tech Stack](#tech-stack)
- [Known Limitations](#known-limitations)
- [Ethical Use](#ethical-use)
- [License](#license)

---

## 🧠 Overview

This project tackles the task of **face inpainting / reconstruction** — given a masked or partially occluded face image, the model generates a realistic reconstruction of the complete face.

The system is built on the **Pix2Pix** framework (Image-to-Image Translation with Conditional Adversarial Networks). The Generator uses a **U-Net** encoder-decoder architecture with skip connections, and the Discriminator uses a **PatchGAN** design that evaluates realism at a patch level rather than the full image.

**Key highlights:**
- Trained for **45 epochs** on a custom paired dataset
- Generator uses **skip connections** to preserve spatial detail
- Combined **GAN + L1 loss** (λ=100) for sharp, realistic outputs
- One-click inference via **Gradio** web UI
- GPU-accelerated training (tested on RTX 3050 4GB)

---

## 🎬 Demo

```
Upload:  [ masked_face | real_face ]  (side-by-side combined image)
Output:  [ reconstructed face ]
```

The app automatically crops the left half (masked face) from your uploaded image and feeds it to the generator. No manual cropping needed.

To launch:
```bash
python app.py
```
Then open `http://127.0.0.1:7860` in your browser.

---

## 📁 Project Structure

```
pix2pix-face-regeneration/
│
├── app.py                      # Gradio inference app
├── pix2pix.ipynb               # Full training notebook
│
├── dataset/                    # Dataset folder (masked | real pairs)
│   ├── image_001.png
│   ├── image_002.png
│   └── ...
│
├── pix2pix_epoch_45.pth        # Trained generator weights (final)
├── pix2pix_epoch_40.pth        # Checkpoint (intermediate)
│
├── generated_output/           # Test output images (masked | fake | real)
│   └── result_0_0.png
│
├── training_log.txt            # Epoch-wise G_loss and D_loss log
│
└── README.md                   # You are here
```

---

## 🏗️ Architecture

### Generator — U-Net Encoder-Decoder

The Generator takes a **3-channel masked face image (128×128)** and outputs a **3-channel reconstructed face (128×128)**.

```
Input (3, 128, 128)
    │
    ▼
d1: Conv(3→64,   k=4, s=2) + LeakyReLU          → (64,  64, 64)
d2: Conv(64→128, k=4, s=2) + BN + LeakyReLU     → (128, 32, 32)
d3: Conv(128→256,k=4, s=2) + BN + LeakyReLU     → (256, 16, 16)
d4: Conv(256→512,k=4, s=2) + BN + LeakyReLU     → (512,  8,  8)
    │
    ▼  (bottleneck)
    │
u1: ConvT(512→256) + BN + ReLU  ──cat(d3)──►  (512, 16, 16)
u2: ConvT(512→128) + BN + ReLU  ──cat(d2)──►  (256, 32, 32)
u3: ConvT(256→64)  + BN + ReLU  ──cat(d1)──►  (128, 64, 64)
    │
    ▼
final: ConvT(128→3, k=4, s=2) + Tanh            → (3, 128, 128)
```

> Skip connections from encoder layers (d1–d3) are concatenated into the corresponding decoder layers, helping preserve fine spatial detail like facial edges and textures.

---

### Discriminator — PatchGAN

The Discriminator receives a **6-channel input** (masked image + real or fake image concatenated along channels) and outputs a **patch-level real/fake prediction**.

```
Input (6, 128, 128)  ← concat(masked, real_or_fake)
    │
Conv(6→64,   k=4, s=2) + LeakyReLU
Conv(64→128, k=4, s=2) + BN + LeakyReLU
Conv(128→256,k=4, s=2) + BN + LeakyReLU
Conv(256→1,  k=4, s=1)
    │
    ▼
Patch prediction map (real=1 / fake=0)
```

---

### Loss Functions

| Loss | Formula | Purpose |
|------|---------|---------|
| GAN Loss (G) | BCE(D(masked, fake), 1) | G tries to fool D |
| GAN Loss (D) | BCE(D(m,real),1) + BCE(D(m,fake),0) × 0.5 | D learns real vs fake |
| L1 Loss | λ × \|fake − real\|₁, λ=100 | Pixel-level accuracy |
| **Total G Loss** | **GAN + L1** | **Sharp & realistic output** |

---

## 🗂️ Dataset

The dataset consists of **side-by-side combined images** where:
- **Left half** → masked or occluded face
- **Right half** → corresponding ground truth face

```
┌──────────┬──────────┐
│  masked  │   real   │
│   face   │   face   │
└──────────┴──────────┘
     256 × 128 px
```

- Images are stored in `./dataset/`
- Total dataset split: **90% train / 10% test**
- Images are resized to `128×256` before splitting into `128×128` pairs

---

## ⚙️ Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended) or CPU
- pip

### Install Dependencies

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install gradio
pip install pillow numpy tqdm
```

Or if you have a `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Verify GPU

```python
import torch
print(torch.cuda.is_available())       # True
print(torch.cuda.get_device_name(0))   # e.g. NVIDIA GeForce RTX 3050
```

---

## 🏋️ Training

Open and run `pix2pix.ipynb` in Jupyter or VS Code.

### Key Training Settings

```python
data_path    = "./dataset"
batch_size   = 4
epochs       = 40          # base; extended to 45 via checkpoint resume
lambda_L1    = 100
image_height = 128
image_width  = 256
lr           = 0.0002
betas        = (0.5, 0.999)
```

### Resume from Checkpoint

The notebook supports resuming training from a saved `.pth` file:

```python
start_epoch  = 40
extra_epochs = 5
G.load_state_dict(torch.load("pix2pix_epoch_40.pth"))
```

### Training Log

Epoch-wise losses are saved to `training_log.txt`:

```
Time, Epoch, G_loss, D_loss
2026-03-12 23:24:58, 41, 4.4727, 0.6177
2026-03-12 23:33:41, 42, 4.5809, 0.5547
2026-03-12 23:37:39, 43, 4.6955, 0.5081
2026-03-12 23:42:06, 44, 5.8259, 0.2985
2026-03-12 23:47:09, 45, 6.5471, 0.2415
```

> **Note:** Rising G_loss with falling D_loss in later epochs suggests the discriminator became stronger — consider adding dropout or reducing D learning rate for future runs.

---

## 🚀 Inference / Running the App

```bash
python app.py
```

**What the app does, step by step:**

1. Loads `pix2pix_epoch_45.pth` into the Generator
2. You upload a combined image (masked | real) via the Gradio UI
3. The app automatically **crops the left half** (masked face)
4. Resizes to `128×128` and normalizes to `[-1, 1]`
5. Passes through the Generator (`torch.no_grad()`)
6. Denormalizes the output tensor and converts to `uint8`
7. Displays the reconstructed face in the output panel

**No GPU required at inference** — runs fine on CPU.

---

## 📊 Results

Test outputs are saved to `./generated_output/` as side-by-side comparisons:

```
[ masked | generated | real ]
```

Each file is named `result_{batch_index}_{image_index}.png`.

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Deep Learning Framework | PyTorch |
| Model Architecture | Pix2Pix (U-Net + PatchGAN) |
| Web Interface | Gradio |
| Image Processing | PIL, torchvision, NumPy |
| Training Hardware | NVIDIA RTX 3050 4GB (CUDA) |
| Notebook Environment | Jupyter / VS Code |
| Language | Python 3.12 |

---

## ⚠️ Known Limitations

- **128×128 output resolution** — low resolution may cause blurry fine details (hair, eyes). Upscaling to 256×256 would improve quality.
- **No dropout in Generator** — the original Pix2Pix paper uses dropout in the bottleneck for stochasticity. Adding it may improve diversity.
- **Discriminator dominance** — in later epochs (44–45), D_loss dropped very low (0.12, 0.10), indicating the discriminator may have overpowered the generator.
- **Single GPU, small batch** — batch size of 4 on a 4GB GPU limits training stability. Larger batch sizes on better hardware would help.
- **No data augmentation** — adding random flips or color jitter during training could improve generalization.

---

## 🔒 Ethical Use

This project is intended **strictly for educational and research purposes**.

- ❌ Do not use this system to generate faces of real people without their explicit consent.
- ❌ Do not use outputs for deepfakes, impersonation, or any deceptive purpose.
- ❌ Do not upload images of minors.
- ✅ Ensure all images used comply with the license of your dataset.
- ✅ If publishing results, disclose AI involvement transparently.

Misuse of face generation technology may violate privacy laws including **GDPR**, **India's DPDPA 2023**, **CCPA**, and **Illinois BIPA**.

---

## 📄 License

This project is released for **academic and research use only**.  
All generated outputs are synthetic and do not represent real individuals.  
Use responsibly.

---

*Built with PyTorch & Gradio | Trained on RTX 3050 | March 2026*
