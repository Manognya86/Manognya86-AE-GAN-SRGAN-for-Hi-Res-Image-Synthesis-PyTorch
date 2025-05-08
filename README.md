# Auto-Embedding Generative Adversarial Network (AEGAN) for High-Resolution Image Synthesis

![System Architecture](figures/fig5.1_architecture.png)  
*High-Level System Architecture (Refer to [Documentation](#documentation))*

[![GitHub](https://img.shields.io/badge/GitHub-Repo-blue)](https://github.com/Manognya86/Manognya86-AE-GAN-SRGAN-for-Hi-Res-Image-Synthesis-PyTorch.git)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.13.1%2B-orange)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

A deep learning system that transforms low-resolution images into high-quality outputs (up to 512x512) using a two-stage GAN architecture. Developed by **Team R2D2** for the course **CIS 579 - Artificial Intelligence**.

---

## ✨ Features
- **High-Resolution Synthesis**: Upscale images from 64x64 → 512x512 using a refiner network.
- **Role-Based Access Control**:
  - **Admins**: Manage users, roles, and permissions.
  - **Data Scientists**: Trigger training and evaluate models.
  - **Users**: Upload images and retrieve enhanced outputs.
- **GAN Architecture**:
  - Two-stage training with Wasserstein loss + gradient penalty.
  - Automatic model versioning and checkpointing.
- **Deployment**:
  - Thread-safe REST API (Flask) with JWT authentication.
  - GPU-optimized inference using mixed precision.
- **Database**: SQLite storage for users, image records, and deployment logs.
- **Monitoring**: TensorBoard integration for loss/FID tracking.

---

## 🛠️ Technical Stack
| Component       | Technologies                                                                 |
|-----------------|------------------------------------------------------------------------------|
| **AI Core**     | PyTorch 1.13.1, CUDA 11.7, Two-stage GAN (`_netG1`, `_netG2`, `_RefinerG`)  |
| **Backend**     | Flask 2.2.3, Flask-JWT-Extended 4.4.4, SQLAlchemy 1.4.45                    |
| **Frontend**    | Tkinter (Admin Panel), HTML/CSS/JS (Web UI)                                 |
| **Training**    | TensorBoard, Multi-threaded DataLoader, Xavier weight initialization        |
| **Deployment**  | Docker-ready, Thread-safe singleton pattern, Prometheus-ready metrics       |

---

📥 Installation

### Prerequisites
- **Python 3.8+**  
- **NVIDIA GPU** (Recommended for training) with CUDA 11.7  
- **RAM**: 16GB+ (32GB for large datasets)  

### Steps
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Manognya86/Manognya86-AE-GAN-SRGAN-for-Hi-Res-Image-Synthesis-PyTorch.git
   cd Manognya86-AE-GAN-SRGAN-for-Hi-Res-Image-Synthesis-PyTorch
   ```

2. **Set Up Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   venv\Scripts\activate     # Windows
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements-lock.txt
   ```

4. **Initialize Database**:
   ```bash
   python database.py create
   ```

5. **Configure Environment**:
   Create a `.env` file:
   ```bash
   JWT_SECRET_KEY="your_secure_key_here"
   DATABASE_URI="sqlite:///aegan.db"
   ```

---

## 🖥️ Usage

### Start the Backend Server
```bash
python deploy.py
```
*Server runs at `http://localhost:5000`.*

### Interfaces
1. **Web UI**:
   - Access `http://localhost:5000` to upload images and view results.
   - Supported formats: JPEG, PNG, BMP.

2. **Admin Panel (Tkinter)**:
   ```bash
   python frontend.py
   ```
   - Manage users, roles, and permissions.

### Example: Image Enhancement via API
```bash
curl -X POST -H "Authorization: Bearer <JWT_TOKEN>" -F "image=@input.jpg" http://localhost:5000/api/enhance
```
*Returns JSON with `enhanced_image_path`.*

---

## 📡 API Documentation

| Endpoint              | Method | Description                          | Headers                          |
|-----------------------|--------|--------------------------------------|----------------------------------|
| `/api/enhance`        | POST   | Enhance an image                     | `Authorization: Bearer <JWT>`    |
| `/api/train`          | POST   | Start model training                 | Admin/Data Scientist role required |
| `/api/users`          | GET    | List all users                       | Admin role required              |
| `/api/login`          | POST   | Authenticate (returns JWT)           | `{"username": "...", "password": "..."}` |

---

## 🧠 Model Training

### Command
```bash
python train.py --epochs 100 --batch-size 32 --dataset-path ./data --lr 0.0002
```

### Key Arguments
- `--dataset-path`: Folder containing training images (subfolders for classes).
- `--lambda-adv`: Adversarial loss weight (default: `0.1`).
- `--lambda-l1`: L1 loss weight (default: `100`).

### Training Pipeline
1. **Stage 1**: Train `_netG2` and `_netRS` with L1 loss.
2. **Stage 2**: Adversarial training of `_netG1` and `_RefinerG`.
3. **Validation**: FID score thresholding for checkpointing.
---

## 🧪 Validation & Testing
- **Model Metrics**:
  - FID Score < 25 (target).
  - PSNR > 28 dB.
- **API Testing**:
  - Postman collection in `/tests/postman`.
  - Role-based access tests with `pytest`.
- **Security**:
  - OWASP ZAP scans for JWT/auth vulnerabilities.
  - Penetration tests for RBAC bypass.

---

## 🏗️ Core Models

### Generator Networks
| Model       | Purpose                          | Architecture                              |
|-------------|----------------------------------|-------------------------------------------|
| `_netG1`    | Stage 1 latent space generation | 4-layer ConvTranspose2d + BatchNorm + ReLU|
| `_netG2`    | Stage 2 image synthesis (64x64) | Skip connections for detail preservation  |
| `_RefinerG` | Upscale to 512x512              | 9-layer encoder-decoder with residuals    |

### Discriminator & Utilities
| Model       | Purpose                          | Architecture                              |
|-------------|----------------------------------|-------------------------------------------|
| `_netD1`    | Latent space discrimination      | Spectral normalization + LeakyReLU        |
| `_netRS`    | Real image encoder               | ResNet-18 backbone                        |


## 🚧 Challenges & Solutions

| Challenge                      | Root Cause                          | Solution                                  |
|--------------------------------|-------------------------------------|-------------------------------------------|
| **Training Instability**       | Mode collapse in Stage 2            | Wasserstein loss + gradient penalty       |
| **GPU Memory Overflow**        | Unoptimized RefinerG ops            | Mixed precision + input downsampling      |
| **Race Conditions**            | Global model instance               | Thread-safe singleton with `Lock()`       |
| **JWT Bypass**                 | Missing server-side role validation | Added RBAC decorators in `auth.py`        |
| **Data Loading Bottleneck**    | Sequential PIL loading              | Multi-threaded `DataLoader`               |

---


## 👥 Contributors
| Member                  | Role                                  |
|-------------------------|---------------------------------------|
| Vanshika Sangtani       | Frontend, Use Case Design             |
| Manognya Lokesh Reddy   | GAN Architecture, Training Pipeline   |
| Kavan Kumareshan        | Deployment, Security, API Design      |

---

## 📚 Documentation
- **Key Files**:
  - Model Architecture: `aegan_model.py`, `refiner_model.py`
  - Training: `train.py`, `dataloader.py`
  - API: `deploy.py`, `inference_api.py`
  - Database: `database.py`

---
## 📜 License
MIT License. See [LICENSE](LICENSE) for details.
```
