# deepassimilate

**deepassimilate** is a modular framework for training **unconditional diffusion models** and applying them to **generative data assimilation (DA)**.  
It is built on **PyTorch** and **Hugging Face Diffusers**, and is designed for scientific machine learning, inverse problems, physical sciences, weather/climate modeling, and uncertainty quantification workflows.

The library follows a simple two-step paradigm:

1. **Train an unconditional diffusion model**  
2. **Perform generative data assimilation** using the trained model + custom observation operators

---

## ✨ Features

### **1. Unconditional Diffusion Training**

Supports:
- **EDM-style UNet2D** architectures
- **Basic UNet** baselines
- **Custom user-defined architectures**

Fully compatible with diffusers schedulers:
- **HeunDiscrete (EDM)**
- **DDPM**
- **DDIM**

Supports:
- Configurable model hyperparameters
- Flexible training timesteps
- Optional post-training quantization

---

### **2. Generative Data Assimilation**

Implements gradient-based posterior correction:

\[
x_{t}' = x_t + \eta\,\sigma(t)\,\nabla_x \log p(y \mid H(x_t))
\]

Where:
- \( y \) = observations
- \( H \) = observation operator
- \( \eta \) = DA step size
- \( \sigma(t) \) = noise level

Compatible with any scheduler from diffusers

Supports:
- Arbitrary observation operators (masking, subsampling, projections, sensors)
- Multiple posterior samples per observation
- DA correction at user-defined frequencies
- Physics-agnostic and physics-informed workflows

---

### **3. Scheduler-Only Distillation**

- No student network needed
- Only modifies the scheduler used during inference
- Converts long training schedule (e.g., 1000 steps) into fast inference (e.g., 20–50 steps)
- Works with any scheduler, including Heun/EDM

---

## 📦 Installation

```bash
pip install -e .
```

Dependencies (if not already installed):

```bash
pip install torch diffusers
```

---

## 🚀 Quickstart

### **Step 1 — Train an unconditional diffusion model**

```python
import torch
import deepassimilate as da

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

dataset = MyDataset(torch.randn(1000, 1, 64, 64))

model, scheduler, distilled_steps = da.train_unconditional(
    dataset=dataset,
    architecture="edm_unet_2d",
    scheduler="heun_edm",
    img_size=64,
    channels=1,
    num_epochs=5,
    distilled_num_steps=40
)
```

### **Step 2 — Run generative data assimilation**

```python
import torch
from deepassimilate import IdentityObservationOperator, DAConfig

obs = torch.randn(8, 1, 64, 64)

analysis_samples = da.run_data_assimilation(
    model=model,
    scheduler=scheduler,
    observations=obs,
    obs_operator=IdentityObservationOperator(),
    da_cfg=DAConfig(da_steps=50, da_strength=0.1),
    distilled_num_steps=40,
)

print(analysis_samples.shape)
# (24, 1, 64, 64)
```

---

## 🧱 Project Structure

```
deepassimilate/
  __init__.py
  models/
    factory.py
  schedulers/
    factory.py
    distilled.py
  training/
    uncond_trainer.py
    quantization.py
  assimilation/
    observation_ops.py
    da_posterior.py
    pipeline.py
```

---

## 🔧 Customization Guide

### **Custom UNet Architecture**

```python
def my_unet(img_size, in_channels, out_channels):
    return MyCustomUNet(...)

model, scheduler, distilled_steps = da.train_unconditional(
    dataset=dataset,
    architecture="custom",
    custom_builder=my_unet
)
```

### **Custom Observation Operator**

```python
from deepassimilate import ObservationOperator

class MaskOperator(ObservationOperator):
    def __init__(self, mask):
        self.mask = mask
    def forward(self, x):
        return x * self.mask
```

### **Custom Noise Schedules (EDM / Karras / Log-Sigma)**

Modify:
- `deepassimilate/schedulers/distilled.py`

to implement:
- EDM/Karras sigma sampling
- Log-spaced sigmas
- Custom timestep interpolations

### **Custom Likelihood or Physics Constraints**

Modify:
- `deepassimilate/assimilation/da_posterior.py`

to implement:
- PDE residual corrections
- Physical conservation terms
- Spatial correlation models
- Multi-sensor likelihoods
- Hybrid EnKF-diffusion gradients

---

## 🔮 Coming Next (Roadmap & Extensions)

The following are planned enhancements and open research directions. Contributions are welcome!

### **1. EDM μ(t) and σ(t) Formulations**

Currently μ(t) and σ(t) are placeholders.  
Future work includes:
- Implementing true EDM log-σ sampling
- Proper forward and reverse SDE parameterization
- Compatibility with consistency models
- Support for VE-SDE, VP-SDE, and Sub-VP SDE families

### **2. Timestep Distillation Improvements**

Add utilities for:
- Karras ρ-schedules
- Curvature-aware timestep compression
- Progressive inference distillation
- Targeted refinement around observation times

### **3. Full SDE Family Support**

Allow the user to choose:
- VE-SDE (EDM)
- VP-SDE (DDPM)
- Sub-VP SDE
- Deterministic ODE score-based models
- Flow-matching / rectified flows

Each with consistent:
- Drift term
- Diffusion amplitude
- Integrator
- Sampling timesteps

### **4. Physics-Informed Generative DA**

Integrate more scientific ML features:
- PDE residual penalties
- Score regularizers enforcing conservation laws
- Domain-aware priors (energy, vorticity, mass, etc.)
- Coupling with data-driven or operator-learned dynamics models
- Time-dependent observation operators \( H(t) \)

### **5. Spatiotemporal Diffusion Models**

Add support for:
- 3D fields (H × W × T)
- Diffusion Transformers (DiT)
- Temporal UNet variants
- Autoregressive or blockwise temporal sampling
- Diffusion-based ensemble forecast generation

### **6. Advanced Quantization / Compression**

Support for:
- 8-bit and 4-bit quantization
- QAT (Quantization-Aware Training)
- Pruning + LoRA hybrid lightweight models
- ONNX export and TensorRT compatibility

### **7. Example Notebooks (Planned)**

Notebooks demonstrating:
- Training diffusion priors on weather/ocean datasets
- Running DA on masked observations
- Inspecting DA gradients and likelihood contributions
- Implementing custom timesteps
- Using physical constraints in DA
- Building high-resolution ground truth datasets (soil moisture, wind gust, preciptiation, temperature)
- Update weights of models like StormCast/Cordiff using GDA
- 
---

## 📝 License

GNU General Public License 
