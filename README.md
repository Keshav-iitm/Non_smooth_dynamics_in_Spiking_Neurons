# 📦 Non smooth dynamics in spiking neuron models.


# 🔗 [GitHub Repository] https://github.com/Keshav-iitm/Non_smooth_dynamics_in_Spiking_Neurons


This repository contains modular python codes for simulating non-smooth dynamics in spiking neurons using the QIF model. Includes v-t and v-a plots for tonic/burst firing, Poincaré maps, spiking freq vs coupling (k), and freq vs current (f-I) analysis. Visualizes firing patterns across varying input currents.


## 📁 Folder Structure & Dataset Placement

Non_smooth_dynamics_in_Spiking_Neurons/
├──README.md
├── AIF.py
├── QIF.py
├── firing_freq_I.py


## ⚙️ Environment Setup (Tested with CUDA GPU)

Use the following commands to create a reproducible environment:

```bash
conda create -n torch_gpu python=3.9
conda activate torch_gpu

pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install scikit-learn==1.0.2
pip install wandb==0.12.21
pip install numpy==1.21.6
pip install tqdm==4.62.3
pip install thop==0.0.31.post2005241907
pip install matplotlib==3.5.3
```

### ➕ Additional Imports

- Python standard: `os`, `argparse`, `sys`, `traceback`, `types`, `getpass`
- All included by default in Python ≥ 3.6


## 🚀 How to Run Scripts from Terminal

### ✅ Always run from `Non_smooth_dynamics_in_Spiking_Neurons/` using this format:

```bash
python <script_name.py> [--arguments]
```
---

## 🧠 Script Descriptions

### 1. `QIF.py` — Plotting QIF V vs t with flexibility in I_ext, V_th and V_reset

**Run**:
```bash
python QIF.py
```
**Argparse options**:
- `--I_ext`
- `--V_th`
- `--V_reset`
---

### 2. `AIF.py` — Plot for v vs t, phase plot v vs a, poincare maps and k vs spike count for tonic and burst firing.

**Run**:
```bash
python AIF.py --plot               #for v vs t, phase plot v vs a in tonic and burst firing
python AIF.py --spike_vs_k         #for v vs t, k vs spike count in tonic and burst firing
python AIF.py --poincare_tonic     #for v vs t, poincare map in tonic firing
python AIF.py --poincare_burst     #for v vs t, poincare map in burst firing

```
**Argparse options**:
- `--I_ext`
- `--V_th`
- `--V_reset`
- `--omega`

---

### 3. `firing_freq_I.py` — freq vs current (f-I) analysis

**Run**:
```bash
python firing_freq_I.py 
```

**Argparse options**:
- `--s`
- `--k`
- `--omega`
- `--v_th`
- `--v_R`
- `--t_max`
- `--dt`
---


## ✍️ Author 
>  *A B Keshav Kumar (AE24S021),MS Scholar, IIT Madras* 


## 💬 Need Help?

If any script fails due to import/module issues, check:
- Python version (3.9 recommended)
- CUDA 11.3 required for GPU support
- Dataset path structure
