# A Variational Method for Reconstructing and Separating Balanced Motions and Internal Tides from Wide-Swath Altimetric Sea Surface Height Observations

This repository contains the code used in:

**Bellemin-Laponnaz, V., et al. (2025)**  
*A variational method for reconstructing and separating balanced motions and internal tides from wide-swath altimetric sea surface height observations.*  
Submitted to *Journal of Advances in Modeling Earth Systems (JAMES)*.

Preprint available on **ESS Open Archive**.

**DOI:** https://doi.org/10.22541/essoar.175455107.74338212/v1  
**URL:** https://essopenarchive.org/doi/full/10.22541/essoar.175455107.74338212

---

## Overview

The implemented method is a **variational data assimilation framework** designed to reconstruct sea surface height (SSH) fields from altimetry measurements from **SWOT** and **Nadir** satellites, while separating:

- **balanced motions**
- **internal tides**

The method is implemented and performances are evaluated within an **Observing System Simulation Experiment (OSSE)**, using synthetic satellite observations generated from **MITgcm LLC4320** simulation ([dataset](https://catalog.pangeo.io/browse/master/ocean/LLC4320/)), over a region located around **Hawai'i**. 

This repository provides the code required to create the **OSSE experiment** (`./OSSE_generator/`), reproduce the **SSH reconstruction experiments** (`./mapping/`) and compute the **performance analyses** (`./analysis/`) presented in the manuscript.

---

## OSSE Data

The datasets of the **Observing System Simulation Experiment (OSSE)** include:

- simulated satellite observations (**a**) 
- target reference SSH fields for balanced motions (**b**) and internal tides (**e**)  
- SSH fields reconstructed with the variational method for balanced motions (**c**) and internal tides (**f**)  

<p align="center">
  <img src="./figures/fig_OSSE_github_SEANOE.png" width="900">
</p>

The complete datasets are available from the **SEANOE data repository**:

**Observing System Simulation Experiment (OSSE) around Hawai‘i for Sea Surface Height (SSH) reconstruction and separation of balanced motions and internal tides from Nadir and SWOT Altimeters.**

*SEANOE (2025)*

**DOI:** https://doi.org/10.17882/107806  
**URL:** https://www.seanoe.org/data/00966/107806/

---

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/vbellemin/Bellemin-Laponnaz_2026_JAMES.git
   cd Bellemin-Laponnaz_2026_JAMES
   ```

2. **Create a new environment**
   ```bash
   conda create -n new_env python=3.10
   ```
   ```bash
   conda activate new_env
   ```

3. **Install `pyinterp` with conda-forge** 
   \
   \
   `pyinterp` provides tools for interpolating geo-referenced data used in this repository. \
   ⚠️ Installation can be very long due to several dependencies. Use of `mamba` is strongly recommended, see [Mamba instructions](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html). ⚠️
   ```bash
   conda install -c conda-forge pyinterp
   ```
5. (OPTIONAL) **Install `jax` for GPU or TPU** 
\
\
Users with access to GPUs or TPUs should first install `jax` separately in order to fully benefit from its high-performance computing capacities. See [JAX instructions](https://docs.jax.dev/en/latest/installation.html). By default, a CPU-only version of JAX will be installed if no other version is already present in the Python environment. 
   
6. **Install other dependencies with pip** 
   ```bash
   pip install --upgrade pip setuptools wheel
   pip install -e .
   ```

7. **Download OSSE data on SEANOE** 
   ```bash
   ./download_data.sh path_data
   ```
   where `path_data`is the path where to save the data (requires ~12 GB of disk space).
---

## Reproducibility

To reproduce the experiments described in the paper:

### Creation of the OSSE experiment (`./OSSE_generator/`)

The OSSE is built upon the high-resolution global ocean simulation **MITgcm LLC4320** ([dataset](https://catalog.pangeo.io/browse/master/ocean/LLC4320/)), which serves as the ground truth for sea surface height (SSH).

The model fields were first interpolated from the native grid to a regular longitude–latitude grid. The **Dynamical Atmospheric Correction** ([DOI](https://doi.org/10.24400/527896/a01-2022.001)) was then applied to account for atmospheric effects.

Successive spatial and temporal filtering operations are performed to isolate the different dynamical contributions to SSH. These operations isolate the two reference fields for **balanced motions** and **internal tides**, highlighted in **red** in the figure below, as well as the barotropic tide used to correct the ground-truth SSH.

<p align="center">
  <img src="./figures/Figure_filtering_github.001.png" width="1000">
</p>
<p align="center">
<b>Figure: </b> Scheme of successive filters applied to the total ground-truth SSH fields.
</p>

The notebooks used to perform the filtering operations are listed below. All notebooks are located in the `./OSSE_generator/` directory.

* `extract_bm.ipynb` — Applies a temporal low-pass filter to the ground-truth total SSH to extract the **balanced motions** (reference field).
* `extract_bar.ipynb` — Applies a spatial low-pass filter to the high-frequency SSH to extract the **barotropic tide** (used for correction).
* `extract_it.ipynb` — Applies a temporal band-pass filter to the internal gravity wave field to extract the **semi-diurnal total internal tide**.
* `extract_modes.ipynb` — Applies a spatial band-pass filter to the semi-diurnal internal tide field to isolate the **mode-1 semi-diurnal internal tides** (reference field).
* `notebook_miost_like_ssh.ipynb`- Applies a temporal and spatial lowpass filter to the reference balanced motions to mimick the resolution of **operational mapping products** (used for initial and boundary conditions of the QG model). 

Synthetic observations are generated by performing a trilinear interpolation along the satellite tracks. The corresponding notebooks are located in the `./OSSE_generator/interp_satellite_track` directory:

* `interp_nadir.ipynb` for the Nadir satellites.
* `interp_swot.ipynb` for the SWOT satellite in the CalVal orbit.

### SSH reconstruction (`./mapping/`)

SSH reconstruction from the synthetic OSSE observations is performed using the variational data assimilation method described in **Bellemin-Laponnaz, V., et al. (2025)**.

To run a reconstruction experiment, use the generic notebook:

`./mapping/notebooks/notebook.ipynb`

In the first section of the notebook, specify the configuration file corresponding to the desired experiment.

The configuration files used in **Bellemin-Laponnaz, V., et al. (2025)** are located in:

`./mapping/config_files/`

* `config_QGSW.py` — Framework combining **quasi-geostrophic (QG)** and **shallow-water (SW)** models for the reconstruction and separation of **balanced motions** and **internal tides**.

* `config_QG.py` *(sensitivity experiment)* — Framework based on a **QG model only**, used to reconstruct **balanced motions**.

* `config_QGSW_notime.py` *(sensitivity experiment)* — Framework combining **QG** and **SW** models, where the time dependency of the SW control variables is removed. This configuration is used to reconstruct **balanced motions** and the **coherent part of the internal tides**.

### Performance analyses (`./analysis/`)

---

## Citation

If you use this repository, please cite: `Valentin Bellemin-Laponnaz, Florian Le Guillou, Ubelmann Clément, Blayo Eric, Cosme Emmanuel. A variational method for reconstructing and separating balanced motions and internal tides from wide-swath altimetric sea surface height observations. ESS Open Archive. 2025.`
 
https://doi.org/10.22541/essoar.175455107.74338212/v1

---

## License

CC0 1.0 Universal

---

## Contact

For questions regarding the code or the experiments, please open an issue on this repository or contact the corresponding author **Emmanuel Cosme**.

