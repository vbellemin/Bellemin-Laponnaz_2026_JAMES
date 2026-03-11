# A variational method for reconstructing and separating balanced motions and internal tides from wide-swath altimetric sea surface height observations

Sources for the paper Valentin Bellemin-Laponnaz, et al. A variational method for reconstructing and separating balanced motions and internal tides from wide-swath altimetric sea surface height observations. *ESS Open Archive*, 2025, submitted to the Journal of Advances in Modeling Earth Systems (JAMES). 

DOI: https://doi.org/10.22541/essoar.175455107.74338212/v1
URL: https://essopenarchive.org/doi/full/10.22541/essoar.175455107.74338212

This repository can be used to reproduce the mapping experiments and the performance analyzes presented the manuscript.

The datasets of the OSSE experiment (satellite observations, reference SSH fields), as well as the SSH reconstructed fields can be found on the dedicated SEANOE repository: Observing System Simulation Experiment (OSSE) around Hawai‘i for Sea Surface Height (SSH) reconstruction and separation of balanced motions and internal tides from Nadir and SWOT Altimeters. *SEANOE*, 2025. 

DOI: https://doi.org/10.17882/107806
URL: https://www.seanoe.org/data/00966/107806/

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

3. **Install ```pyinterp``` with conda-forge** 
   \
   \
   ```pyinterp``` provides tools for interpolating geo-referenced data used in this repository. \
   ⚠️ Installation can be very long due to several dependencies (up to 2 hours). 
   ```bash
   conda install -c conda-forge pyinterp
   ```
   
3. **Install other dependencies with pip** 
   ```bash
   pip install --upgrade pip setuptools wheel
   pip install -e .
   ```
