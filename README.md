# Mamba-TempNet: Deep Learning-Based Error Correction of CMIP6 Arctic SST Projections

Projected Arctic SST warming trend halved by deep learning-based error correction of CMIP6 simulations.

![Model Architecture](Figure/Figure1/figure1.png)

## Overview

Mamba-TempNet is a deep learning error correction framework that extends a UNet backbone with residual learning blocks throughout the encoder, bottleneck, and decoder stages, and integrates a Bidirectional Mamba (BMamba) block at the bottleneck to enable global spatial feature modeling via state-space models (SSMs) with linear computational complexity. It corrects daily Arctic sea surface temperature (SST) errors across 21 CMIP6 models under the SSP2-4.5 scenario, using OISST v2.1 as the observational reference.

Evaluated on an independent test set (2020–2024), Mamba-TempNet outperforms the statistical method EDCDF and deep learning methods ConvLSTM and UNet across all 21 models, demonstrating cross-model generalizability and spatiotemporal robustness.

## Key Findings

- Mamba-TempNet reduces **bias** by 57.4%–88.9%, **RMSE** by 13.1%–36.9%, **MAE** by 15.2%–50.2%, and improves **PCC** by 2.1%–9.5% relative to other correction methods across all 21 CMIP6 models
- The projected MMM Arctic SST warming trend is reduced from **0.203°C decade⁻¹** to **0.103°C decade⁻¹** after correction, approximately halved
- Correction performance is largely **independent of original model error levels**, demonstrating applicability to both high- and low-error models

## Data

Preprocessed data used in this study will be made available on Baidu Netdisk (coming soon).

Raw data sources:

- **CMIP6 simulations**: [ESGF](https://esgf-node.ornl.gov/search) (variable: `tos`, frequency: daily, experiments: historical & SSP2-4.5, member: r1i1p1f1)
- **OISST v2.1**: [NCEI](https://www.ncei.noaa.gov/data/sea-surface-temperature-optimum-interpolation/v2.1) (daily, 0.25° × 0.25°, 1982–2024)

The 21 CMIP6 models used in this study are listed below:

| Model | Country (Institution) | Ocean Component | Resolution (lon × lat) |
|---|---|---|---|
| ACCESS-CM2 | Australia (CSIRO-ARCCSS) | ACCESS-OM2 | 360 × 300 |
| ACCESS-ESM1-5 | Australia (CSIRO) | ACCESS-OM2 | 360 × 300 |
| BCC-CSM2-MR | China (BCC) | MOM4 | 360 × 232 |
| CanESM5 | Canada (CCCma) | NEMO3.4.1 | 360 × 291 |
| CESM2-WACCM | USA (NCAR) | POP2 | 320 × 384 |
| CMCC-CM2-SR5 | Italy (CMCC) | NEMO3.6 | 362 × 292 |
| CMCC-ESM2 | Italy (CMCC) | NEMO3.6 | 362 × 292 |
| EC-Earth3-CC | Europe (EC-Earth Consortium) | NEMO3.6 | 362 × 292 |
| EC-Earth3-Veg-LR | Europe (EC-Earth Consortium) | NEMO3.6 | 362 × 292 |
| EC-Earth3-veg | Europe (EC-Earth Consortium) | NEMO3.6 | 362 × 292 |
| EC-Earth3 | Europe (EC-Earth Consortium) | NEMO3.6 | 362 × 292 |
| GFDL-CM4 | USA (NOAA-GFDL) | GFDL-OM4p25 | 1440 × 1080 |
| GFDL-ESM4 | USA (NOAA-GFDL) | GFDL-OM4p25 | 720 × 576 |
| IPSL-CM6A-LR | France (IPSL) | NEMO-OPA | 362 × 332 |
| MIROC6 | Japan (MIROC) | COCO4.9 | 360 × 256 |
| MPI-ESM1-2-HR | Germany (MPI-M) | MPIOM1.63 | 802 × 404 |
| MPI-ESM1-2-LR | Germany (MPI-M) | MPIOM1.63 | 256 × 220 |
| MRI-ESM2-0 | Japan (MRI) | MRI.COM4.4 | 360 × 363 |
| NESM3 | China (NUIST) | NEMO3.4 | 362 × 292 |
| NorESM2-LM | Norway (NCC) | MICOM | 360 × 385 |
| NorESM2-MM | Norway (NCC) | MICOM | 360 × 385 |

## Requirements

Using conda (recommended):

```bash
conda env create -f environment.yml
conda activate pytorch
```

Key dependencies include Python 3.12, PyTorch 2.7.1 (CUDA 12.8), NumPy, xarray, netCDF4, cartopy, and scikit-learn.

## Usage

Train and evaluate Mamba-TempNet on a specific CMIP6 model:

```bash
python Project/main.py \
  --model Mamba_TempNet \
  --seq_len 3 \
  --batch_size 32 \
  --learning_rate 1e-5 \
  --train_epochs 200 \

```
o
Available models: `Mamba_TempNet`, `UNet`, `CnvLSTM`, `Linear Regress`, `EDCDF`

Recommended training configurations:

| Model | learning_rate | batch_size |
|---|---|---|
| Mamba_TempNet | 1e-5 | 32 |
| UNet | 1e-5 | 32 |
| ConvLSTM | 1e-3 | 32 |

## Acknowledgments

This work is supported by the National Natural Science Foundation of China (No. 42130402 and No. 42376231), National Key Research and Development Program of China (No. 2019YFA0607001), and Natural Science Foundation of Shanghai (No. 22ZR1427400).

## License

This project is for academic research purposes.
