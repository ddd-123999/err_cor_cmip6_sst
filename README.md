# Mamba-TempNet: Deep Learning-Based Error Correction of CMIP6 Arctic SST Projections

Projected Arctic SST warming trend halved by deep learning-based error correction of CMIP6 simulations.

![Model Architecture](Figure/Figure1/figure1.png)

## Data

Raw data sources:

- **CMIP6 simulations**: [ESGF](https://esgf-node.ornl.gov/search) (variable: `tos`, frequency: daily, experiments: historical & SSP2-4.5, member: r1i1p1f1)
 **OISST v2.1**: [NCEI](https://www.ncei.noaa.gov/data/sea-surface-temperature-optimum-interpolation/v2.1) (daily, 0.25° × 0.25°, 1982–2024)

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
Available models: `Mamba_TempNet`, `UNet`, `CnvLSTM`, `EDCDF`

Recommended training configurations:

| Model | learning_rate | batch_size |
|---|---|---|
| Mamba_TempNet | 1e-5 | 32 |
| UNet | 1e-5 | 32 |
| ConvLSTM | 1e-3 | 32 |

## License

This project is for academic research purposes.
