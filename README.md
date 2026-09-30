# Mamba-TempNet: Deep Learning-Based Error Correction of CMIP6 Arctic SST Projections

Projected Arctic SST warming trend halved by deep learning-based error correction of CMIP6 simulations.

![Model Architecture](Figure/Figure1/figure1.png)

## Data

Raw data sources:

- **CMIP6 simulations**: [ESGF](https://esgf-node.ornl.gov/search) (variable: `tos`, frequency: daily, experiments: historical & SSP2-4.5, member: r1i1p1f1)
 **OISST v2.1**: [NCEI](https://www.ncei.noaa.gov/data/sea-surface-temperature-optimum-interpolation/v2.1) (daily, 0.25° × 0.25°, 1982–2024)

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
Available models: `Mamba_TempNet`, `UNet`, `CnvLSTM`, `EDCDF`

Recommended training configurations:

| Model | learning_rate | batch_size |
|---|---|---|
| Mamba_TempNet | 1e-5 | 32 |
| UNet | 1e-5 | 32 |
| ConvLSTM | 1e-3 | 32 |

## License

This project is for academic research purposes.
