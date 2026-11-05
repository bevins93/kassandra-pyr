# KASSANDRA-PYR

Attention ConvLSTM-UNet for spatiotemporal wildfire danger forecasting in the La Primavera Flora and Fauna Protection Area (APFFLP), Jalisco, Mexico.

Code accompanying the article *Kassandra-pyr: Spatiotemporal Wildfire Danger Forecasting Using Deep Learning and Multi-Source Geospatial Data* (submitted to *Forests*, MDPI).

## Repository contents
- `code/v19py/` model V19 (training, SHAP/Grad-CAM, F2 analysis, seasonal danger maps) and the two article notebooks exported to `.py` (outputs removed).
- `code/` evaluation and figure scripts, `setup_v19.*`, `requirements.txt`, and `generador_mensual_y.py` (builds the monthly burned-area target masks).
- `results/` small result files (training history, validation-selected threshold, test metrics).
- `figures/` article and supplementary figures.

## Method in brief
ConvLSTM + U-Net with attention gates and a static-feature branch; Focal Tversky loss; Platt calibration; F2-optimal threshold selected on 2023 validation and applied unchanged to 2024 test (train 2020-2023).
Primary metrics are computed on the pixels inside the protected-area polygon (ANP); the full raster rectangle is a bounding box and is reported only as a reference in the article.

## Reproducing
Run from the project root; paths are set at the top of each script (or via `--base` / `--run-dir` where available). Dependencies: `pip install -r code/requirements.txt`.
1. Train: `code/v19py/ConvLSTM_UNet_V19.py` (SLURM example: `code/v19py/submit_v19_gpu.sh`).
2. Predictions for 2023 validation and 2024 test: `python code/reevaluar_completo.py`
3. Restrict evaluation to in-polygon pixels and refit Platt scaling: `python code/reevaluar_solo_ANP.py`
4. Figures: `python code/generar_todas_las_figuras.py --sub evaluacion_ANP`

## Data layout used on the processing server
- Predictors (X): `x/` (HLS mosaics, TerraClimate, static stack `x/Topographic_DL_Dataset/version1/static_features_LaPrimavera_30m_V20.tif`).
- Target (y): `y_binaria/GEE_Incendios_Binario/` holds one binary mask per fire event (260 events, 2020-2024). Their monthly union is identical to the 42 monthly masks in `y_binaria/GEE_Incendios_Mensual/`, which the evaluation scripts read (checked with `code/comparar_targets.py`).
- Absolute paths such as `/lustre/home/...` are those of our server and must be adapted.

## Data availability
Inputs are public (HLS, TerraClimate, NASADEM, NALCMS 2020, WorldPop, fire polygons); see the Data Availability Statement of the article.
Model weights and processed arrays are not included here because of size: [Zenodo DOI to be added].

## License
Code released under the [MIT/Apache-2.0] License (see `LICENSE`).

## Citation
[Add DOI after publication]
