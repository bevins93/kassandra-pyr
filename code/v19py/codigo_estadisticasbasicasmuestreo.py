python3 -c "
import numpy as np
shap = np.load('/lustre/home/kgalvan/cnn/resultados/ConvLSTM_UNet_V19_E60/ConvLSTM_UNet_V19_E60_r2/shap_n30/shap_values.npy')
labels = np.load('/lustre/home/kgalvan/cnn/resultados/ConvLSTM_UNet_V19_E60/ConvLSTM_UNet_V19_E60_r2/shap_n30/shap_labels.npy')
print('Shape SHAP:', shap.shape)
print('Positivos:', labels.sum(), '| Negativos:', (labels==0).sum())
print('Rango SHAP:', shap.min(), shap.max())
print('Media |SHAP| por canal:')
import json
nombres = ['NDVI','NBR','NDWI','mask_NDVI','mask_NBR','mask_NDWI','TempMin','TempMax','Viento','Precip','PDSI','Evapo']
vals = abs(shap).mean(axis=(1,2,3))
for n, v in zip(nombres, vals.mean(axis=0)):
    print(f'  {n}: {v:.6f}')
"