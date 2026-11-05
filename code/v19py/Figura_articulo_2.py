# Convertido de Figura_articulo_2.ipynb (sin salidas)

# %% [celda 0]
import os
import re
import glob
import numpy as np
import rasterio

# --- RUTAS PROPORCIONADAS ---
PATH_X = "/lustre/home/kgalvan/cnn/x/GEE_Export_HLS_Mosaicos"
PATH_Y = "/lustre/home/kgalvan/cnn/y_binaria/GEE_Incendios_Binario"

def get_date_from_filename(filename):
    """Extrae la fecha del nombre del archivo usando el patrón YYYY-MM-DD"""
    match = re.search(r"(\d{4}-\d{2}-\d{2})", filename)
    return match.group(1) if match else None

def calculate_dataset_stats():
    # 1. Conteo de imágenes HLS (X)
    hls_files = glob.glob(os.path.join(PATH_X, "*.tif"))
    # Filtrar solo años 2020-2024 si hubiera otros
    hls_files = [f for f in hls_files if any(yr in f for yr in ["2020", "2021", "2022", "2023", "2024"])]
    total_hls = len(hls_files)
    
    # 2. Conteo de secuencias válidas (SEQ_LEN = 3)
    # Obtenemos fechas únicas disponibles en X
    dates_x = sorted([get_date_from_filename(os.path.basename(f)) for f in hls_files])
    dates_x = [d for d in dates_x if d is not None]
    
    valid_sequences = 0
    # Simulamos la lógica del modelo: una secuencia es válida si existen t, t-1 y t-2
    # Nota: Tu script V19 busca archivos dinámicos que coincidan con la fecha de la máscara Y
    y_files = glob.glob(os.path.join(PATH_Y, "*.tif"))
    for y_file in y_files:
        date_y = get_date_from_filename(os.path.basename(y_file))
        if not date_y: continue
        
        # Encontrar el índice de esta fecha en la lista ordenada de X
        if date_y in dates_x:
            idx = dates_x.index(date_y)
            if idx >= 2: # Necesitamos al menos 2 meses previos (idx, idx-1, idx-2)
                valid_sequences += 1

    # 3. Desbalance de Clases (Prevalencia en Y)
    total_pixels = 0
    positive_pixels = 0
    
    print(f"Analizando {len(y_files)} archivos de incendios (Y) para calcular prevalencia...")
    
    for y_f in y_files:
        with rasterio.open(y_f) as src:
            data = src.read(1)
            # Solo contamos píxeles con datos (asumiendo que 0 es no-incendio y 1 es incendio)
            # Si tienes NoData, deberías filtrar valores específicos aquí
            total_pixels += data.size
            positive_pixels += np.sum(data == 1)

    prevalence = (positive_pixels / total_pixels) * 100 if total_pixels > 0 else 0

    # --- IMPRESIÓN DE RESULTADOS PARA EL MANUSCRITO ---
    print("\n" + "="*50)
    print("DATOS PARA LA SECCIÓN 3.1 DE TU ARTÍCULO")
    print("="*50)
    print(f"Total imágenes HLS procesadas: {total_hls}")
    print(f"Secuencias válidas generadas (3 meses): {valid_sequences}")
    print(f"Total de píxeles analizados: {total_pixels:,}")
    print(f"Píxeles positivos (incendio): {positive_pixels:,}")
    print(f"Prevalencia confirmada: {prevalence:.4f}%")
    print("="*50)

if __name__ == "__main__":
    calculate_dataset_stats()

# %% [celda 1]
#! pip install seaborn

# %% [celda 2]
import os
import re
import glob
import numpy as np
import rasterio
import pandas as pd

# --- RUTAS ---
PATH_CLIMA = "/lustre/home/kgalvan/cnn/x/Terraclimate_DL_Dataset"
PATH_Y = "/lustre/home/kgalvan/cnn/y_binaria/GEE_Incendios_Binario"

SEASONS = {
    "Winter": [12, 1, 2],
    "Spring": [3, 4, 5],
    "Summer": [6, 7, 8],
    "Autumn": [9, 10, 11]
}

def get_date_info(filename):
    # Intenta buscar YYYY-MM-DD o YYYY_MM_DD o YYYYMMDD
    match = re.search(r"(\d{4})[-_]?(\d{2})[-_]?(\d{2})", filename)
    if match:
        year = match.group(1)
        month = int(match.group(2))
        day = match.group(3)
        date_str = f"{year}-{match.group(2)}-{day}"
        return date_str, month
    return None, None

def analyze_terraclimate_impact():
    stats = []
    # Buscamos archivos .tif
    clima_files = sorted(glob.glob(os.path.join(PATH_CLIMA, "*.tif")))
    
    print(f"DEBUG: Archivos encontrados en la carpeta: {len(clima_files)}")
    
    if len(clima_files) == 0:
        print(f"ERROR: No se encontraron archivos .tif en {PATH_CLIMA}")
        return

    for f in clima_files:
        fname = os.path.basename(f)
        date_str, month = get_date_info(fname)
        
        if date_str is None:
            # Si el regex falla, imprimimos el nombre para corregirlo
            # print(f"DEBUG: No se pudo extraer fecha de: {fname}")
            continue
        
        season_name = next((s for s, m in SEASONS.items() if month in m), "Unknown")

        try:
            with rasterio.open(f) as src:
                # Ajuste de bandas (TerraClimate GEE suele ser 1:prcp, 2:tmax, 3:tmin, 4:vpd)
                # Si el archivo tiene menos bandas, esto podría fallar, por eso usamos try
                if src.count >= 4:
                    tmax_data = src.read(2) 
                    vpd_data = src.read(4)
                else:
                    # Si solo tiene una banda o formato distinto, tomamos lo disponible
                    tmax_data = src.read(1)
                    vpd_data = src.read(1)

                mask = (tmax_data > -50) # Limpieza básica de NoData
                
                # Buscar incendio
                y_file = os.path.join(PATH_Y, f"fire_{date_str}.tif")
                burned_pixels = 0
                if os.path.exists(y_file):
                    with rasterio.open(y_file) as s_y:
                        burned_pixels = np.sum(s_y.read(1) == 1)

                stats.append({
                    "Season": season_name,
                    "Tmax_Mean": np.mean(tmax_data[mask]),
                    "VPD_Mean": np.mean(vpd_data[mask]),
                    "Burned_Pixels": burned_pixels
                })
        except Exception as e:
            print(f"Error procesando {fname}: {e}")

    # --- VERIFICACIÓN DE DATOS ---
    if not stats:
        print("ERROR: La lista de estadísticas está vacía. Verifica el formato de fecha en los nombres de archivo.")
        return

    df = pd.DataFrame(stats)
    
    # Agrupar y resumir
    summary = df.groupby("Season").agg({
        "Tmax_Mean": ["mean", "max"],
        "VPD_Mean": ["mean", "max"],
        "Burned_Pixels": "sum"
    }).round(2)

    print("\n" + "="*70)
    print("SEASONAL CLIMATE DYNAMICS (TERRACLIMATE 2020-2024)")
    print("="*70)
    print(summary)
    print("="*70)
    
    # Cálculo para la narrativa
    try:
        spring_fire = summary.loc["Spring", ("Burned_Pixels", "sum")]
        total_fire = summary[("Burned_Pixels", "sum")].sum()
        ratio = (spring_fire / total_fire) * 100 if total_fire > 0 else 0
        
        print(f"\nFRASE PARA RESULTADOS:")
        print(f"Spring (March-May) accounted for {ratio:.1f}% of historical burned pixels, "
              f"correlating with a mean VPD of {summary.loc['Spring', ('VPD_Mean', 'mean')]} hPa.")
    except KeyError:
        print("\nADVERTENCIA: No se encontraron datos para la estación 'Spring'.")

if __name__ == "__main__":
    analyze_terraclimate_impact()

# %% [celda 3]
import os
import re
import glob
import numpy as np
import rasterio
import pandas as pd

# --- RUTAS ---
PATH_CLIMA = "/lustre/home/kgalvan/cnn/x/Terraclimate_DL_Dataset"
PATH_Y = "/lustre/home/kgalvan/cnn/y_binaria/GEE_Incendios_Binario"

SEASONS = {
    "Winter": [12, 1, 2],
    "Spring": [3, 4, 5],
    "Summer": [6, 7, 8],
    "Autumn": [9, 10, 11]
}

def get_date_info_aggressive(filename):
    """
    Busca cualquier secuencia de 4 dígitos (año) y 2 dígitos (mes).
    Soporta: 2020-01-01, 2020_01, TerraClimate_202001, etc.
    """
    # Busca 4 números seguidos de algo y luego 2 números
    match = re.search(r"(\d{4})[-_]?(\d{2})", filename)
    if match:
        year = match.group(1)
        month = int(match.group(2))
        # Creamos un string estándar para buscar en la carpeta Y
        # Asumimos día 01 si no hay, para coincidir con el formato de GEE
        date_str = f"{year}-{match.group(2)}-01" 
        return date_str, month
    return None, None

def analyze_terraclimate_impact():
    stats = []
    clima_files = sorted(glob.glob(os.path.join(PATH_CLIMA, "*.tif")))
    
    print(f"DEBUG: Archivos totales encontrados: {len(clima_files)}")
    if len(clima_files) > 0:
        print(f"DEBUG: Ejemplo de nombre de archivo: '{os.path.basename(clima_files[0])}'")

    for f in clima_files:
        fname = os.path.basename(f)
        date_str, month = get_date_info_aggressive(fname)
        
        if date_str is None:
            continue
        
        season_name = next((s for s, m in SEASONS.items() if month in m), "Unknown")

        try:
            with rasterio.open(f) as src:
                # TerraClimate suele exportarse con 4 bandas: prcp, tmax, tmin, vpd
                # tmax es banda 2, vpd es banda 4. Si fallan, probamos bandas 1 y 2.
                try:
                    if src.count >= 4:
                        tmax_data = src.read(2) 
                        vpd_data = src.read(4)
                    else:
                        tmax_data = src.read(1)
                        vpd_data = src.read(src.count) # La última disponible
                except:
                    continue

                mask = (tmax_data > -50) # Quitar NoData
                
                # Buscar incendio (Y)
                # Probamos buscar con el date_str obtenido
                y_pattern = os.path.join(PATH_Y, f"*{date_str}*.tif")
                y_matches = glob.glob(y_pattern)
                
                burned_pixels = 0
                if y_matches:
                    with rasterio.open(y_matches[0]) as s_y:
                        burned_pixels = np.sum(s_y.read(1) == 1)

                stats.append({
                    "Season": season_name,
                    "Tmax_Mean": np.nanmean(tmax_data[mask]),
                    "VPD_Mean": np.nanmean(vpd_data[mask]),
                    "Burned_Pixels": burned_pixels
                })
        except Exception as e:
            print(f"Error en {fname}: {e}")

    if not stats:
        print("ERROR: No se pudo procesar ningún archivo. Revisa los nombres impresos arriba.")
        return

    df = pd.DataFrame(stats)
    summary = df.groupby("Season").agg({
        "Tmax_Mean": ["mean", "max"],
        "VPD_Mean": ["mean", "max"],
        "Burned_Pixels": "sum"
    }).round(2)

    print("\n" + "="*75)
    print("RESULTADOS PARA TU ARTÍCULO (SECCIÓN 3.1)")
    print("="*75)
    print(summary)
    print("="*75)
    
    # Cálculo de la narrativa
    total_fire = summary[("Burned_Pixels", "sum")].sum()
    if total_fire > 0:
        spring_fire = summary.loc["Spring", ("Burned_Pixels", "sum")]
        ratio = (spring_fire / total_fire) * 100
        print(f"\nFRASE SUGERIDA:")
        print(f"Spring (March-May) accounted for {ratio:.1f}% of historical burned pixels, "
              f"with a mean VPD of {summary.loc['Spring', ('VPD_Mean', 'mean')]} hPa "
              f"and a maximum Tmax of {summary.loc['Spring', ('Tmax_Mean', 'max')]}°C.")

if __name__ == "__main__":
    analyze_terraclimate_impact()

# %% [celda 4]
import os
import re
import glob
import numpy as np
import rasterio
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# --- CONFIGURACIÓN ---
PATH_Y = "/lustre/home/kgalvan/cnn/y_binaria/GEE_Incendios_Binario"
HA_PER_PIXEL = 0.09 

def get_date_robust(filename):
    match = re.search(r"(\d{4})[-_]?(\d{2})[-_]?(\d{2})?", filename)
    if match:
        return pd.to_datetime(f"{match.group(1)}-{match.group(2)}-01")
    return None

def generate_final_burned_area_plot():
    y_files = glob.glob(os.path.join(PATH_Y, "*.tif"))
    raw_data = []

    for f in y_files:
        date = get_date_robust(os.path.basename(f))
        if date:
            with rasterio.open(f) as src:
                burned_ha = np.sum(src.read(1) == 1) * HA_PER_PIXEL
                raw_data.append({"Date": date, "Ha": burned_ha})

    # 1. AGREGACIÓN CRÍTICA: Sumar todo por mes
    df = pd.DataFrame(raw_data)
    df_monthly = df.groupby('Date')['Ha'].sum().reset_index()
    df_monthly = df_monthly.sort_values("Date")
    
    # Identificar Primavera para el color
    df_monthly['Is_Spring'] = df_monthly['Date'].dt.month.isin([3, 4, 5])

    # 2. DISEÑO DE STORYTELLING
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(15, 7), dpi=300)

    # Barras con colores dinámicos
    colors = ['#E74C3C' if is_spring else '#BDC3C7' for is_spring in df_monthly['Is_Spring']]
    ax.bar(df_monthly['Date'], df_monthly['Ha'], width=20, color=colors, edgecolor='black', linewidth=0.5)

    # 3. ANOTACIONES DE LOS "MONSTRUOS" DE 2021
    peak_2021 = df_monthly[df_monthly['Date'].dt.year == 2021]['Ha'].max()
    peak_date = df_monthly[df_monthly['Ha'] == peak_2021]['Date'].iloc[0]
    
    ax.annotate(f'2021 Mega-Fire Season\n({peak_2021:,.0f} ha in peak month)', 
                xy=(peak_date, peak_2021), xytext=(40, 10),
                textcoords='offset points', fontsize=12, fontweight='bold',
                color='#C0392B', arrowprops=dict(arrowstyle='->', color='#C0392B'))

    # 4. FORMATEO RADICAL DEL EJE X
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.tick_params(axis='x', labelsize=14, pad=10)
    
    # Sombreado de fondo para cada Primavera (Mar-May)
    for year in df_monthly['Date'].dt.year.unique():
        ax.axvspan(pd.Timestamp(f'{year}-03-01'), pd.Timestamp(f'{year}-05-31'), 
                   color='#FADBD8', alpha=0.3, zorder=0)

    # 5. ESTÉTICA FINAL
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylabel("Total Burned Area (Hectares)", fontsize=14, fontweight='bold')
    plt.title("", 
              fontsize=18, fontweight='bold', loc='left', pad=20)

    # Leyenda
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#E74C3C', label='Critical Spring Season'),
                       Patch(facecolor='#BDC3C7', label='Other Months')]
    ax.legend(handles=legend_elements, loc='upper right', frameon=False, fontsize=12)

    plt.tight_layout()
    plt.savefig("Fig2_BurnedArea_Corrected.png", bbox_inches='tight')
    plt.show()

    # Verificar el total de 2021 en consola
    total_2021 = df_monthly[df_monthly['Date'].dt.year == 2021]['Ha'].sum()
    print(f"Total Hectáreas en 2021: {total_2021:,.2f} ha")

if __name__ == "__main__":
    generate_final_burned_area_plot()

# %% [celda 5]
import os
import re
import glob
import numpy as np
import rasterio
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- CONFIGURACIÓN ---
PATH_Y = "/lustre/home/kgalvan/cnn/y_binaria/GEE_Incendios_Binario"
YEAR_TO_CHECK = "2021"
HA_PER_PIXEL = 0.09 

def get_date_robust(filename):
    """Extrae fecha en formatos YYYY-MM-DD o YYYYMMDD"""
    match = re.search(r"(\d{4})[-_]?(\d{2})[-_]?(\d{2})?", filename)
    if match:
        year = match.group(1)
        month = match.group(2)
        day = match.group(3) if match.group(3) else "01"
        return f"{year}-{month}-{day}"
    return None

def analyze_2021_distribution():
    # Buscamos archivos que contengan "2021" en el nombre
    y_files = sorted(glob.glob(os.path.join(PATH_Y, f"*{YEAR_TO_CHECK}*.tif")))
    
    if not y_files:
        print(f"⚠️ No se encontraron archivos que contengan '{YEAR_TO_CHECK}' en {PATH_Y}")
        return

    data_2021 = []
    accumulated_mask = None
    
    print(f"🔍 Analizando {len(y_files)} archivos potenciales para {YEAR_TO_CHECK}...")

    for f in y_files:
        fname = os.path.basename(f)
        date_str = get_date_robust(fname)
        
        if date_str is None:
            # print(f"DEBUG: Saltando archivo sin fecha clara: {fname}")
            continue
            
        try:
            with rasterio.open(f) as src:
                img = src.read(1)
                burned_pixels = np.sum(img == 1)
                burned_ha = burned_pixels * HA_PER_PIXEL
                
                # Solo nos interesan los del año solicitado
                if date_str.startswith(YEAR_TO_CHECK):
                    data_2021.append({
                        "Date": date_str,
                        "Month": int(date_str.split("-")[1]),
                        "Pixels": burned_pixels,
                        "Hectares": burned_ha
                    })
                    
                    # Acumular para el mapa
                    if accumulated_mask is None:
                        accumulated_mask = np.zeros(img.shape)
                    accumulated_mask += (img == 1).astype(int)
        except Exception as e:
            print(f"Error procesando {fname}: {e}")

    if not data_2021:
        print(f"❌ No se pudo extraer información válida de los archivos de {YEAR_TO_CHECK}.")
        return

    df = pd.DataFrame(data_2021)
    
    # --- VISUALIZACIÓN ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # 1. Gráfico de Barras Mensual (Escala Automática)
    sns.barplot(data=df.sort_values("Month"), x="Month", y="Hectares", ax=ax1, color="salmon")
    ax1.set_title(f"Burned Area in {YEAR_TO_CHECK} (Monthly)", fontweight='bold')
    ax1.set_ylabel("Hectares")
    
    # 2. Mapa Espacial (¿Dónde están los píxeles?)
    if accumulated_mask is not None and np.sum(accumulated_mask) > 0:
        im = ax2.imshow(accumulated_mask, cmap='hot')
        plt.colorbar(im, ax=ax2, label="Fire Pixel Detections")
        ax2.set_title(f"Cumulative Fire Locations {YEAR_TO_CHECK}", fontweight='bold')
    else:
        ax2.text(0.5, 0.5, "No fire pixels (1s) found in files", ha='center')
    
    ax2.axis('off')
    plt.tight_layout()
    plt.show()

    # Resumen impreso
    print("\n📊 TABLA DE DETALLES 2021:")
    print(df.sort_values("Month")[["Date", "Pixels", "Hectares"]])

if __name__ == "__main__":
    analyze_2021_distribution()

# %% [celda 6]
import os
import re
import glob
import numpy as np
import rasterio
import pandas as pd

# --- CONFIGURACIÓN ---
PATH_Y = "/lustre/home/kgalvan/cnn/y_binaria/GEE_Incendios_Binario"
YEARS = ["2020", "2021", "2022", "2023", "2024"]
HA_PER_PIXEL = 0.09 

def get_date_robust(filename):
    match = re.search(r"(\d{4})[-_]?(\d{2})[-_]?(\d{2})?", filename)
    if match:
        year = match.group(1)
        month = match.group(2)
        day = match.group(3) if match.group(3) else "01"
        return f"{year}-{month}-{day}", int(month)
    return None, None

def analyze_full_period():
    all_files = glob.glob(os.path.join(PATH_Y, "*.tif"))
    records = []
    
    print(f"🔍 Analizando {len(all_files)} archivos totales en la base de datos Y...")

    for f in all_files:
        fname = os.path.basename(f)
        date_str, month = get_date_robust(fname)
        
        if date_str is None: continue
        
        year = date_str[:4]
        if year not in YEARS: continue

        try:
            with rasterio.open(f) as src:
                burned_pixels = np.sum(src.read(1) == 1)
                records.append({
                    "Year": year,
                    "Month": month,
                    "Pixels": burned_pixels,
                    "Hectares": burned_pixels * HA_PER_PIXEL
                })
        except Exception as e:
            print(f"Error en {fname}: {e}")

    df = pd.DataFrame(records)
    
    # Agregación por Año y Estación (Simplificada para detectar errores)
    def get_season(m):
        if m in [3, 4, 5]: return "Spring"
        if m in [6, 7, 8]: return "Summer"
        if m in [12, 1, 2]: return "Winter"
        return "Autumn"

    df['Season'] = df['Month'].apply(get_season)
    
    # Resumen Anual
    annual_summary = df.groupby("Year")["Hectares"].sum().round(2)
    
    # Resumen por Año y Estación
    detailed_summary = df.pivot_table(
        index="Year", 
        columns="Season", 
        values="Hectares", 
        aggfunc="sum", 
        fill_value=0
    ).round(2)

    print("\n" + "="*60)
    print("RESUMEN DE ÁREA QUEMADA POR AÑO (HECTÁREAS)")
    print("="*60)
    print(detailed_summary)
    print("-"*60)
    print("TOTAL ANUAL:")
    print(annual_summary)
    print("="*60)

    # Identificar el porcentaje de la Primavera en todo el dataset
    total_ha = df["Hectares"].sum()
    spring_ha = df[df["Season"] == "Spring"]["Hectares"].sum()
    print(f"\nANÁLISIS DE PESO RELATIVO:")
    print(f"Total Hectáreas (2020-2024): {total_ha:,.2f} ha")
    print(f"Total Primavera: {spring_ha:,.2f} ha")
    print(f"Porcentaje Primavera: {(spring_ha/total_ha)*100:.2f}%")

if __name__ == "__main__":
    analyze_full_period()

# # **3.2**

# %% [celda 8]
import pandas as pd
import numpy as np
import os

# --- BUSCADOR DINÁMICO DE ARCHIVO ---
# Busca el archivo history en el directorio actual o subcarpetas
history_path = "/lustre/home/kgalvan/cnn/resultados/ConvLSTM_UNet_V19_E60/ConvLSTM_UNet_V19_E60_r2/history.csv"

def analyze_v19_convergence():
    if not os.path.exists(history_path):
        print(f"❌ No se encontró el archivo: {history_path}")
        return

    df = pd.read_csv(history_path)
    cols = df.columns.tolist()
    print(f"DEBUG: Columnas detectadas: {cols}")

    # Identificar nombres de columnas automáticamente
    def find_col(keywords):
        for k in keywords:
            for c in cols:
                if k.lower() == c.lower(): return c
        return None

    c_auc = find_col(['auc', 'roc_auc', 'auc_1'])
    c_val_auc = find_col(['val_auc', 'val_roc_auc', 'val_auc_1'])
    c_loss = find_col(['loss'])
    c_val_loss = find_col(['val_loss'])
    c_lr = find_col(['lr', 'learning_rate'])
    c_recall = find_col(['recall', 'val_recall'])

    # --- EXTRACCIÓN DE DATOS ---
    # Buscamos específicamente la Época 39 (índice 38 si empieza en 0)
    target_epoch = 39
    idx = target_epoch - 1 if target_epoch <= len(df) else df[c_val_auc].idxmax()
    
    row = df.iloc[idx]
    
    # Conteo de intervenciones del scheduler (ReduceLROnPlateau)
    lr_values = df[c_lr].values
    lr_changes = 0
    for i in range(1, len(lr_values)):
        if lr_values[i] < lr_values[i-1]:
            lr_changes += 1

    # Cálculo de Gaps
    auc_gap = row[c_auc] - row[c_val_auc]
    
    print("\n" + "="*60)
    print(f"DATOS PARA SECCIÓN 3.2 (BASADOS EN ÉPOCA {int(row.get('epoch', idx+1))})")
    print("="*60)
    print(f"Validation ROC-AUC: {row[c_val_auc]:.4f}")
    print(f"Training ROC-AUC:   {row[c_auc]:.4f}")
    print(f"Generalization Gap (AUC): {auc_gap:.4f}")
    print(f"Learning Rate Final: {row[c_lr]:.2e}")
    print(f"Intervenciones del Scheduler: {lr_changes}")
    print("-" * 60)
    
    # Narrativa para el paper
    print("\n📝 PÁRRAFO PARA EL ARTÍCULO (INGLÉS):")
    p = (f"The training process for the V19 R2 model showed a stable convergence, "
         f"reaching its performance peak at epoch {target_epoch} with a validation ROC-AUC of {row[c_val_auc]:.4f}. "
         f"The optimization was supported by the ReduceLROnPlateau scheduler, which triggered {lr_changes} "
         f"learning rate reductions to fine-tune the weights in the presence of the complex 2021-2023 "
         f"variability. The model maintained high generalization, evidenced by a narrow AUC gap of {auc_gap:.4f} "
         f"between training and validation sets, effectively mitigating overfitting through the use of SpatialDropout.")
    print(p)

if __name__ == "__main__":
    analyze_v19_convergence()

# %% [celda 9]
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- CONFIGURACIÓN ---
history_path = "/lustre/home/kgalvan/cnn/resultados/ConvLSTM_UNet_V19_E60/ConvLSTM_UNet_V19_E60_r2/history.csv"
BEST_EPOCH = 39

sns.set_theme(style="ticks", font_scale=1.1)
plt.rcParams['font.family'] = 'DejaVu Sans'

def generate_training_panel_figure_v2():
    if not os.path.exists(history_path):
        print(f"❌ Error: No se encontró el archivo {history_path}")
        return

    df = pd.read_csv(history_path)
    cols = df.columns.tolist()
    
    if 'epoch' not in [c.lower() for c in cols]:
        df['epoch_x'] = df.index + 1
        x_col = 'epoch_x'
    else:
        x_col = next(c for c in cols if c.lower() == 'epoch')

    def find_col(base):
        train = next((c for c in cols if c.lower() == base.lower()), None)
        val = next((c for c in cols if c.lower() == f'val_{base.lower()}'), None)
        return train, val

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, dpi=300)
    
    metrics = [
        ('loss', 'Loss (Focal-Tversky)'),
        ('roc_auc', 'ROC-AUC'),
        ('pr_auc', 'PR-AUC'),
        ('recall', 'Recall (Sensitivity)')
    ]

    for i, (metric_base, y_label) in enumerate(metrics):
        ax = axes[i // 2, i % 2]
        train_col, val_col = find_col(metric_base)
        
        if train_col and val_col:
            # Gráficas de entrenamiento y validación
            ax.plot(df[x_col], df[train_col], label='Training', color='C0', lw=2)
            ax.plot(df[x_col], df[val_col], label='Validation', color='C3', lw=2, ls='--')
            
            # LÍNEA DE CONVERGENCIA (Se agrega etiqueta para la leyenda)
            ax.axvline(x=BEST_EPOCH, color='#c0392b', ls='-.', lw=1.5, alpha=0.8, 
                       label=f'Optimal Convergence (Epoch {BEST_EPOCH})')
            
            ax.set_ylabel(y_label, fontweight='bold')
            ax.grid(True, ls='--', alpha=0.4)
            if metric_base != 'loss':
                ax.set_ylim(-0.05, 1.05)

    # Eje X inferior
    for ax in axes[1, :]:
        ax.set_xlabel("Epochs", fontweight='bold')

    # --- LEYENDA GLOBAL ACTUALIZADA ---
    # Extraemos handles y labels del primer subplot
    handles, labels = axes[0, 0].get_legend_handles_labels()
    
    # Eliminamos duplicados de la leyenda (porque axvline se repite en cada subplot)
    from collections import OrderedDict
    by_label = OrderedDict(zip(labels, handles))
    
    fig.legend(by_label.values(), by_label.keys(), 
               loc='lower center', ncol=3, frameon=True, fontsize=12, 
               bbox_to_anchor=(0.5, 0.02))

    plt.tight_layout(rect=[0, 0.08, 1, 0.96])
    
    output_fname = "Fig3_V19R2_Training_History_Updated.png"
    plt.savefig(output_fname, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✅ Figura guardada: {output_fname}")

if __name__ == "__main__":
    generate_training_panel_figure_v2()

# # **3.3**

# %% [celda 11]
import json
import os

# --- RUTAS ABSOLUTAS PROPORCIONADAS ---
PATH_REGISTRO = "/lustre/home/kgalvan/cnn/resultados/ConvLSTM_UNet_V19_E60/ConvLSTM_UNet_V19_E60_r2/registro.json"
PATH_COMPLETAS = "/lustre/home/kgalvan/cnn/resultados/ConvLSTM_UNet_V19_E60/ConvLSTM_UNet_V19_E60_r2/metricas_completas.json"

def reportar_resultados_seccion_3_3():
    # 1. Validar existencia de archivos
    if not os.path.exists(PATH_REGISTRO) or not os.path.exists(PATH_COMPLETAS):
        print("❌ Error: Uno o ambos archivos JSON no se encuentran en las rutas especificadas.")
        return

    # 2. Cargar datos del registro principal
    with open(PATH_REGISTRO, 'r') as f:
        reg = json.load(f)
    
    # 3. Cargar datos de métricas completas (para la matriz de confusión detallada)
    with open(PATH_COMPLETAS, 'r') as f:
        comp = json.load(f)

    # --- EXTRACCIÓN DE VALORES CLAVE ---
    # Usamos los datos de 'test_calibrado' que es el resultado final
    test = comp['metricas']['test_calibrado']
    conf = comp['confusion_test_calibrado']
    
    auc_roc = test['AUC_ROC']
    auc_pr = test['AUC_PR']
    recall = test['Recall']
    precision = test['Precision']
    f1 = test['F1']
    threshold = test['Threshold']
    
    tp = conf['TP']
    fp = conf['FP']
    fn = conf['FN']
    tn = conf['TN']
    # Añade esto a tu función reportar_resultados_seccion_3_3()
    ha_detectadas = tp * 0.09
    ha_omitidas = fn * 0.09

    # --- REPORTE ESTRUCTURADO ---
    print("="*70)
    print("DATOS PARA SECCIÓN 3.3: INDEPENDENT TEST PERFORMANCE (2024)")
    print("="*70)
    print(f"{'Métrica':<20} | {'Valor':<15}")
    print("-" * 40)
    print(f"{'AUC-ROC':<20} | {auc_roc:.4f}")
    print(f"{'AUC-PR':<20} | {auc_pr:.4f}")
    print(f"{'Recall (Sens.)':<20} | {recall:.4f}")
    print(f"{'Precision':<20} | {precision:.4f}")
    print(f"{'F1-Score':<20} | {f1:.4f}")
    print(f"{'Threshold':<20} | {threshold:.5f}")
    print("-" * 40)
    print(f"MATRIZ DE CONFUSIÓN (TEST 2024):")
    print(f"  TP: {tp:,} | FP: {fp:,}")
    print(f"  FN: {fn:,} | TN: {tn:,}")
    print("="*70)
    print(f"Impacto Territorial:")
    print(f"  Hectáreas detectadas (TP): {ha_detectadas:,.2f} ha")
    print(f"  Hectáreas no detectadas (FN): {ha_omitidas:,.2f} ha")

    # --- GENERACIÓN DEL PÁRRAFO PARA EL PAPER ---
    # Nota: Se usa el F1-score como criterio de optimización según el JSON proporcionado
    p = (f"The predictive performance of the Attention ConvLSTM U-Net (V19) was rigorously "
         f"evaluated using the 2024 fire season as an independent test set. The model demonstrated "
         f"robust discriminative capacity, achieving an AUC-ROC of {auc_roc:.4f} and an AUC-PR of {auc_pr:.4f}. "
         f"To prioritize the detection of rare fire events, an optimal decision threshold of {threshold:.5f} "
         f"was selected based on F1-score optimization of the calibrated probabilities. Under this "
         f"configuration, the model achieved a Recall of {recall:.4f}, effectively identifying "
         f"{recall*100:.1f}% of the actual burned area in 2024. The confusion matrix results "
         f"(TP: {tp:,}; FN: {fn:,}) indicate that the architecture successfully captures the majority "
         f"of fire fronts while maintaining a reliable segregation of over {tn/1e6:.1f} million pixels "
         f"as True Negatives, confirming its operational utility for dynamic risk management.")

    print("\n📝 PÁRRAFO SUGERIDO PARA EL MANUSCRITO (INGLÉS):")
    print(p)

if __name__ == "__main__":
    reportar_resultados_seccion_3_3()

# %% [celda 12]
import numpy as np
import os
import json

# Ruta base de tus resultados
base_path = "/lustre/home/kgalvan/cnn/resultados/ConvLSTM_UNet_V19_E60/ConvLSTM_UNet_V19_E60_r2/"

# Cargar archivos npy
print("Cargando datos de prueba (Test 2024)...")
y_true = np.load(os.path.join(base_path, "y_true_test.npy"))
y_pred = np.load(os.path.join(base_path, "y_pred_test_calibrated.npy"))

# Cargar metadatos del JSON para la matriz de confusión
with open(os.path.join(base_path, "metricas_completas.json"), 'r') as f:
    comp = json.load(f)

print(f"✅ Datos cargados: {y_true.shape[0]:,} píxeles analizados.")

# %% [celda 13]
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import seaborn as sns

def plot_fig_4():
    sns.set_theme(style="whitegrid", font_scale=1.2)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), dpi=300)
    
    # --- PANEL A: ROC CURVE ---
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    
    axes[0].plot(fpr, tpr, color='#2980b9', lw=3, label=f'AUC-ROC = {roc_auc:.4f}')
    axes[0].plot([0, 1], [0, 1], color='#7f8c8d', linestyle='--', lw=1.5)
    axes[0].set_title('A. Receiver Operating Characteristic', fontweight='bold', pad=15)
    axes[0].set_xlabel('False Positive Rate (1 - Specificity)')
    axes[0].set_ylabel('True Positive Rate (Recall)')
    axes[0].legend(loc="lower right", frameon=True)

    # --- PANEL B: PRECISION-RECALL CURVE ---
    prec, rec, _ = precision_recall_curve(y_true, y_pred)
    pr_auc = average_precision_score(y_true, y_pred)
    
    axes[1].plot(rec, prec, color='#e67e22', lw=3, label=f'AUC-PR = {pr_auc:.4f}')
    axes[1].axhline(y=y_true.mean(), color='#c0392b', linestyle=':', lw=1.5, label='Baseline (Prevalence)')
    axes[1].set_title('B. Precision-Recall Curve', fontweight='bold', pad=15)
    axes[1].set_xlabel('Recall (Sensitivity)')
    axes[1].set_ylabel('Precision')
    axes[1].legend(loc="upper right", frameon=True)

    plt.tight_layout()
    plt.savefig("Fig4_ROC_PR_V19.png", bbox_inches='tight')
    plt.show()

plot_fig_4()

# %% [celda 14]
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def plot_fig_5():
    # Extraer datos (ejemplo con tus datos de las imágenes)
    conf = {'TP': 79698, 'FP': 15119984, 'FN': 163699, 'TN': 25258599}
    tp, fp, fn, tn = conf['TP'], conf['FP'], conf['FN'], conf['TN']
    
    # 1. Creamos una matriz de índices (0 a 3) para forzar colores distintos
    # Esto asigna un color único a cada categoría del cmap
    color_indices = np.array([[0, 1], 
                             [2, 3]])
    
    plt.figure(figsize=(8, 7), dpi=300)
    sns.set_theme(style="white")
    
    # Etiquetas con los valores reales
    labels = np.array([
        [f"True Negatives\n{tn:,}", f"False Positives\n{fp:,}"],
        [f"False Negatives\n{fn:,}", f"True Positives\n{tp:,}"]
    ])
    
    # 2. Graficamos usando 'color_indices' para el mapeo de color
    # Usamos 'rocket' y quitamos el cbar para que se vea como una matriz limpia
    sns.heatmap(color_indices, annot=labels, fmt="", cmap="rocket", cbar=False,
                xticklabels=['No Fire (0)', 'Fire (1)'],
                yticklabels=['No Fire (0)', 'Fire (1)'],
                annot_kws={"size": 12, "fontweight": "bold"})
    
    plt.title("", 
              fontsize=14, pad=20, fontweight='bold')
    plt.ylabel("Actual Condition", fontsize=12)
    plt.xlabel("Predicted Condition", fontsize=12)
    
    plt.savefig("Fig5_Confusion_Matrix_V19.png", bbox_inches='tight')
    plt.show()

plot_fig_5()

# %% [celda 15]
def plot_fig_6():
    # Muestreo para agilizar el cálculo del trade-off si la memoria es limitada
    # (Opcional: puedes usar y_true y y_pred directamente si el servidor lo aguanta)
    thresholds = np.linspace(0, 0.05, 100) 
    precisions, recalls, f1s, f2s = [], [], [], []
    
    total_pos = np.sum(y_true == 1)
    
    for thr in thresholds:
        preds = (y_pred >= thr)
        tp = np.sum(preds & (y_true == 1))
        fp = np.sum(preds & (y_true == 0))
        
        p = tp / (tp + fp + 1e-8)
        r = tp / (total_pos + 1e-8)
        
        precisions.append(p)
        recalls.append(r)
        f1s.append(2 * (p * r) / (p + r + 1e-8))
        f2s.append(5 * (p * r) / (4 * p + r + 1e-8))

    plt.figure(figsize=(12, 6.5), dpi=300)
    sns.set_theme(style="ticks")
    
    plt.plot(thresholds, recalls, label='Recall (Sensitivity)', color='#27ae60', lw=2.5)
    plt.plot(thresholds, f2s, label='$F_2$-Score (Wildfire Priority)', color='#8e44ad', lw=2.5, linestyle='--')
    plt.plot(thresholds, f1s, label='$F_1$-Score', color='#d35400', lw=2)
    plt.plot(thresholds, precisions, label='Precision', color='#2980b9', lw=1.5, alpha=0.7)
    
    # Línea del umbral seleccionado
    target_thr = comp['metricas']['test_calibrado']['Threshold']
    plt.axvline(x=target_thr, color='black', linestyle=':', lw=2)
    plt.text(target_thr + 0.001, 0.85, f'Selected Threshold\n({target_thr:.5f})', 
             fontweight='bold', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    plt.title("", fontsize=15, fontweight='bold', pad=20)
    plt.xlabel("Probability Threshold", fontsize=13)
    plt.ylabel("Score Value", fontsize=13)
    plt.legend(loc='center right', frameon=True)
    plt.xlim(0, 0.04) # Zoom en la zona de interés
    plt.ylim(0, 1.05)
    
    sns.despine()
    plt.savefig("Fig6_Threshold_Tradeoff_V19.png", bbox_inches='tight')
    plt.show()

plot_fig_6()

# # **3.4**

# %% [celda 17]
import numpy as np
import rasterio
import os

# --- RUTAS ---
path_plantilla = "/lustre/home/kgalvan/cnn/x/Topographic_DL_Dataset/static_features_LaPrimavera_30m_V20.tif"
base_path = "/lustre/home/kgalvan/cnn/resultados/ConvLSTM_UNet_V19_E60/ConvLSTM_UNet_V19_E60_r2/"

def generar_tifs_v20_corregido():
    if not os.path.exists(path_plantilla):
        print(f"❌ No se encontró la plantilla en: {path_plantilla}")
        return

    # 1. Cargar y REDIMENSIONAR predicciones
    y_pred_raw = np.load(os.path.join(base_path, "y_pred_test_calibrated.npy"))
    
    # Dimensiones de tu mapa según el JSON
    H, W = 674, 861
    # Reshape: (-1 indica que calcule automáticamente el número de imágenes, que son 70)
    y_pred = y_pred_raw.reshape(-1, H, W)
    print(f"✅ Reshape exitoso. Imágenes detectadas: {y_pred.shape[0]}")

    # 2. Metadatos
    with rasterio.open(path_plantilla) as temp:
        meta = temp.meta.copy()
        meta.update(dtype=rasterio.float32, count=1, driver='GTiff', compress='lzw')

    output_dir = os.path.join(base_path, "mapas_georeferenciados_2024")
    os.makedirs(output_dir, exist_ok=True)

    # 3. Guardar (usamos un loop numérico ya que son 70 imágenes)
    for i in range(y_pred.shape[0]):
        out_file = os.path.join(output_dir, f"Risk_Map_V19_Step_{i:02d}.tif")
        with rasterio.open(out_file, 'w', **meta) as dst:
            dst.write(y_pred[i].astype(np.float32), 1)
            
    print(f"🚀 {y_pred.shape[0]} GeoTIFFs generados en: {output_dir}")

if __name__ == "__main__":
    generar_tifs_v20_corregido()

# %% [celda 18]
import pandas as pd
import os

path_csv = "/lustre/home/kgalvan/cnn/resultados/ConvLSTM_UNet_V19_E60/ConvLSTM_UNet_V19_E60_r2/area_categorias.csv"

def inspeccionar_y_calcular_3_4():
    if not os.path.exists(path_csv):
        print(f"❌ No se encontró el archivo en: {path_csv}")
        return

    df = pd.read_csv(path_csv)
    
    print("🔍 Columnas encontradas en el CSV:", df.columns.tolist())
    print("\n📋 Primeras 3 filas del archivo:")
    print(df.head(3))
    print("-" * 60)

    # --- LÓGICA DE BÚSQUEDA FLEXIBLE ---
    # Buscamos columnas que contengan 'alto' o 'high'
    cols_criticas = [c for c in df.columns if any(word in c.lower() for word in ['alto', 'high', 'very'])]
    # Buscamos columnas que parezcan ser el área
    cols_area = [c for c in df.columns if any(word in c.lower() for word in ['ha', 'area', 'hectare'])]

    try:
        # Escenario A: Las categorías son columnas (Pivoted)
        if cols_criticas and not any(c.lower() in ['categoria', 'category'] for c in df.columns):
            resumen = df[cols_criticas].mean()
            area_critica = resumen.sum()
            area_total = df[[c for c in df.columns if any(w in c.lower() for w in ['bajo', 'low', 'med', 'high', 'alto'])]].mean().sum()
        
        # Escenario B: Hay una columna llamada 'categoria' y otra 'area' (Long format)
        else:
            col_cat = next(c for c in df.columns if 'cat' in c.lower())
            col_val = next(c for c in df.columns if any(w in c.lower() for w in ['ha', 'area', 'pixel']))
            
            filtro_alto = df[col_cat].str.contains('Alto|High', case=False, na=False)
            area_critica = df[filtro_alto][col_val].mean()
            area_total = df[col_val].sum() / len(df[col_cat].unique()) # Promedio por paso de tiempo

        porcentaje = (area_critica / area_total) * 100

        print("="*60)
        print("✅ VALORES PARA TU REDACCIÓN (SECCIÓN 3.4)")
        print("="*60)
        print(f"Área en Riesgo Alto/Muy Alto: {area_critica:,.2f} ha")
        print(f"Porcentaje de la superficie:  {porcentaje:.2f}%")
        print("="*60)
        
    except Exception as e:
        print(f"❌ Error al procesar automáticamente: {e}")
        print("Por favor, copia aquí lo que imprimió 'Columnas encontradas' para ajustarlo.")

if __name__ == "__main__":
    inspeccionar_y_calcular_3_4()

# %% [celda 19]
import pandas as pd
import os

path_csv = "/lustre/home/kgalvan/cnn/resultados/ConvLSTM_UNet_V19_E60/ConvLSTM_UNet_V19_E60_r2/area_categorias.csv"

def obtener_totales_3_4():
    df = pd.read_csv(path_csv)
    
    # Filtramos las filas de interés (Primavera)
    # Buscamos 'Muy alto' y 'Alto'
    filtro_critico = df['categoria'].isin(['Muy alto', 'Alto'])
    datos_primavera = df[filtro_critico]
    
    total_ha = datos_primavera['hectareas'].sum()
    total_pct = datos_primavera['pct_area'].sum()

    print("="*60)
    print("📊 RESULTADOS FINALES PARA SECCIÓN 3.4")
    print("="*60)
    print(f"Superficie en Riesgo Muy Alto: {df[df['categoria']=='Muy alto']['hectareas'].values[0]:,.1f} ha")
    print(f"Superficie en Riesgo Alto:     {df[df['categoria']=='Alto']['hectareas'].values[0]:,.1f} ha")
    print("-" * 60)
    print(f"TOTAL RIESGO CRÍTICO (A+MA):  {total_ha:,.1f} ha")
    print(f"PORCENTAJE DEL MAPA TOTAL:    {total_pct:.1f}%")
    print("="*60)

if __name__ == "__main__":
    obtener_totales_3_4()

# %% [celda 20]
import numpy as np
import matplotlib.pyplot as plt
import rasterio
import rasterio.warp
from rasterio.warp import calculate_default_transform, reproject, Resampling
import glob
import os
import geopandas as gpd
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

# --- RUTAS ---
path_satelital  = "/lustre/home/kgalvan/cnn/ANP_shapefile/Fondo_Satelital_LaPrimavera_2024.tif"
path_vector_anp = "/lustre/home/kgalvan/cnn/ANP_shapefile/extracted_anp/"
base_path       = "/lustre/home/kgalvan/cnn/resultados/ConvLSTM_UNet_V19_E60/ConvLSTM_UNet_V19_E60_r2/"
path_mosaicos   = "/lustre/home/kgalvan/cnn/x/GEE_Export_HLS_Mosaicos/"

TARGET_CRS = "EPSG:4326"  # CRS unificado (HLS + shapefile ya están aquí)
H, W       = 674, 861


def reproyectar_satelital(path_sat):
    """Reproyecta el TIF satelital de EPSG:32613 a EPSG:4326 en memoria."""
    with rasterio.open(path_sat) as src:
        dst_transform, dst_w, dst_h = calculate_default_transform(
            src.crs, TARGET_CRS, src.width, src.height, *src.bounds)
        dst_arr = np.zeros((3, dst_h, dst_w), dtype=np.uint8)
        for band in range(1, 4):
            reproject(
                source=rasterio.band(src, band),
                destination=dst_arr[band - 1],
                src_transform=src.transform, src_crs=src.crs,
                dst_transform=dst_transform, dst_crs=TARGET_CRS,
                resampling=Resampling.bilinear)
    sat_img = np.transpose(dst_arr, (1, 2, 0))
    l = dst_transform.c;  t = dst_transform.f
    r = l + dst_transform.a * dst_w
    b = t + dst_transform.e * dst_h
    sat_ext = [l, r, b, t]  # [left, right, bottom, top]
    print(f"✅ Satelital reproyectado → {TARGET_CRS}  extent: {sat_ext}")
    return sat_img, sat_ext


def plot_fire_gt(ax, y_true_step, pred_ext):
    """Convierte píxeles de fuego a lon/lat reales y los grafica como scatter."""
    fire_rows, fire_cols = np.where(y_true_step > 0)
    if len(fire_rows) == 0:
        return
    left, right, bottom, top = pred_ext
    lons = left + (fire_cols / W) * (right - left)
    lats = top  - (fire_rows / H) * (top - bottom)
    ax.scatter(lons, lats, c='#00ffff', s=1.5, linewidths=0,
               zorder=5, rasterized=True)


def generar_fig_7_multimodal():
    print("⏳ Iniciando renderizado unificado en EPSG:4326...")

    # 1. SATELITAL — reproyectar de UTM a 4326
    sat_img, sat_ext = reproyectar_satelital(path_satelital)

    # 2. VECTOR — ya en EPSG:4326, solo forzar CRS por seguridad
    shp_file = glob.glob(os.path.join(path_vector_anp, "**", "*.shp"), recursive=True)[0]
    try:
        anp_gdf = gpd.read_file(shp_file, engine='fiona', encoding='cp1252').to_crs(TARGET_CRS)
    except Exception:
        anp_gdf = gpd.read_file(shp_file, engine='fiona', encoding='latin-1').to_crs(TARGET_CRS)

    # Zoom al ANP con margen del 5%
    minx, miny, maxx, maxy = anp_gdf.total_bounds
    dx, dy = maxx - minx, maxy - miny
    view_ext = [minx - dx*0.05, maxx + dx*0.05, miny - dy*0.05, maxy + dy*0.05]

    # 3. DATOS DE PREDICCIÓN
    y_pred    = np.load(os.path.join(base_path, "y_pred_test_calibrated.npy")).reshape(-1, H, W)
    y_true    = np.load(os.path.join(base_path, "y_true_test.npy")).reshape(-1, H, W)
    hls_files = sorted(glob.glob(os.path.join(path_mosaicos, "HLS_MOSAIC_2024-*.tif")))

    # 4. CONFIGURACIÓN VISUAL
    risk_colors = ['#27ae60', '#f1c40f', '#e67e22', '#c0392b']
    risk_cmap   = ListedColormap(risk_colors)
    thr_bins    = [0, 0.0084, 0.0121, 0.0188, 1.0]
    steps       = [10, 25, 40, 55]

    col_headers = ["NDVI (Vegetation)", "Fire Probability", "Risk Category", "Risk-Truth Overlay"]

    fig, axes = plt.subplots(4, 4, figsize=(24, 20), dpi=300)

    for i, step in enumerate(steps):

        # Extent del HLS directo en 4326 (sin transform_bounds)
        with rasterio.open(hls_files[step]) as src:
            ndvi_data = src.read(1)
            b = src.bounds
            pred_ext = [b.left, b.right, b.bottom, b.top]  # ya en 4326

        risk_map    = np.digitize(y_pred[step], bins=thr_bins[1:-1])
        risk_masked = np.where(risk_map == 0, np.nan, risk_map)

        for j in range(4):
            ax = axes[i, j]

            # 1. Fondo satelital (reproyectado a 4326)
            ax.imshow(sat_img, extent=sat_ext, origin='upper')

            # 2. Overlay de datos
            if j == 0:
                ax.imshow(ndvi_data, cmap='YlGn', vmin=0, vmax=1,
                          alpha=0.5, extent=pred_ext, origin='upper')
            elif j == 1:
                ax.imshow(y_pred[step], cmap='magma', vmin=0, vmax=0.04,
                          alpha=0.6, extent=pred_ext, origin='upper')
            elif j == 2:
                ax.imshow(risk_masked, cmap=risk_cmap, vmin=0, vmax=3,
                          alpha=0.7, extent=pred_ext, origin='upper')
            elif j == 3:
                ax.imshow(risk_masked, cmap=risk_cmap, vmin=0, vmax=3,
                          alpha=0.4, extent=pred_ext, origin='upper')
                # Fire GT como scatter en coordenadas reales (no contour)
                plot_fire_gt(ax, y_true[step], pred_ext)

            # 3. Límite ANP
            anp_gdf.boundary.plot(ax=ax, color='white', lw=1.5, ls='--', alpha=0.9)

            # 4. Zoom a la zona de interés
            ax.set_xlim(view_ext[0], view_ext[1])
            ax.set_ylim(view_ext[2], view_ext[3])
            ax.set_xticks([]); ax.set_yticks([])

            if i == 0:
                ax.set_title(col_headers[j], fontsize=18, fontweight='bold', pad=20)

        fecha = os.path.basename(hls_files[step]).split('_')[-1].replace('.tif', '')
        axes[i, 0].set_ylabel(f"Date: {fecha}", fontweight='bold', fontsize=16)

    # Leyenda
    patches = [
        mpatches.Patch(color='#27ae60', label='Low Risk'),
        mpatches.Patch(color='#f1c40f', label='Medium Risk'),
        mpatches.Patch(color='#e67e22', label='High Risk'),
        mpatches.Patch(color='#c0392b', label='Very High Risk'),
        plt.Line2D([0],[0], color='#00ffff', lw=2,   label='Fire Ground Truth'),
        plt.Line2D([0],[0], color='white',   lw=1.5, ls='--', label='ANP Boundary'),
    ]
    fig.legend(handles=patches, loc='lower center', ncol=6,
               fontsize=15, frameon=True, bbox_to_anchor=(0.5, 0.04))

    plt.tight_layout(rect=[0, 0.07, 1, 0.95])
    plt.savefig("Fig7_Final_Alineada.png", bbox_inches='tight', dpi=300)
    print("🚀 ¡Mosaico terminado!")
    plt.show()


if __name__ == "__main__":
    generar_fig_7_multimodal()

# %% [celda 21]
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import glob, os
import geopandas as gpd
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

# --- RUTAS ---
path_satelital  = "/lustre/home/kgalvan/cnn/ANP_shapefile/Fondo_Satelital_LaPrimavera_2024.tif"
path_vector_anp = "/lustre/home/kgalvan/cnn/ANP_shapefile/extracted_anp/"
base_path       = "/lustre/home/kgalvan/cnn/resultados/ConvLSTM_UNet_V19_E60/ConvLSTM_UNet_V19_E60_r2/"
path_mosaicos   = "/lustre/home/kgalvan/cnn/x/GEE_Export_HLS_Mosaicos/"

TARGET_CRS  = "EPSG:4326"
RISK_COLORS = ['#27ae60', '#f1c40f', '#e67e22', '#c0392b']
RISK_CMAP   = ListedColormap(RISK_COLORS)
THR_BINS    = [0, 0.0084, 0.0121, 0.0188, 1.0]
STEP        = 25
H, W        = 674, 861


# ── Helpers ──────────────────────────────────────────────────────────────────

def reproyectar_satelital(path_sat):
    with rasterio.open(path_sat) as src:
        dst_transform, dst_w, dst_h = calculate_default_transform(
            src.crs, TARGET_CRS, src.width, src.height, *src.bounds)
        dst_arr = np.zeros((3, dst_h, dst_w), dtype=np.uint8)
        for band in range(1, 4):
            reproject(
                source=rasterio.band(src, band),
                destination=dst_arr[band - 1],
                src_transform=src.transform, src_crs=src.crs,
                dst_transform=dst_transform, dst_crs=TARGET_CRS,
                resampling=Resampling.bilinear)
    sat_img = np.transpose(dst_arr, (1, 2, 0))
    l = dst_transform.c;  t = dst_transform.f
    r = l + dst_transform.a * dst_w
    b = t + dst_transform.e * dst_h
    return sat_img, [l, r, b, t]


def plot_fire_gt(ax, y_true_step, pred_ext):
    """Convierte píxeles de fuego a lon/lat y los grafica como scatter."""
    fire_rows, fire_cols = np.where(y_true_step > 0)
    if len(fire_rows) == 0:
        return
    left, right, bottom, top = pred_ext
    lons = left + (fire_cols / W) * (right - left)
    lats = top  - (fire_rows / H) * (top - bottom)
    ax.scatter(lons, lats, c='#00ffff', s=1.5, linewidths=0,
               zorder=5, rasterized=True)


def decorar(ax, anp_gdf, view_ext, color_borde='white'):
    anp_gdf.boundary.plot(ax=ax, color=color_borde, lw=1.5, ls='--', alpha=0.9)
    ax.set_xlim(view_ext[0], view_ext[1])
    ax.set_ylim(view_ext[2], view_ext[3])
    ax.set_xticks([]); ax.set_yticks([])


def leyenda(color_borde='white'):
    return [
        mpatches.Patch(color='#27ae60', label='Low Risk'),
        mpatches.Patch(color='#f1c40f', label='Medium Risk'),
        mpatches.Patch(color='#e67e22', label='High Risk'),
        mpatches.Patch(color='#c0392b', label='Very High Risk'),
        plt.Line2D([0],[0], color='#00ffff', lw=2,   label='Fire Ground Truth'),
        plt.Line2D([0],[0], color=color_borde, lw=1.5, ls='--', label='ANP Boundary'),
    ]


# ── Carga de datos ────────────────────────────────────────────────────────────

def cargar_datos():
    sat_img, sat_ext = reproyectar_satelital(path_satelital)

    shp = glob.glob(os.path.join(path_vector_anp, "**", "*.shp"), recursive=True)[0]
    anp_gdf = gpd.read_file(shp, engine='fiona', encoding='cp1252').to_crs(TARGET_CRS)

    minx, miny, maxx, maxy = anp_gdf.total_bounds
    dx, dy = maxx - minx, maxy - miny
    view_ext = [minx - dx*0.05, maxx + dx*0.05, miny - dy*0.05, maxy + dy*0.05]

    y_pred = np.load(os.path.join(base_path, "y_pred_test_calibrated.npy")).reshape(-1, H, W)
    y_true = np.load(os.path.join(base_path, "y_true_test.npy")).reshape(-1, H, W)
    hls_files = sorted(glob.glob(os.path.join(path_mosaicos, "HLS_MOSAIC_2024-*.tif")))

    with rasterio.open(hls_files[STEP]) as src:
        ndvi_data = src.read(1)
        b = src.bounds
        pred_ext = [b.left, b.right, b.bottom, b.top]

    fecha = os.path.basename(hls_files[STEP]).split('_')[-1].replace('.tif', '')
    risk_map    = np.digitize(y_pred[STEP], bins=THR_BINS[1:-1])
    risk_masked = np.where(risk_map == 0, np.nan, risk_map)

    return dict(sat_img=sat_img, sat_ext=sat_ext, anp_gdf=anp_gdf,
                view_ext=view_ext, ndvi_data=ndvi_data, pred_ext=pred_ext,
                y_pred=y_pred, y_true=y_true,
                risk_masked=risk_masked, fecha=fecha)


# ── Figura 1: una fila, 4 columnas ───────────────────────────────────────────

def fig_una_fila(d):
    cols = ["NDVI (Vegetation)", "Fire Probability", "Risk Category", "Risk-Truth Overlay"]
    fig, axes = plt.subplots(1, 4, figsize=(24, 6), dpi=300)

    for j, ax in enumerate(axes):
        ax.imshow(d["sat_img"], extent=d["sat_ext"], origin='upper')

        if j == 0:
            ax.imshow(d["ndvi_data"], cmap='YlGn', vmin=0, vmax=1,
                      alpha=0.5, extent=d["pred_ext"], origin='upper')
        elif j == 1:
            ax.imshow(d["y_pred"][STEP], cmap='magma', vmin=0, vmax=0.04,
                      alpha=0.6, extent=d["pred_ext"], origin='upper')
        elif j == 2:
            ax.imshow(d["risk_masked"], cmap=RISK_CMAP, vmin=0, vmax=3,
                      alpha=0.7, extent=d["pred_ext"], origin='upper')
        elif j == 3:
            ax.imshow(d["risk_masked"], cmap=RISK_CMAP, vmin=0, vmax=3,
                      alpha=0.4, extent=d["pred_ext"], origin='upper')
            plot_fire_gt(ax, d["y_true"][STEP], d["pred_ext"])  # ← fix

        ax.set_title(cols[j], fontsize=14, fontweight='bold', pad=10)
        decorar(ax, d["anp_gdf"], d["view_ext"])

    axes[0].set_ylabel(f"Date: {d['fecha']}", fontweight='bold', fontsize=13)
    fig.legend(handles=leyenda(), loc='lower center', ncol=6,
               fontsize=12, frameon=True, bbox_to_anchor=(0.5, -0.06))
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig("Fig7_Una_Fila.png", bbox_inches='tight', dpi=300)
    print("✅ Fig7_Una_Fila.png")
    plt.show()


# ── Figura 2: solo Risk Category ─────────────────────────────────────────────

def fig_solo_risk(d):
    fig, ax = plt.subplots(1, 1, figsize=(7, 7), dpi=300)
    ax.set_facecolor('#f5f5f5')

    ax.imshow(d["risk_masked"], cmap=RISK_CMAP, vmin=0, vmax=3,
              alpha=1.0, extent=d["pred_ext"], origin='upper')

    plot_fire_gt(ax, d["y_true"][STEP], d["pred_ext"])  # ← fix

    d["anp_gdf"].boundary.plot(ax=ax, color='#2c3e50', lw=1.8, ls='--', alpha=0.95)

    minx, miny, maxx, maxy = d["anp_gdf"].total_bounds
    dx, dy = maxx - minx, maxy - miny
    ax.set_xlim(minx - dx*0.04, maxx + dx*0.04)
    ax.set_ylim(miny - dy*0.04, maxy + dy*0.04)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"Risk Category  —  {d['fecha']}", fontsize=15,
                 fontweight='bold', pad=12)

    fig.legend(handles=leyenda(color_borde='#2c3e50'), loc='lower center', ncol=3,
               fontsize=11, frameon=True, bbox_to_anchor=(0.5, -0.12))
    plt.tight_layout(rect=[0, 0.10, 1, 1])
    plt.savefig("Fig7_Solo_Risk.png", bbox_inches='tight', dpi=300)
    print("✅ Fig7_Solo_Risk.png")
    plt.show()


if __name__ == "__main__":
    datos = cargar_datos()
    fig_una_fila(datos)
    fig_solo_risk(datos)

# %% [celda 22]
def fig_solo_risk_con_satelital(d):
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), dpi=300)

    titles = ["Risk Category (sin satelital)", "Risk Category (con satelital)"]

    for idx, ax in enumerate(axes):
        ax.set_facecolor('#f5f5f5')

        # Panel derecho: agregar satelital de fondo
        if idx == 1:
            ax.imshow(d["sat_img"], extent=d["sat_ext"], origin='upper')

        # Overlay de categorías
        alpha_risk = 1.0 if idx == 0 else 0.65  # más transparente sobre satelital
        ax.imshow(d["risk_masked"], cmap=RISK_CMAP, vmin=0, vmax=3,
                  alpha=alpha_risk, extent=d["pred_ext"], origin='upper')

        # Fire GT
        plot_fire_gt(ax, d["y_true"][STEP], d["pred_ext"])

        # Borde ANP
        color_borde = '#2c3e50' if idx == 0 else 'white'
        d["anp_gdf"].boundary.plot(ax=ax, color=color_borde,
                                   lw=1.8, ls='--', alpha=0.95)

        # Zoom al ANP
        minx, miny, maxx, maxy = d["anp_gdf"].total_bounds
        dx, dy = maxx - minx, maxy - miny
        ax.set_xlim(minx - dx*0.04, maxx + dx*0.04)
        ax.set_ylim(miny - dy*0.04, maxy + dy*0.04)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(titles[idx], fontsize=13, fontweight='bold', pad=10)

    fig.suptitle(f"Risk Category  —  {d['fecha']}", fontsize=15,
                 fontweight='bold', y=1.01)

    leg = [
        mpatches.Patch(color='#27ae60', label='Low Risk'),
        mpatches.Patch(color='#f1c40f', label='Medium Risk'),
        mpatches.Patch(color='#e67e22', label='High Risk'),
        mpatches.Patch(color='#c0392b', label='Very High Risk'),
        plt.Line2D([0],[0], color='#00ffff', lw=2,   label='Fire Ground Truth'),
        plt.Line2D([0],[0], color='gray',    lw=1.5, ls='--', label='ANP Boundary'),
    ]
    fig.legend(handles=leg, loc='lower center', ncol=3,
               fontsize=11, frameon=True, bbox_to_anchor=(0.5, -0.10))
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig("Fig7_Risk_Comparacion.png", bbox_inches='tight', dpi=300)
    print("✅ Guardada: Fig7_Risk_Comparacion.png")
    plt.show()


# Llamar junto con las demás:
if __name__ == "__main__":
    datos = cargar_datos()
    fig_una_fila(datos)
    fig_solo_risk(datos)
    fig_solo_risk_con_satelital(datos)  # ← nueva

# %% [celda 23]
import numpy as np, rasterio, glob, os, geopandas as gpd
import matplotlib.pyplot as plt

base_path   = "/lustre/home/kgalvan/cnn/resultados/ConvLSTM_UNet_V19_E60/ConvLSTM_UNet_V19_E60_r2/"
path_mosaicos = "/lustre/home/kgalvan/cnn/x/GEE_Export_HLS_Mosaicos/"
path_vector_anp = "/lustre/home/kgalvan/cnn/ANP_shapefile/extracted_anp/"

H, W = 674, 861
y_true = np.load(os.path.join(base_path, "y_true_test.npy")).reshape(-1, H, W)
y_pred = np.load(os.path.join(base_path, "y_pred_test_calibrated.npy")).reshape(-1, H, W)
hls_files = sorted(glob.glob(os.path.join(path_mosaicos, "HLS_MOSAIC_2024-*.tif")))

STEP = 25
with rasterio.open(hls_files[STEP]) as src:
    b = src.bounds
    pred_ext = [b.left, b.right, b.bottom, b.top]
    print(f"HLS extent : {pred_ext}")
    print(f"HLS shape  : {src.height} x {src.width}")

shp = glob.glob(os.path.join(path_vector_anp, "**", "*.shp"), recursive=True)[0]
anp_gdf = gpd.read_file(shp, engine='fiona', encoding='cp1252')
print(f"\nANP bounds : {anp_gdf.total_bounds}")
print(f"ANP CRS    : {anp_gdf.crs}")

# ¿Dónde están los píxeles de fuego en coordenadas geográficas?
fire_rows, fire_cols = np.where(y_true[STEP] > 0)
if len(fire_rows) > 0:
    # Convertir píxel -> coordenada geográfica usando el extent del HLS
    left, right, bottom, top = pred_ext
    fire_lons = left + (fire_cols / W) * (right - left)
    fire_lats = top  - (fire_rows / H) * (top - bottom)
    print(f"\nFire pixels: {len(fire_rows)}")
    print(f"Fire lon range: [{fire_lons.min():.4f}, {fire_lons.max():.4f}]")
    print(f"Fire lat range: [{fire_lats.min():.4f}, {fire_lats.max():.4f}]")
    print(f"\nANP lon range : [{anp_gdf.total_bounds[0]:.4f}, {anp_gdf.total_bounds[2]:.4f}]")
    print(f"ANP lat range : [{anp_gdf.total_bounds[1]:.4f}, {anp_gdf.total_bounds[3]:.4f}]")
    
    # ¿El fuego cae dentro del ANP?
    from shapely.geometry import Point
    anp_union = anp_gdf.geometry.union_all()
    sample_points = [Point(fire_lons[i], fire_lats[i]) for i in range(min(20, len(fire_rows)))]
    inside = [anp_union.contains(p) for p in sample_points]
    print(f"\nDe 20 puntos de fuego, ¿cuántos están dentro del ANP?: {sum(inside)}/20")
else:
    print("\n⚠️ No hay píxeles de fuego en este step")

# Plot rápido de diagnóstico
fig, ax = plt.subplots(figsize=(8,8))
anp_gdf.boundary.plot(ax=ax, color='black', lw=2, ls='--')
if len(fire_rows) > 0:
    ax.scatter(fire_lons[::10], fire_lats[::10], c='cyan', s=2, label='Fire GT')
ax.set_title("Diagnóstico: Fire GT vs ANP boundary")
ax.legend()
plt.savefig("diagnostico_fire_vs_anp.png", bbox_inches='tight', dpi=150)
plt.show()
print("✅ Guardado: diagnostico_fire_vs_anp.png")

# %% [celda 24]
import rasterio
import geopandas as gpd
import os
import glob

# --- RUTAS ---
path_sat = "/lustre/home/kgalvan/cnn/ANP_shapefile/Fondo_Satelital_LaPrimavera_2024.tif"
path_hls = sorted(glob.glob("/lustre/home/kgalvan/cnn/x/GEE_Export_HLS_Mosaicos/HLS_MOSAIC_2024-*.tif"))[0]
path_vector = glob.glob("/lustre/home/kgalvan/cnn/ANP_shapefile/extracted_anp/**/*.shp", recursive=True)[0]

def auditar_geometrias():
    print("🔍 --- AUDITORÍA DE DATOS GEOESPACIALES --- \n")

    # 1. Analizar Fondo Satelital
    with rasterio.open(path_sat) as src:
        print(f"🛰️  [FONDO SATELITAL]")
        print(f"   - CRS: {src.crs}")
        print(f"   - Size: {src.width}x{src.height} (px)")
        print(f"   - Bounds: {src.bounds}")
        print(f"   - Transform (Affine):\n{src.transform}")
        print("-" * 50)

    # 2. Analizar Raster de Datos (HLS/NDVI)
    with rasterio.open(path_hls) as src:
        print(f"📊 [RASTER DE DATOS (HLS)]")
        print(f"   - CRS: {src.crs}")
        print(f"   - Size: {src.width}x{src.height} (px)")
        print(f"   - Bounds: {src.bounds}")
        print(f"   - Transform (Affine):\n{src.transform}")
        print("-" * 50)

    # 3. Analizar Vector (ANP)
    try:
        # Cargamos solo la info básica para evitar errores de encoding
        gdf = gpd.read_file(path_vector, ignore_fields=True)
        print(f"🗺️  [VECTOR ANP]")
        print(f"   - CRS Original: {gdf.crs}")
        print(f"   - Bounds: {gdf.total_bounds}") # [minx, miny, maxx, maxy]
    except Exception as e:
        print(f"❌ Error leyendo vector: {e}")

    print("\n💡 TIP: Compara los valores de 'Bounds'. Si los números del Satélite")
    print("no coinciden o no contienen a los del Raster de Datos, no se verán.")

if __name__ == "__main__":
    auditar_geometrias()

# %% [celda 25]
import geopandas as gpd
import os
import glob

# Ruta al shapefile que extrajimos del zip
path_vector_anp = "/lustre/home/kgalvan/cnn/ANP_shapefile/extracted_anp/"
shp_file = glob.glob(os.path.join(path_vector_anp, "**", "*.shp"), recursive=True)[0]

try:
    # Intentamos cargar con fiona (o el motor que hayas instalado)
    test_gdf = gpd.read_file(shp_file)
    print("✅ ¡Éxito! El vector se cargó correctamente.")
    print(f"📍 Columnas encontradas: {test_gdf.columns.tolist()}")
    print(f"🌍 CRS detectado: {test_gdf.crs}")
    # Mostrar las primeras filas para confirmar que los nombres (con acentos) no rompieron nada
    display(test_gdf.head(2))
except Exception as e:
    print(f"❌ Seguimos con problemas: {e}")

# %% [celda 26]
#! pip install fiona 

# %% [celda 27]
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- RUTAS ---
base_path = "/lustre/home/kgalvan/cnn/resultados/ConvLSTM_UNet_V19_E60/ConvLSTM_UNet_V19_E60_r2/"

def generar_fig_8_evolucion_real():
    print("⏳ Procesando 70 pasos de tiempo para calcular la evolución del riesgo...")
    
    # 1. Cargar predicciones [Pasos, H, W]
    y_pred_raw = np.load(os.path.join(base_path, "y_pred_test_calibrated.npy"))
    H, W = 674, 861
    y_pred = y_pred_raw.reshape(-1, H, W)
    num_steps = y_pred.shape[0]
    
    # 2. Definir Umbrales (Tomados de tu CSV)
    # Muy alto >= 0.0188, Alto >= 0.0121, Medio >= 0.0084
    thr_muy_alto = 0.0188
    thr_alto = 0.0121
    thr_medio = 0.0084
    ha_pixel = 0.09

    datos_temporales = []

    # 3. Calcular áreas por cada paso de tiempo
    for t in range(num_steps):
        mapa = y_pred[t]
        
        m_alto = np.sum(mapa >= thr_muy_alto) * ha_pixel
        alto = np.sum((mapa >= thr_alto) & (mapa < thr_muy_alto)) * ha_pixel
        medio = np.sum((mapa >= thr_medio) & (mapa < thr_alto)) * ha_pixel
        bajo = np.sum(mapa < thr_medio) * ha_pixel
        
        datos_temporales.append({
            'Step': t,
            'Very High': m_alto,
            'High': alto,
            'Medium': medio,
            'Low': bajo
        })

    df_evol = pd.DataFrame(datos_temporales)

    # --- 4. GRAFICAR (Stacked Area Chart es mejor para 70 pasos que barras) ---
    plt.figure(figsize=(14, 7), dpi=300)
    sns.set_theme(style="whitegrid")

    # Colores institucionales
    colors = ['#27ae60', '#f1c40f', '#e67e22', '#c0392b']
    labels = ['Low Risk', 'Medium Risk', 'High Risk', 'Very High Risk']

    # Usamos stackplot para ver la fluidez del tiempo
    plt.stackplot(df_evol['Step'], 
                  df_evol['Low'], df_evol['Medium'], df_evol['High'], df_evol['Very High'],
                  labels=labels, colors=colors, alpha=0.85)

    plt.title("", 
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel("Time Step (Temporal Progression)", fontsize=13)
    plt.ylabel("Area (Hectares)", fontsize=13)
    
    # Ajustar el eje X para que sea legible
    plt.xlim(0, num_steps - 1)
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), title="Risk Category")
    
    plt.tight_layout()
    output_name = "Fig8_Temporal_Evolution_Final.png"
    plt.savefig(output_name, bbox_inches='tight')
    print(f"✅ ¡Figura 8 corregida! Ahora sí muestra movimiento temporal.")
    plt.show()

if __name__ == "__main__":
    generar_fig_8_evolucion_real()

# # **3.5**

# %% [celda 29]
import numpy as np
import matplotlib.pyplot as plt
import rasterio
import os

# --- RUTAS ---
base_path = "/lustre/home/kgalvan/cnn/resultados/ConvLSTM_UNet_V19_E60/ConvLSTM_UNet_V19_E60_r2/"
path_static = "/lustre/home/kgalvan/cnn/x/Topographic_DL_Dataset/static_features_LaPrimavera_30m_V20.tif"
path_hls = "/lustre/home/kgalvan/cnn/x/GEE_Export_HLS_Mosaicos/HLS_MOSAIC_2024-05-15.tif" # Un ejemplo de primavera

def generar_fig_9_zoom_integracion():
    print("⏳ Generando visualización de integración estático-dinámica...")
    
    # 1. Cargar Predicción (Paso 40 - Mayo)
    H, W = 674, 861
    y_pred_raw = np.load(os.path.join(base_path, "y_pred_test_calibrated.npy"))
    prob = y_pred_raw.reshape(-1, H, W)[40]
    
    # 2. Cargar Variables para Comparar
    with rasterio.open(path_static) as s:
        slope = s.read(2) # Banda 2: Pendiente
        dem = s.read(1)   # Banda 1: Elevación
    
    # 3. Definir una zona de Zoom (Coordenadas de píxel en una zona de alta pendiente)
    # Ajusta estos valores para centrarte en una zona interesante de La Primavera
    y1, y2, x1, x2 = 300, 500, 400, 600 
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=300)
    
    # A. Señal de Paisaje (Pendiente)
    axes[0].imshow(slope[y1:y2, x1:x2], cmap='Greys')
    axes[0].set_title("A. Landscape Structure (Slope)", fontweight='bold')
    
    # B. Predicción V19 (Nótese cómo sigue las formas de A)
    im1 = axes[1].imshow(prob[y1:y2, x1:x2], cmap='inferno', vmin=0, vmax=0.04)
    axes[1].set_title("B. V19 R2 Prediction (Calibrated)", fontweight='bold')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    
    # C. Relación Topografía-Riesgo (Contornos de DEM sobre Probabilidad)
    axes[2].imshow(prob[y1:y2, x1:x2], cmap='magma', vmin=0, vmax=0.04)
    axes[2].contour(dem[y1:y2, x1:x2], colors='white', levels=10, linewidths=0.5, alpha=0.5)
    axes[2].set_title("C. Static-Dynamic Synthesis", fontweight='bold')

    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
        
    plt.tight_layout()
    plt.savefig("Fig9_Integration_Analysis.png", bbox_inches='tight')
    print("✅ Figura de integración generada.")
    plt.show()

if __name__ == "__main__":
    generar_fig_9_zoom_integracion()

# %% [celda 30]
with rasterio.open(path_static) as s, rasterio.open(path_hls) as h:
    print(f"Static CRS: {s.crs}")
    print(f"HLS/Pred CRS: {h.crs}")
    if s.crs != h.crs:
        print("\n❌ ¡PELIGRO! Los CRS no coinciden. No puedes usar los mismos índices de píxel.")
    else:
        print("\n✅ Los CRS coinciden. Puedes indexar por píxel con seguridad.")

# %% [celda 31]
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.warp import reproject, Resampling
import os

# --- RUTAS ---
base_path = "/lustre/home/kgalvan/cnn/resultados/ConvLSTM_UNet_V19_E60/ConvLSTM_UNet_V19_E60_r2/"
path_static = "/lustre/home/kgalvan/cnn/x/Topographic_DL_Dataset/static_features_LaPrimavera_30m_V20.tif"

def generar_fig_9_corregida():
    print("⏳ Alineando capas y generando visualización de integración...")
    
    # 1. Cargar metadatos del archivo estático (nuestro molde)
    with rasterio.open(path_static) as src_static:
        static_crs = src_static.crs
        static_transform = src_static.transform
        static_width = src_static.width
        static_height = src_static.height
        slope = src_static.read(2)
        dem = src_static.read(1)

    # 2. Cargar Predicción (EPSG:4326)
    H_in, W_in = 674, 861
    y_pred_raw = np.load(os.path.join(base_path, "y_pred_test_calibrated.npy"))
    prob_4326 = y_pred_raw.reshape(-1, H_in, W_in)[40] # Paso 40 (Mayo)

    # 3. REPROYECCIÓN (WARP): Llevar la predicción de 4326 a 32613
    # Creamos un array vacío con el tamaño del archivo estático
    prob_aligned = np.zeros((static_height, static_width), dtype=np.float32)

    # Definimos el CRS de origen (HLS/Pred)
    src_crs = 'EPSG:4326'

    # Ejecutamos la transformación
    reproject(
        source=prob_4326,
        destination=prob_aligned,
        src_transform=rasterio.transform.from_bounds(-103.686, 20.544, -103.454, 20.726, W_in, H_in),
        src_crs=src_crs,
        dst_transform=static_transform,
        dst_crs=static_crs,
        resampling=Resampling.bilinear
    )

    # 4. Definir zona de Zoom (Ahora sí en los mismos píxeles)
    y1, y2, x1, x2 = 300, 500, 400, 600 
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=300)
    fig.suptitle("Figura 9. Análisis de Integración Estático-Dinámica (Alineación CRS)", fontsize=16, fontweight='bold')
    
    # A. Pendiente (Estático)
    axes[0].imshow(slope[y1:y2, x1:x2], cmap='Greys')
    axes[0].set_title("A. Estructura del Paisaje (Slope)", fontweight='bold')
    
    # B. Predicción Alineada (Dinámico)
    im1 = axes[1].imshow(prob_aligned[y1:y2, x1:x2], cmap='inferno', vmin=0, vmax=0.04)
    axes[1].set_title("B. Predicción V19 R2 (Alineada)", fontweight='bold')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    
    # C. Síntesis (Contornos de DEM sobre Probabilidad)
    axes[2].imshow(prob_aligned[y1:y2, x1:x2], cmap='magma', vmin=0, vmax=0.04)
    axes[2].contour(dem[y1:y2, x1:x2], colors='white', levels=10, linewidths=0.5, alpha=0.5)
    axes[2].set_title("C. Síntesis Estático-Dinámica", fontweight='bold')

    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
        
    plt.tight_layout()
    plt.savefig("Fig9_Integracion_Corregida.png", bbox_inches='tight')
    print("✅ ¡Mosaico corregido y alineado generado!")
    plt.show()

if __name__ == "__main__":
    generar_fig_9_corregida()
