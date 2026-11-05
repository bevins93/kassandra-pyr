# Convertido de generador_mensual_y.ipynb (sin salidas)

# %% [celda 0]
from pathlib import Path

# Definimos la ruta de tu dataset
ruta_base = Path("/lustre/home/kgalvan/cnn/y_binaria")

# 1. Conteo en la carpeta principal (no entra en subcarpetas)
conteo_tif = len(list(ruta_base.glob('*.tif'))) + len(list(ruta_base.glob('*.tiff')))

# 2. Conteo recursivo (busca en todas las subcarpetas, ej: y_binaria/clase_0/...)
conteo_total_tif = len(list(ruta_base.rglob('*.tif'))) + len(list(ruta_base.rglob('*.tiff')))

print(f"--- Reporte de archivos ---")
print(f"Imágenes .tif en carpeta raíz: {conteo_tif}")
print(f"Imágenes .tif en total (incluyendo subcarpetas): {conteo_total_tif}")

# # **diagnostico_target**
# =====================
# 
# Analiza la estructura de archivos en Y_DIR y detecta:
#   - Cuántos .tif hay en total
#   - Cuántos se están PERDIENDO por colisión de fecha en el generador actual
#   - Distribución por mes/año
#   - Cuántos píxeles positivos tiene cada archivo
# 
# Uso:
#     python diagnostico_target.py

# %% [celda 2]
import os
import re
import numpy as np
import rasterio
from datetime import datetime
from collections import defaultdict

# %% [celda 3]
# ── Ajusta esta ruta ──────────────────────────────────────────────
Y_DIR = "/lustre/home/kgalvan/cnn/y_binaria/GEE_Incendios_Binario/"
# ─────────────────────────────────────────────────────────────────

# %% [celda 4]
# ==============================
# 1. LISTAR TODOS LOS .tif
# ==============================

def find_all_tifs(directory):
    """Encuentra todos los .tif incluyendo subcarpetas."""
    tifs = []
    for root, dirs, files in os.walk(directory):
        for f in files:
            if f.endswith(".tif"):
                tifs.append(os.path.join(root, f))
    return sorted(tifs)


def parse_date(filename):
    """Extrae fecha del nombre de archivo."""
    tokens = re.findall(r"\d{4}-\d{2}-\d{2}|\d{6}", os.path.basename(filename))
    if tokens:
        t = tokens[-1]
        try:    return datetime.strptime(t, "%Y-%m-%d")
        except Exception:
            try: return datetime.strptime(t, "%Y%m")
            except Exception: return None
    return None

all_tifs = find_all_tifs(Y_DIR)
print(f"{'='*60}")
print(f"DIAGNÓSTICO DEL TARGET — Y_DIR")
print(f"{'='*60}")
print(f"Total .tif encontrados (incluyendo subcarpetas): {len(all_tifs)}\n")

# %% [celda 5]
# ==============================
# 2. DETECTAR COLISIONES POR MES
#    (lo que pierde el generador actual)
# ==============================

mes_map = defaultdict(list)   # "%Y-%m" → [lista de archivos]

for path in all_tifs:
    dt = parse_date(path)
    if dt:
        key = dt.strftime("%Y-%m")
        mes_map[key].append(path)
    else:
        print(f"  ⚠️  Sin fecha reconocible: {os.path.basename(path)}")

colisiones = {k: v for k, v in mes_map.items() if len(v) > 1}
unicos     = {k: v for k, v in mes_map.items() if len(v) == 1}

print(f"Meses únicos (el generador actual los lee bien): {len(unicos)}")
print(f"Meses con COLISIÓN (archivos que se pierden)  : {len(colisiones)}")

archivos_perdidos = sum(len(v) - 1 for v in colisiones.values())
print(f"Archivos ignorados por colisión               : {archivos_perdidos}")
print(f"  → El generador actual solo usa {len(all_tifs) - archivos_perdidos} "
      f"de {len(all_tifs)} archivos ({(len(all_tifs)-archivos_perdidos)/len(all_tifs)*100:.1f}%)\n")

if colisiones:
    print("Detalle de colisiones:")
    for mes, archivos in sorted(colisiones.items()):
        print(f"  {mes}  →  {len(archivos)} archivos:")
        for a in archivos:
            print(f"       {os.path.basename(a)}")
    print()


# %% [celda 6]
# ==============================
# 3. DISTRIBUCIÓN POR AÑO
# ==============================

print(f"{'='*60}")
print("DISTRIBUCIÓN POR AÑO")
print(f"{'='*60}")

año_map = defaultdict(list)
for path in all_tifs:
    dt = parse_date(path)
    if dt:
        año_map[dt.year].append(path)

for año in sorted(año_map.keys()):
    print(f"  {año}: {len(año_map[año]):>4} archivos")
print()

# %% [celda 7]
# ==============================
# 4. ESTADÍSTICAS DE PÍXELES POSITIVOS POR ARCHIVO
# ==============================

print(f"{'='*60}")
print("PÍXELES POSITIVOS POR ARCHIVO")
print(f"{'='*60}")

resultados = []

for path in all_tifs:
    try:
        with rasterio.open(path) as src:
            data = src.read(1).astype(np.float32)

        # Limpiar nodata
        data = np.where(np.isfinite(data), data, 0)

        total_pix = data.size
        pos_pix   = int(np.sum(data > 0))
        ratio_pct = pos_pix / total_pix * 100

        resultados.append({
            "archivo" : os.path.basename(path),
            "total"   : total_pix,
            "positivos": pos_pix,
            "ratio_pct": ratio_pct,
            "fecha"   : parse_date(path)
        })

    except Exception as e:
        print(f"  ❌ Error leyendo {os.path.basename(path)}: {e}")

# Ordenar por positivos descendente
resultados.sort(key=lambda x: x["positivos"], reverse=True)

# Resumen global
total_pos_global = sum(r["positivos"] for r in resultados)
archivos_vacios  = sum(1 for r in resultados if r["positivos"] == 0)
archivos_con_pos = sum(1 for r in resultados if r["positivos"] > 0)

print(f"Archivos con al menos 1 positivo : {archivos_con_pos}")
print(f"Archivos completamente vacíos    : {archivos_vacios}")
print(f"Total píxeles positivos global   : {total_pos_global:,}")
print(f"Promedio positivos por archivo   : {total_pos_global/len(resultados):.1f}\n")

print(f"{'Archivo':<45} {'Positivos':>10} {'Ratio':>8}  {'Fecha'}")
print("-" * 80)
for r in resultados:
    fecha_str = r["fecha"].strftime("%Y-%m-%d") if r["fecha"] else "sin fecha"
    flag = "  ✅" if r["positivos"] > 100 else ("  ⚠️ " if r["positivos"] > 0 else "  ❌")
    print(f"  {r['archivo']:<43} {r['positivos']:>10,}  {r['ratio_pct']:>6.4f}%  {fecha_str}{flag}")

# %% [celda 8]
# ==============================
# 5. RECOMENDACIÓN AUTOMÁTICA
# ==============================

print(f"\n{'='*60}")
print("RECOMENDACIÓN")
print(f"{'='*60}")

if archivos_perdidos > 0:
    print(f"⚠️  El generador actual IGNORA {archivos_perdidos} archivos por colisión de mes.")
    print(f"   → Solución: construir acumulado mensual (fusionar todos los .tif del mismo mes)")

if archivos_vacios > 0:
    print(f"⚠️  {archivos_vacios} archivos no tienen ningún píxel positivo.")
    print(f"   → Revisar si el registro CONAFOR coincide con la fecha de esos rasters")

promedio = total_pos_global / max(archivos_con_pos, 1)
if promedio < 100:
    print(f"⚠️  Promedio de {promedio:.0f} píxeles positivos por imagen es muy bajo.")
    print(f"   → Considerar acumulado mensual o buffer alrededor de los puntos CONAFOR")
else:
    print(f"✅ Promedio de {promedio:.0f} píxeles positivos por imagen es razonable.")

print(f"\n💡 Siguiente paso recomendado:")
print(f"   → Ejecutar: python construir_acumulado_mensual.py")
print(f"      (script para fusionar todos los .tif por mes en un solo raster)")

# # **Creador de Target Mensual**

# ## construir_acumulado_mensual
# ==============================
# 
# Fusiona todos los .tif de incendio del mismo mes en un solo raster binario.
# 
# Soluciona 3 problemas detectados en el diagnóstico:
#   1. Archivos con formato de fecha inconsistente (2021_01_04 vs 2022-01-04)
#   2. Generador ignoraba 173 de 260 archivos por colisión de mes
#   3. Todos los incendios del mes se acumulan → más píxeles positivos por raster
# 
# Resultado: una carpeta nueva con 1 .tif por mes donde
#            pixel = 1 si hubo incendio ese mes, 0 si no.
# 
# Uso:
#     python construir_acumulado_mensual.py

# %% [celda 11]
import os
import re
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from datetime import datetime
from collections import defaultdict

# %% [celda 12]
# ── Ajusta estas rutas ────────────────────────────────────────────
Y_DIR     = "/lustre/home/kgalvan/cnn/y_binaria/GEE_Incendios_Binario"
OUT_DIR   = "/lustre/home/kgalvan/cnn/y_binaria/GEE_Incendios_Mensual"
# ─────────────────────────────────────────────────────────────────

os.makedirs(OUT_DIR, exist_ok=True)

# %% [celda 13]
# ==============================
# 1. PARSER DE FECHA ROBUSTO
#    Reconoce ambos formatos:
#      incendio_2021_01_04_0010.tif  → 2021-01-04
#      incendio_2022-03-23_0057.tif  → 2022-03-23
# ==============================

def parse_date_robusto(filename):
    """
    Extrae fecha de nombre de archivo tolerando guiones y guiones bajos.
    Formatos soportados:
      - YYYY-MM-DD  (ej: incendio_2022-03-23_0057.tif)
      - YYYY_MM_DD  (ej: incendio_2021_01_04_0010.tif)
      - YYYYMM      (ej: incendio_202103.tif)
    """
    base = os.path.basename(filename)

    # Intentar YYYY-MM-DD
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", base)
    if m:
        try:
            return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            pass

    # Intentar YYYY_MM_DD (formato 2021)
    m = re.search(r"(\d{4})_(\d{2})_(\d{2})", base)
    if m:
        try:
            return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            pass

    # Intentar YYYYMM
    m = re.search(r"(\d{4})(\d{2})(?!\d)", base)
    if m:
        try:
            return datetime(int(m.group(1)), int(m.group(2)), 1)
        except ValueError:
            pass

    return None


# %% [celda 14]
# ==============================
# 2. ENCONTRAR TODOS LOS .tif
# ==============================

def find_all_tifs(directory):
    tifs = []
    for root, dirs, files in os.walk(directory):
        for f in sorted(files):
            if f.endswith(".tif"):
                tifs.append(os.path.join(root, f))
    return tifs


all_tifs = find_all_tifs(Y_DIR)
print(f"{'='*60}")
print(f"CONSTRUCCIÓN DE ACUMULADO MENSUAL")
print(f"{'='*60}")
print(f"Total .tif encontrados : {len(all_tifs)}")

# %% [celda 15]
# ==============================
# 3. AGRUPAR POR MES
# ==============================

mes_grupos = defaultdict(list)
sin_fecha  = []

for path in all_tifs:
    dt = parse_date_robusto(path)
    if dt:
        key = dt.strftime("%Y-%m")
        mes_grupos[key].append(path)
    else:
        sin_fecha.append(path)

print(f"Meses únicos detectados: {len(mes_grupos)}")
print(f"Archivos sin fecha     : {len(sin_fecha)}")

if sin_fecha:
    print("\n⚠️  Archivos que no se pudieron parsear:")
    for f in sin_fecha:
        print(f"   {os.path.basename(f)}")

print(f"\nDistribución por mes:")
for mes in sorted(mes_grupos.keys()):
    print(f"  {mes}: {len(mes_grupos[mes]):>3} archivos")

# %% [celda 16]
# ==============================
# 4. FUNCIÓN: ACUMULAR RASTERS DEL MES
# ==============================

def acumular_mes(lista_paths, mes_str):
    """
    Toma todos los .tif de un mes y devuelve un raster binario acumulado.
    pixel = 1 si hubo incendio en AL MENOS uno de los archivos del mes.
    """
    acumulado   = None
    perfil_base = None
    errores     = []

    for path in lista_paths:
        try:
            with rasterio.open(path) as src:
                data    = src.read(1).astype(np.float32)
                perfil  = src.profile.copy()

                # Normalizar nodata a 0
                nodata = src.nodata
                if nodata is not None:
                    data = np.where(data == nodata, 0, data)
                data = np.where(~np.isfinite(data), 0, data)

                # Binarizar: cualquier valor > 0 es incendio
                binario = (data > 0).astype(np.uint8)

            if acumulado is None:
                acumulado   = binario.copy()
                perfil_base = perfil
            else:
                # Si los tamaños no coinciden, redimensionar al del primero
                if binario.shape != acumulado.shape:
                    import warnings
                    warnings.warn(
                        f"  ⚠️  Forma inconsistente en {os.path.basename(path)}: "
                        f"{binario.shape} vs {acumulado.shape} — omitiendo"
                    )
                    continue

                # OR lógico: acumular todos los incendios del mes
                acumulado = np.maximum(acumulado, binario)

        except Exception as e:
            errores.append((path, str(e)))

    if errores:
        for p, err in errores:
            print(f"  ❌ Error en {os.path.basename(p)}: {err}")

    return acumulado, perfil_base

# %% [celda 17]
# ==============================
# 5. GENERAR UN .tif POR MES
# ==============================

print(f"\n{'='*60}")
print("GENERANDO RASTERS MENSUALES")
print(f"{'='*60}")

resumen = []

for mes_str in sorted(mes_grupos.keys()):
    lista = mes_grupos[mes_str]
    acumulado, perfil = acumular_mes(lista, mes_str)

    if acumulado is None:
        print(f"  ❌ {mes_str}: no se pudo generar (todos los archivos fallaron)")
        continue

    # Actualizar perfil para el archivo de salida
    perfil.update(
        dtype   = rasterio.uint8,
        count   = 1,
        nodata  = 255,
        compress= "lzw"
    )

    # Nombre de salida: incendio_YYYY-MM_acumulado.tif
    out_name = f"incendio_{mes_str}_acumulado.tif"
    out_path = os.path.join(OUT_DIR, out_name)

    with rasterio.open(out_path, "w", **perfil) as dst:
        dst.write(acumulado, 1)

    pos_pix   = int(np.sum(acumulado > 0))
    total_pix = acumulado.size
    ratio     = pos_pix / total_pix * 100

    resumen.append({
        "mes"       : mes_str,
        "archivos"  : len(lista),
        "positivos" : pos_pix,
        "ratio_pct" : ratio,
        "out"       : out_name
    })

    flag = "✅" if pos_pix > 100 else ("⚠️ " if pos_pix > 0 else "❌")
    print(f"  {flag} {mes_str}  |  {len(lista):>3} archivos fusionados  "
          f"|  {pos_pix:>7,} px positivos  ({ratio:.3f}%)")

# %% [celda 18]
# ==============================
# 6. RESUMEN FINAL
# ==============================

print(f"\n{'='*60}")
print("RESUMEN FINAL")
print(f"{'='*60}")

total_rasters   = len(resumen)
total_pos       = sum(r["positivos"] for r in resumen)
promedio_pos    = total_pos / max(total_rasters, 1)
meses_vacios    = sum(1 for r in resumen if r["positivos"] == 0)

print(f"Rasters mensuales generados  : {total_rasters}")
print(f"Rasters con positivos        : {total_rasters - meses_vacios}")
print(f"Rasters vacíos (0 positivos) : {meses_vacios}")
print(f"Total píxeles positivos      : {total_pos:,}")
print(f"Promedio positivos/mes       : {promedio_pos:,.0f} px")
print(f"Carpeta de salida            : {OUT_DIR}")

print(f"\n{'='*60}")
print("COMPARACIÓN ANTES / DESPUÉS")
print(f"{'='*60}")
print(f"  ANTES  — Generador usaba    :  ~87 archivos de 260  (33%)")
print(f"  DESPUÉS — Rasters mensuales :  {total_rasters} archivos (100% de la info)")
print(f"  ANTES  — Promedio positivos :  631 px/imagen")
print(f"  DESPUÉS — Promedio positivos:  {promedio_pos:,.0f} px/mes")

print(f"\n✅ Listo. Ahora actualiza Y_DIR en CNNLSTM_V7.py:")
print(f'   Y_DIR = "{OUT_DIR}"')
print(f"\n   El generador leerá directamente los acumulados mensuales.")
print(f"   No necesitas cambiar nada más en el código.")
