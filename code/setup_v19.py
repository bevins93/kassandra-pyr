#!/bin/bash
# ============================================================
# setup_v19.sh
# Crea V19 a partir de V18 con los nuevos hiperparámetros
# V18 queda intacto como referencia
#
# Cambios aplicados:
#   BATCH_TRAIN   : 8  → 16
#   POS_RATIO     : 0.18 → 0.22
#   recurrent_drop: 0.2  → 0.35
#   SpatialDropout: 0.3  → 0.45
#   alpha FT loss : 0.75 → 0.80  (beta 0.25 → 0.20)
#   patience ES   : 20   → 15
#   nombre        : V18  → V19
#
# Uso:
#   bash setup_v19.sh
# ============================================================

set -e

SRC="/lustre/home/kgalvan/cnn/v18py/ConvLSTM_UNet_V18.py"
DST_DIR="/lustre/home/kgalvan/cnn/v19py"
DST="$DST_DIR/ConvLSTM_UNet_V19.py"

echo "=================================================="
echo "  Creando V19 a partir de V18"
echo "=================================================="

# 1. Crear carpeta V19
echo ""
echo "[1] Creando directorio $DST_DIR..."
mkdir -p "$DST_DIR"
echo "    OK"

# 2. Copiar V18 → V19 (V18 queda intacto)
echo ""
echo "[2] Copiando V18 → V19..."
cp "$SRC" "$DST"
echo "    OK: $DST"

# 3. Aplicar cambios de hiperparámetros
echo ""
echo "[3] Aplicando cambios..."

# Nombre del experimento y versión
sed -i 's/ConvLSTM_UNet_V18/ConvLSTM_UNet_V19/g'           "$DST"
sed -i 's/ARCH          = "V18"/ARCH          = "V19"/g'    "$DST"
sed -i 's|v18py|v19py|g'                                    "$DST"

# Hiperparámetros de entrenamiento
sed -i 's/BATCH_TRAIN   = 8/BATCH_TRAIN   = 16/g'          "$DST"
sed -i 's/POS_RATIO     = 0.18/POS_RATIO     = 0.22/g'     "$DST"

# Arquitectura — dropout
sed -i 's/recurrent_dropout=0.2/recurrent_dropout=0.35/g'  "$DST"
sed -i 's/SpatialDropout2D(0.3/SpatialDropout2D(0.45/g'    "$DST"

# Loss — Focal Tversky
sed -i 's/alpha=0.75, beta=0.25, gamma=2.0/alpha=0.80, beta=0.20, gamma=2.0/g' "$DST"

# Callbacks — EarlyStopping
sed -i 's/patience=20, min_delta=0.0002/patience=15, min_delta=0.0002/g'       "$DST"

# JSON config — sincronizar valores registrados
sed -i 's/"patience": 20, "min_delta": 0.0002/"patience": 15, "min_delta": 0.0002/g'                    "$DST"
sed -i 's/"pos_ratio": POS_RATIO, "learning_rate": 2e-5/"pos_ratio": 0.22, "learning_rate": 2e-5/g'     "$DST"
sed -i 's/FocalTversky(alpha=0.75,beta=0.25,gamma=2.0)/FocalTversky(alpha=0.80,beta=0.20,gamma=2.0)/g'  "$DST"

echo "    OK todos los cambios aplicados"

# 4. Verificar sintaxis Python
echo ""
echo "[4] Verificando sintaxis Python..."
python3 -m py_compile "$DST" && echo "    OK sintaxis correcta" || echo "    ERROR de sintaxis"

# 5. Verificar que V18 quedó intacto
echo ""
echo "[5] Verificando V18 intacto..."
V18_BATCH=$(grep "BATCH_TRAIN" "$SRC" | grep -v "#" | head -1)
V18_ARCH=$(grep 'ARCH.*=.*"V' "$SRC" | grep -v "#" | head -1)
echo "    $V18_BATCH"
echo "    $V18_ARCH"

# 6. Mostrar diferencias clave
echo ""
echo "=================================================="
echo "  Comparacion V18 vs V19"
echo "=================================================="
echo ""
printf "  %-25s %-20s %-20s\n" "Parametro" "V18" "V19"
printf "  %-25s %-20s %-20s\n" "─────────────────────────" "────────────────────" "────────────────────"

for param in "BATCH_TRAIN" "POS_RATIO" "recurrent_dropout" "SpatialDropout2D"; do
    v18_val=$(grep -m1 "$param" "$SRC" 2>/dev/null | grep -v "#" | sed 's/.*= *//' | tr -d ' ')
    v19_val=$(grep -m1 "$param" "$DST" 2>/dev/null | grep -v "#" | sed 's/.*= *//' | tr -d ' ')
    printf "  %-25s %-20s %-20s\n" "$param" "$v18_val" "$v19_val"
done

echo ""
echo "=================================================="
echo "  V18 conservado en : $SRC"
echo "  V19 creado en     : $DST"
echo "=================================================="
echo ""
echo "Para lanzar V19:"
echo "  bash /lustre/home/kgalvan/cnn/submit_v19_gpu.sh 60"
echo ""