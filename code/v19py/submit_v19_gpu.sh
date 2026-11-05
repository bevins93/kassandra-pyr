#!/bin/bash
# ============================================================
# submit_v19_gpu.sh — Envía V19 al cluster GPU (MI210)
# Uso: bash submit_v19_gpu.sh <epochs>
# Ejemplo: bash /lustre/home/kgalvan/cnn/submit_v19_gpu.sh 60
# ============================================================

set -e

EPOCHS=$1
SCRIPT="ConvLSTM_UNet_V19.py"
SCRIPTS_DIR="/lustre/home/kgalvan/cnn/v19py"
PYTHON="/lustre/home/kgalvan/cnn/cnn/bin/python3"
LOGS_DIR="/lustre/home/kgalvan/cnn/logs"

mkdir -p "$LOGS_DIR"

if [ -z "$EPOCHS" ]; then
    echo "❌ Uso: bash submit_v19_gpu.sh <epochs>"
    echo "   Ejemplo: bash submit_v19_gpu.sh 40"
    exit 1
fi

JOB_ID=$(sbatch \
    --job-name="V19_E${EPOCHS}" \
    --partition=gpu \
    --gres=gpu:MI210:1 \
    --cpus-per-task=8 \
    --mem=256G \
    --time=10:00:00 \
    --output="$LOGS_DIR/V19_E${EPOCHS}_%j.out" \
    --error="$LOGS_DIR/V19_E${EPOCHS}_%j.err" \
    --wrap="$PYTHON $SCRIPTS_DIR/$SCRIPT --epochs $EPOCHS --use-gpu" \
    | awk '{print $NF}')

echo "=================================================="
echo "✅ Job V19 GPU enviado: V19_E${EPOCHS}"
echo "   Job ID   : $JOB_ID"
echo "   Epochs   : $EPOCHS"
echo "   Script   : $SCRIPT"
echo "   Dir      : $SCRIPTS_DIR"
echo "   Memoria  : 256G  (+16G vs V17 por rama estática)"
echo "   Tiempo   : 10:00:00  (+2h por 2022 en train)"
echo "=================================================="
echo ""
echo "📋 Monitorear:"
echo "   squeue -u \$USER"
echo "   tail -f $LOGS_DIR/V19_E${EPOCHS}_${JOB_ID}.out"
echo ""
echo "📁 Resultados en:"
echo "   /lustre/home/kgalvan/cnn/resultados/ConvLSTM_UNet_V19_E${EPOCHS}/"