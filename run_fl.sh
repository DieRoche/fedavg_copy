#!/bin/bash
#SBATCH --job-name=GS_98sparsy_comp
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=14
#SBATCH --mem=30G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%A.out
#SBATCH --error=logs/%x_%A.err

set -euo pipefail

mkdir -p logs

module load mamba
micromamba activate dieRoche

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-14}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-14}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-14}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-14}"

N_CLIENT=100
CLIENT_FRACTION=0.1
SPARSITY_RATE=0.98
N_REPEATS=3

echo "Starting GS sparsity compression experiments"
echo "n_client=${N_CLIENT}"
echo "client_fraction=${CLIENT_FRACTION}"
echo "sparsity_rate=${SPARSITY_RATE}"
echo "repeats=${N_REPEATS}"

for i in $(seq 1 "${N_REPEATS}")
do
    echo "=========================================="
    echo "Starting repeat ${i}/${N_REPEATS}: CSR"
    echo "=========================================="

    python main.py \
        --n_client "${N_CLIENT}" \
        --client_fraction "${CLIENT_FRACTION}" \
        --enable_sparse_masking \
        --sparsity_rate "${SPARSITY_RATE}" \
        --sparsity_compression CSR \
        --dynamic_quantization

    echo "Finished repeat ${i}/${N_REPEATS}: CSR"

    echo "=========================================="
    echo "Starting repeat ${i}/${N_REPEATS}: BITMSK FP32"
    echo "=========================================="

    python main.py \
        --n_client "${N_CLIENT}" \
        --client_fraction "${CLIENT_FRACTION}" \
        --enable_sparse_masking \
        --sparsity_rate "${SPARSITY_RATE}" \
        --sparsity_compression bitmask_values \
        --quantization_bits none

    echo "Finished repeat ${i}/${N_REPEATS}: BITMSK FP32"

    echo "=========================================="
    echo "Starting repeat ${i}/${N_REPEATS}: BITMSK FP16"
    echo "=========================================="

    python main.py \
        --n_client "${N_CLIENT}" \
        --client_fraction "${CLIENT_FRACTION}" \
        --enable_sparse_masking \
        --sparsity_rate "${SPARSITY_RATE}" \
        --sparsity_compression bitmask_values \
        --quantization_bits 16

    echo "Finished repeat ${i}/${N_REPEATS}: BITMSK FP16"

    echo "=========================================="
    echo "Starting repeat ${i}/${N_REPEATS}: BITMSK INT8"
    echo "=========================================="

    python main.py \
        --n_client "${N_CLIENT}" \
        --client_fraction "${CLIENT_FRACTION}" \
        --enable_sparse_masking \
        --sparsity_rate "${SPARSITY_RATE}" \
        --sparsity_compression bitmask_values \
        --quantization_bits 8

    echo "Finished repeat ${i}/${N_REPEATS}: BITMSK INT8"
done

echo "All GS sparsity compression experiments completed"
