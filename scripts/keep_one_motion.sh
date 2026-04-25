#!/usr/bin/env bash
set -euo pipefail

DIR="${1:-.}"
DRY_RUN="${DRY_RUN:-1}"   # set DRY_RUN=0 to actually delete

declare -A kept

shopt -s nullglob

for file in "$DIR"/*.pkl; do
    base="$(basename "$file" .pkl)"

    # Assumes filename format:
    #   {motion_id}_{gender_beta_key}.pkl
    # and motion_id is before the first "_"
    motion_id="${base%%_*}"

    if [[ -z "${kept[$motion_id]+x}" ]]; then
        kept[$motion_id]="$file"
        echo "[KEEP]   $file"
    else
        if [[ "$DRY_RUN" == "1" ]]; then
            echo "[DRY]    rm '$file'    # duplicate of motion_id=$motion_id, kept=${kept[$motion_id]}"
        else
            echo "[REMOVE] $file"
            rm -- "$file"
        fi
    fi
done