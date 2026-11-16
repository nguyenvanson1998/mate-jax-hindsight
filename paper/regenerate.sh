#!/usr/bin/env bash
# Regenerate the tables (paper/generated/*.tex) and figures (paper/figures/*.pdf) from results/.
#
# The three generator scripts are kept exactly as they were used for the paper. They expect the layout
#     <work>/paper-jnca/make_*.py   and   <work>/mate/results/
# so this wrapper builds that layout in a temporary folder (with a link to results/), runs the scripts
# there and copies the outputs back to paper/generated and paper/figures.
#
#     bash paper/regenerate.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT

mkdir -p "${WORK}/paper-jnca" "${WORK}/mate"
cp "${ROOT}"/paper/make_figures.py "${ROOT}"/paper/make_figures_v2.py "${ROOT}"/paper/make_simpat_additions.py "${WORK}/paper-jnca/"
ln -s "${ROOT}/results" "${WORK}/mate/results"

cd "${WORK}"
"${PYTHON}" paper-jnca/make_figures.py          # tables of the first manuscript version (shared helpers)
"${PYTHON}" paper-jnca/make_figures_v2.py       # main tables, macros and figures
"${PYTHON}" paper-jnca/make_simpat_additions.py # paired statistics, bounds, latency, sensitivity (needs v2 output)

mkdir -p "${ROOT}/paper/generated" "${ROOT}/paper/figures"
cp -R "${WORK}/paper-jnca/generated/." "${ROOT}/paper/generated/"
cp -R "${WORK}/paper-jnca/figures/." "${ROOT}/paper/figures/"
echo "Tables: ${ROOT}/paper/generated   Figures: ${ROOT}/paper/figures"
