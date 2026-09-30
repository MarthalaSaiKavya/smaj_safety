# smaj_safety

SMAJ v4 experiments: occlusion, monitoring and steering with interpretable features of the pi0.5 VLA on LIBERO-Spatial.

## Reports

- [`reports/REPORT_full.md`](reports/REPORT_full.md) is the full report. It covers every experiment (E0–E7a), with exact numbers, their source files, the pre-specified decisions, the deviations from plan and the limitations.
- [`reports/REPORT_CoRL_safety_workshop.md`](reports/REPORT_CoRL_safety_workshop.md) is a draft for a CoRL safety workshop, framed around the safety findings. It lists the gaps to close before submission.
  - [`reports/corl_workshop/`](reports/corl_workshop/) holds the same draft converted to LaTeX (`paper.tex`) and PDF (`paper.pdf`). This is a pandoc conversion, not yet in the CoRL template.

## Results

- `smaj_v4/` is the final result set:
  - E0, E2, E3, E4b, E5 and E7 JSON files;
  - `SUMMARY.md`;
  - the config and feature sets;
  - figures and auto-generated LaTeX tables.
- `smaj_v4 2/` is the GPU-session output:
  - E0, E2, E4b and E5 JSON files, identical to those in `smaj_v4/`;
  - run logs, smoke test and step status files;
  - the pre-specified analysis config;
  - figures for E0, E2, E4 and E5.
- `*.ipynb` are the Colab notebooks with their outputs. The E6 dictionary training log is in cell 26 of `SMAJ_v4_final_experiments_(2) (2).ipynb`.
