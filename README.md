# Allele Frequency Trajectories Across Age Groups Reveal Ongoing Natural Selection Shaping Disease Susceptibility

[![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.ajhg.2026.07.002-blue)](https://doi.org/10.1016/j.ajhg.2026.07.002)
[![Journal](https://img.shields.io/badge/Journal-AJHG-orange)](https://www.cell.com/ajhg/home)
[![License: CC BY-NC-ND 4.0](https://img.shields.io/badge/License-CC%20BY--NC--ND%204.0-lightgrey)](http://creativecommons.org/licenses/by-nc-nd/4.0/)

This repository contains summary statistics and custom analysis code for:

> Chen J-L, Kang M-L, Lin C-J, Lo Y-H, Chen Y-J, Lee H-H, Tanapima V, Lin C-K, Lee I-H, Satta Y, and Ko W-Y. (2026). Allele frequency trajectories across age groups reveal ongoing natural selection shaping disease susceptibility. *The American Journal of Human Genetics* **113**, 1–19. https://doi.org/10.1016/j.ajhg.2026.07.002

---

## Repository Structure

```
ongoing-selection/
├── scripts/
│   ├── detect_frq_change_age.py          # Logistic regression for age-dependent allele frequency detection
│   ├── lrt_simulation_purifying.py       # Power simulations under purifying selection
│   ├── plot_empirical_freq.py            # Allele frequency trajectory plots
│   └── plt_empirical_frq.py              # Additional frequency visualization utilities
│
├── simulations/                          # Simulation results under purifying selection
│
├── summary_stats/
│   ├── TWB_LRT_results_all_variants.csv  # LRT p-values for all 509,817 variants
│   ├── PheWAS/                           # PheWAS results for 168 candidate variants (30 traits)
│   └── LD_iSAFE/                         # LD matrices and iSAFE scores (BRCA1, BRCA2, MLH1)
│
└── suppl_tbls/                           # Supplementary tables S1–S8
```

---

## Data Availability

Individual-level genotype and phenotype data from the Taiwan Biobank cannot be shared directly. Researchers may apply for access through the [TWB Data Access Committee](https://taiwanview.twbiobank.org.tw).

---

## Citation

If you use this code or data, please cite the paper above.
