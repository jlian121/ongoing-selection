"""
LRT power simulation under purifying selection.

Simulates alt allele frequencies under a model where frequency is constant
at or above a given age group (drop_group) and decreases linearly by a
fixed fraction per group step below it.  A logistic regression LRT is run
on each replicate to estimate statistical power across a range of initial
allele frequencies and decline rates.

Usage:
    python lrt_simulation_purifying.py \\
        --trait  TRAIT_CSV \\
        --drop-group  INT \\
        --frac  FLOAT [FLOAT ...] \\
        [--ini-freq-start FLOAT] \\
        [--ini-freq-stop  FLOAT] \\
        [--ini-freq-step  FLOAT] \\
        [--n-sim  INT] \\
        [--output-prefix  STR] \\
        [--no-plot]

Required arguments:
    --trait       Comma-separated trait file with columns:
                  ID, AGE_grp, Batch, PC1-PC10
    --drop-group  Age group at which the alt allele frequency starts to
                  decline; frequency is constant for groups >= drop-group
                  and decreases linearly for younger groups
    --frac        One or more per-step decline rates, expressed as
                  percentages of the initial frequency
                  (e.g. --frac 5 7.5 10 for 5%, 7.5%, 10% per group step)

Optional arguments:
    --ini-freq-start  Start of initial frequency range (default: 0.0005)
    --ini-freq-stop   End of initial frequency range, exclusive (default: 0.0105)
    --ini-freq-step   Step size for initial frequency range (default: 0.001)
    --n-sim           Number of simulation replicates (default: 1000)
    --output-prefix   Prefix for all output files (default: sim_purifying)
    --no-plot         Skip generating power curve plots

Outputs (all named <output-prefix>_grp<drop-group>_*):
    _log.txt              — parameters and run time
    _pwr_genomewide.csv   — power at genome-wide threshold (5e-8), rows = frac values
    _pwr_study_threshold.csv     — power at study-specific threshold (1.6e-5)
    _pval.xlsx            — raw p-values per replicate, one sheet per frac value
    _genomewideP_plot.png — power curve plot (genome-wide threshold)
    _study_thresholdP_plot.png   — power curve plot (study-specific threshold)
"""

import argparse
import time
import warnings

import numpy as np
import pandas as pd
import scipy.stats
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None

FULL_MODEL  = ("geno_binom ~ AGE_grp + Batch"
               " + PC1 + PC2 + PC3 + PC4 + PC5 + PC6 + PC7 + PC8 + PC9 + PC10")
REDUCED_MODEL = ("geno_binom ~ Batch"
                 " + PC1 + PC2 + PC3 + PC4 + PC5 + PC6 + PC7 + PC8 + PC9 + PC10")


def parse_args():
    parser = argparse.ArgumentParser(
        description="LRT power simulation under purifying selection."
    )
    parser.add_argument("--trait", required=True,
                        help="Path to trait CSV")
    parser.add_argument("--drop-group", type=int, required=True,
                        help="Age group at which frequency starts to decline")
    parser.add_argument("--frac", type=float, nargs="+", required=True,
                        help="Per-step decline rates as percentages of initial freq")
    parser.add_argument("--ini-freq-start", type=float, default=0.0005)
    parser.add_argument("--ini-freq-stop",  type=float, default=0.0105)
    parser.add_argument("--ini-freq-step",  type=float, default=0.001)
    parser.add_argument("--n-sim",          type=int,   default=1000)
    parser.add_argument("--output-prefix",  default="sim_purifying")
    parser.add_argument("--no-plot",        action="store_true")
    return parser.parse_args()


def simulate_lrt(s, ini_freq, drop_group, frac):
    """
    Simulate diploid genotypes for one variant and return the LRT p-value.
    Returns None if the design matrix is singular.

    Groups >= drop_group draw from ini_freq; younger groups draw from a
    frequency reduced by frac * ini_freq per group step away from drop_group.
    """
    diff = frac * ini_freq

    s_diff = s[s["AGE_grp"] >= drop_group].copy()
    s_diff["geno_binom"] = np.random.binomial(2, ini_freq, len(s_diff))

    for step, grp in enumerate(range(1, drop_group)[::-1], start=1):
        tmp = s[s["AGE_grp"] == grp].copy()
        frq = max(round(ini_freq - diff * step, 6), 0)
        tmp["geno_binom"] = np.random.binomial(2, frq, len(tmp))
        s_diff = pd.concat([s_diff, tmp], ignore_index=True)

    # Expand diploid genotypes to allele-level observations (two rows per individual)
    hom_ref  = s_diff[s_diff["geno_binom"] == 0]
    het      = s_diff[s_diff["geno_binom"] == 1]
    het_ref  = het.copy(); het_ref["geno_binom"] = 0
    hom_alt  = s_diff[s_diff["geno_binom"] == 2].copy(); hom_alt["geno_binom"] = 1

    df1 = pd.concat([hom_ref, hom_ref, het, het_ref, hom_alt, hom_alt],
                    ignore_index=True)

    try:
        llf_full    = smf.logit(FULL_MODEL,    data=df1).fit(disp=False).llf
        llf_reduced = smf.logit(REDUCED_MODEL, data=df1).fit(disp=False).llf
        lr = -2 * (llf_reduced - llf_full)
        return scipy.stats.chi2.sf(lr, df=1)
    except np.linalg.LinAlgError:
        return None


def run_simulation(s, ini_freq_list, drop_group, frac_list, n_sim):
    """Return (p_lst, genomewide_pwr_list, study_threshold_pwr_list)."""
    p_lst, genomewide_pwr_list, study_threshold_pwr_list = [], [], []

    for frac_pct in frac_list:
        frac = frac_pct / 100
        sim = pd.DataFrame(columns=ini_freq_list)

        for _ in range(n_sim):
            pval = [simulate_lrt(s, f, drop_group, frac) for f in ini_freq_list]
            sim.loc[len(sim)] = pval

        pwr_genomewide, pwr_study_threshold = [], []
        for col in ini_freq_list:
            valid = sim[col].dropna()
            pwr_genomewide.append((valid < 5e-8 ).sum() / len(valid))
            pwr_study_threshold.append( (valid < 1.6e-5).sum() / len(valid))

        p_lst.append(sim)
        genomewide_pwr_list.append(pwr_genomewide)
        study_threshold_pwr_list.append(pwr_study_threshold)

    return p_lst, genomewide_pwr_list, study_threshold_pwr_list


def save_results(p_lst, genomewide_pwr_list, study_threshold_pwr_list,
                 frac_list, ini_freq_list, drop_group, prefix, elapsed):
    tag = f"{prefix}_grp{drop_group}"

    with open(f"{tag}_log.txt", "w") as f:
        print(f"Initial frequencies:     {ini_freq_list}", file=f)
        print(f"Decreasing start group:  {drop_group}",   file=f)
        print(f"Decline rates (%):       {frac_list}",    file=f)
        print(f"Run time:                {elapsed:.1f}s", file=f)

    for data, fname in [
        (genomewide_pwr_list,       f"{tag}_pwr_genomewide.csv"),
        (study_threshold_pwr_list,  f"{tag}_pwr_study_threshold.csv"),
    ]:
        df = pd.DataFrame(data, index=frac_list, columns=ini_freq_list)
        df.index.name = "magnitude(%)"
        df.to_csv(fname)

    with pd.ExcelWriter(f"{tag}_pval.xlsx", engine="xlsxwriter") as writer:
        for sim, frac_pct in zip(p_lst, frac_list):
            sim.to_excel(writer, sheet_name=str(frac_pct))


def plot_power(ini_freq_list, pwr_list, frac_list, filename):
    plt.figure(figsize=(8, 7))
    for pwr, frac_pct in zip(pwr_list, frac_list):
        plt.plot(ini_freq_list, pwr, marker="o", label=f"{frac_pct}%")
    plt.xlabel("Initial frequency")
    plt.ylabel("Power")
    plt.legend()
    plt.savefig(filename, dpi=300)
    plt.close()


def main():
    args = parse_args()

    s = pd.read_csv(args.trait)
    ini_freq_list = list(np.around(
        np.arange(args.ini_freq_start, args.ini_freq_stop, args.ini_freq_step), 4
    ))

    print(f"Drop group: {args.drop_group} | "
          f"Frac (%): {args.frac} | "
          f"Frequencies: {ini_freq_list} | "
          f"Replicates: {args.n_sim}")

    start = time.time()
    p_lst, genomewide_pwr_list, study_threshold_pwr_list = run_simulation(
        s, ini_freq_list, args.drop_group, args.frac, args.n_sim
    )
    elapsed = time.time() - start

    save_results(p_lst, genomewide_pwr_list, study_threshold_pwr_list,
                 args.frac, ini_freq_list, args.drop_group, args.output_prefix, elapsed)

    if not args.no_plot:
        tag = f"{args.output_prefix}_grp{args.drop_group}"
        plot_power(ini_freq_list, genomewide_pwr_list, args.frac,
                   f"{tag}_genomewideP_plot.png")
        plot_power(ini_freq_list, study_threshold_pwr_list, args.frac,
                   f"{tag}_study_thresholdP_plot.png")

    print(f"Done. Run time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
