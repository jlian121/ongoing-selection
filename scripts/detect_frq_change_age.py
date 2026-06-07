"""
Detect variants whose allele frequency changes with age group.

For each variant in a VCF, diploid genotypes are expanded to allele-level
observations (one row per allele copy) so that a logistic regression can
model allele carrier status as a binary outcome.  A likelihood-ratio test
(LRT) then compares a full model (AGE_grp + covariates) against a reduced
model (covariates only) to determine whether age group is associated with
allele frequency.  Variants that cannot be tested — due to a fixed allele
frequency or a singular design matrix — are written to a separate problem
file.

Usage:
    python detect_frq_change_age.py <input.vcf> <traits.csv> <output_file>

Arguments:
    input.vcf   — single-sample or multi-sample VCF (uncompressed)
    traits.csv  — comma-separated covariate file with columns:
                  ID, AGE_grp, SEX, Batch, PC1–PC10
    output_file — full path to the output file for successfully tested variants

Outputs:
    <output_file>             — successfully tested variants with OR, LR, p-value
    <output_stem>_problem.ext — variants that could not be tested
"""

import os
import pandas as pd
import copy
import numpy as np
import statsmodels.formula.api as smf
import scipy
from scipy import stats
import warnings
import time
import sys
warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None  # default='warn'




start_time = time.time()

vcf         = sys.argv[1]
traits_csv  = sys.argv[2]
output_file = sys.argv[3]

trait = pd.read_table(traits_csv, header = 0, sep = ',')

# Accumulate results for variants that pass the LRT
info = pd.DataFrame(columns=['CHROM', 'POS', 'ID', 'REF', 'ALT'])
pval = []
freq = []
odd_ratio = []
LR = []

# Accumulate results for variants that fail (fixed freq or singular matrix)
problem = pd.DataFrame(columns=['CHROM', 'POS', 'ID', 'REF', 'ALT'])
pro_pval = []
pro_freq = []
pro_OR = []
pro_LR = []


with open(vcf, 'r') as file:
    for line in file:
        if line.startswith("##"):
            continue
        elif line.startswith("#CHROM"):
            head = line.strip().split('\t')
        else:
            row = line.strip().split('\t')
            # Build a per-sample genotype table for this variant
            df = pd.DataFrame({
                "ID": head,
                "genotype": row,
                })

            # Convert diploid GT strings to allele-dose integers (0/1/2);
            # any other value (missing, multi-allelic, phased) is flagged -1
            conditions = [
                (df['genotype'] == '0/0'),
                (df['genotype'] == '0/1'),
                (df['genotype'] == '1/1')]
            choices = [0, 1, 2]

            df['geno_binom'] = np.select(conditions, choices, default=-1)

            # Drop samples with missing or unusable genotypes
            miss = df[df["geno_binom"] == -1].index
            df_f = df.drop(miss)
            df_f = df_f.reset_index(drop = True)

            # Merge with phenotype/covariate table on sample ID
            df_f_mrg = pd.merge(df_f, trait, left_on="ID", right_on="ID", how = "inner")

            # --- Allele-level expansion for diploid samples ---
            # Each diploid individual contributes two independent allele observations.
            # Homozygous ref (0/0): two ref alleles → duplicate the row twice with geno_binom=0
            condition = df_f_mrg['geno_binom'] == 0
            tmp1 = df_f_mrg[condition]

            # Heterozygous (0/1): one ref allele (geno_binom=0) and one alt allele (geno_binom=1)
            condition = df_f_mrg['geno_binom'] == 1
            tmp2 = df_f_mrg[condition]           # alt allele copy (keep geno_binom=1)
            tmp2_1 = copy.deepcopy(tmp2)
            tmp2_1['geno_binom'] = 0             # ref allele copy

            # Homozygous alt (1/1): two alt alleles → recode to 1 and duplicate twice
            condition = df_f_mrg['geno_binom'] == 2
            tmp3 = df_f_mrg[condition]
            tmp3['geno_binom'] = 1

            # Each genotype class appears twice in frames to yield two allele rows per sample
            frames = [tmp1, tmp1, tmp2, tmp2_1, tmp3, tmp3]


            # Handle hemizygous calls on chrX in males (GT is '0' or '1', not diploid)
            df_m = df[(df['genotype'] == '0') | (df['genotype'] == '1')]
            df_m['geno_binom'] = df_m['genotype'].astype(int)
            df_m_mrg = pd.merge(df_m, trait, left_on="ID", right_on="ID", how = "inner")

            frames.append(df_m_mrg)

            # Combine all allele-level observations into one dataframe
            df1 = pd.DataFrame()
            df1 = pd.concat(frames, ignore_index=True)


            OR = "o"
            # Skip variants where the allele is fixed (freq = 0 or 1); LRT is undefined
            if (sum(df1["geno_binom"])/len(df1) == 0) | (sum(df1["geno_binom"])/len(df1) == 1):
                problem.loc[len(problem)] = row[0:5]
                pro_freq.append(sum(df1["geno_binom"])/len(df1))
                pro_pval.append("X")
                pro_OR.append("X")
                pro_LR.append("X")


            else:
                try:
                    # Full model: allele ~ AGE_grp + covariates
                    lgs1 = smf.logit("geno_binom ~ AGE_grp + SEX + Batch + PC1 + PC2 + PC3 + PC4 + PC5 + PC6 + PC7 + PC8 + PC9 + PC10", data=df1).fit()
                    OR = lgs1.params[1]   # log-OR for AGE_grp

                    # Reduced model: allele ~ covariates (no age term)
                    lgs2 = smf.logit("geno_binom ~ SEX + Batch + PC1 + PC2 + PC3 + PC4 + PC5 + PC6 + PC7 + PC8 + PC9 + PC10", data=df1).fit()
                    odd_ratio.append(OR)

                    # LRT: 2 * (log-likelihood difference), chi-squared with 1 df
                    w_age = lgs1.llf
                    w_o_age = lgs2.llf
                    likelihood_ratio = -2*(w_o_age - w_age)
                    p = scipy.stats.chi2.sf(likelihood_ratio, df=1)
                    pval.append(p)
                    info.loc[len(info)] = row[0:5]
                    freq.append(sum(df1["geno_binom"])/len(df1))
                    LR.append(likelihood_ratio)

                except np.linalg.LinAlgError:
                    # Singular matrix — design matrix is not invertible for this variant
                    problem.loc[len(problem)] = row[0:5]
                    pro_freq.append(sum(df1["geno_binom"])/len(df1))
                    pro_pval.append("-")
                    if OR == "o":
                        pro_OR.append("-")
                    else:
                        pro_OR.append(OR)
                    pro_LR.append('-')





info["freq"] = freq
info["odd_ratio"] = odd_ratio
info["LR"] = LR
info["pval"] = pval

problem["freq"] = pro_freq
problem["odd_ratio"] = pro_OR
problem["pval"] = pro_pval
problem["LR"] = pro_LR

base, ext = os.path.splitext(output_file)
info_out    = output_file
problem_out = base + '_problem' + ext

info.to_csv(info_out, header = True, index=False, sep='\t')
problem.to_csv(problem_out, header = True, index=False, sep='\t')


print("run_time:", time.time()-start_time)
