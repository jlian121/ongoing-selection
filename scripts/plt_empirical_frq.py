#!/usr/bin/env python3
"""
plt_empirical_frq.py — Plot empirical allele-frequency trajectories across age groups.

DESCRIPTION
-----------
Reads genotype calls from a VCF file, merges them with a per-individual trait
table, and plots allele frequency ± SE for each requested SNP across ordered
age groups.  Optionally overlays logistic-regression–predicted frequencies that
control for sex, genotyping batch, and the first 10 principal components (PCs).

USAGE (command-line)
--------------------
    plt_empirical_frq.py --vcf <VCF> --trait <TRAIT_CSV> --snps <rsID [...]>
                          --output <OUT_FILE>
                          [--age-grp-col AGE_grp]
                          [--predict] [--reg-tbl <REG_CSV>]
                          [--n-col N] [--n-row N]
                          [--color COLOR]
                          [--xylab]

It can also be imported as a module; the public API is:
    get_freq_se_list(vcf, trait_tbl, age_grp_col, out_list, predict,
                     select_list, cal_p=False)
    plot_freq(data_list, n_col, n_row, predict=False, reg_tbl=None, ...)
    find_trend(vcf, trait_tbl, age_grp_col)
    plot_freq_two(list_1, list_2, n_col, n_row, ...)

INPUT FILES
-----------
--vcf   VCF file (tab-separated, standard VCF format).
        Required columns (in order): CHROM, POS, ID (rsID), REF, ALT,
        followed by one column per individual (sample ID as column header).
        Genotypes must be diploid (0/0, 0/1, 1/1) or haploid (0, 1)
        for hemizygous regions.
        Missing / non-variant genotypes are silently dropped per SNP.


--trait Comma-separated (CSV) trait table, one row per individual.
        Required columns
          ID        — individual ID matching VCF sample header
          <age_grp> — integer age-group label (default column name: AGE_grp)
                      Groups are treated as ordered integers (e.g. 1–8
                      mapping to successive 5-year bins starting at age 30).
        Required only when --predict is set
          SEX       — sex indicator (any two distinct values)
          Batch     — genotyping batch
          PC1–PC10  — the first 10 principal components
        All other columns are ignored.


--reg-tbl  (optional) Comma-separated regression-output table.
        Used to annotate each subplot with an empirical p-value and, when
        available, the nearest gene name.
        Required columns
          ID        — rsID (matched against the VCF ID field)
          ChrPosID  — fallback identifier in "chrCHROM_POS" format
          pval      — association p-value (displayed as scientific notation)
        Optional column
          gene      — nearest gene(s); semicolon-separated; "intergenic" → rsID only


--snps  One or more rsIDs to plot (space-separated on the command line).
        Only VCF records whose ID field matches an entry in this list are
        processed; all others are skipped.

EXAMPLE
--------
    # Six SNPs arranged in a 2×3 grid, no prediction, plain blue
    plt_empirical_frq.py \\
        --vcf   variants.vcf \\
        --trait traits.csv \\
        --snps  rs1 rs2 rs3 rs4 rs5 rs6 \\
        --n-col 3 --n-row 2 \\
        --output six_snps.png
"""

import pandas as pd
import copy
import numpy as np
import math
import scipy
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

import contextlib
import sys
import os

plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.it'] = 'DejaVu Sans:italic'
plt.rcParams['mathtext.rm'] = 'DejaVu Sans:italic'
plt.rcParams['font.cursive'] = ['DejaVu Sans']

    
    

def get_freq_se_list(vcf, trait_tbl, age_grp_col, out_list, predict, select_list, cal_p=False):

    """
    vcf: a vcf file
    trait_tbl: a table contains traits (e.g. batch, age group, PCs ...)
    age_grp_col: the name of "age group" column, a string
    predict: True if PCs are available, otherwise False
    """

    trait = trait_tbl
    age_grp = age_grp_col

    with(open(vcf, 'r')) as file:
        for line in file:
            if line.startswith("##"):
                continue
            elif line.startswith("#CHROM"):
                head = line.strip().split('\t')
            else:
                snp_i_vcf = line.strip().split('\t')
                snpID = 'chr' + str(snp_i_vcf[0]) + '_' + str(snp_i_vcf[1])
                rsID = snp_i_vcf[2]
                if rsID not in select_list:
                    continue


                df = pd.DataFrame({
                    "ID": head,
                    "genotype": snp_i_vcf,
                    })

                
                conditions = [
                    (df['genotype'] == '0/0'),
                    (df['genotype'] == '0/1'),
                    (df['genotype'] == '1/1')]
                choices = [0, 1, 2]

                df['geno_binom'] = np.select(conditions, choices, default=-1)

                miss = df[df["geno_binom"] == -1].index
                df_f = df.drop(miss)
                df_f = df_f.reset_index(drop = True)



                df_f_mrg = pd.merge(df_f, trait, left_on="ID", right_on="ID", how = "inner")

                condition = df_f_mrg['geno_binom'] == 0
                tmp1 = df_f_mrg[condition]
            
                condition = df_f_mrg['geno_binom'] == 1
                tmp2 = df_f_mrg[condition]
                tmp2_1 = copy.deepcopy(tmp2)
                tmp2_1.loc[tmp2_1['geno_binom'] == 1, 'geno_binom'] = 0 

                condition = df_f_mrg['geno_binom'] == 2
                tmp3 = df_f_mrg[condition]
                tmp3.loc[tmp3['geno_binom'] == 2, 'geno_binom'] = 1 
                frames = [tmp1, tmp1, tmp2, tmp2_1, tmp3, tmp3]
                

                df_m = df[(df['genotype'] == '0') | (df['genotype'] == '1')]
                df_m.loc[:, 'geno_binom'] = df_m['genotype'].astype(int)
                df_m_mrg = pd.merge(df_m, trait, left_on="ID", right_on="ID", how = "inner")

                frames.append(df_m_mrg)


                df1 = pd.DataFrame()
                df1 = pd.concat(frames, ignore_index=True)
                
                n_haplotypes = []
                n_allele = []
                freq = []
                se = []
                expect = []
                
                for i in list(set(df1[age_grp])):
                    temp = df1[df1[age_grp] == i]
                    total_hap = len(temp)
                    n_haplotypes.append(total_hap)
                    n = sum(temp["geno_binom"])
                    n_allele.append(n)
                    p = n/total_hap
                    freq.append(p)
                    s = math.sqrt(p*(1-p)/len(temp))
                    se.append(s)


                if predict == True:
                    if len(set(df1['SEX'])) == 2:
                        try:
                            lgr = smf.logit("geno_binom ~ SEX + Batch + PC1 + PC2 + PC3 + PC4 + PC5 + PC6 + PC7 + PC8 + PC9 + PC10", data=df1).fit()
                            if cal_p == True:
                                lgr_age = smf.logit("geno_binom ~ AGE_grp + SEX + Batch + PC1 + PC2 + PC3 + PC4 + PC5 + PC6 + PC7 + PC8 + PC9 + PC10", data=df1).fit()
                                w_age = lgr_age.llf
                                w_o_age = lgr.llf
                                likelihood_ratio = -2*(w_o_age - w_age)
                                p = scipy.stats.chi2.sf(likelihood_ratio, df=1)
                            for i in list(set(df1[age_grp])):
                                temp = df1[df1[age_grp] == i]
                                xtest = temp[['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10', "Batch", "SEX"]]
                                yhat = lgr.predict(xtest)
                                temp["predict"] = yhat
                                exp = sum(temp["predict"])/len(temp)
                                expect.append(exp)  

                        except np.linalg.LinAlgError:
                            expect = [0.0005]*8        

                    else:
                        try:
                            lgr = smf.logit("geno_binom ~ Batch + PC1 + PC2 + PC3 + PC4 + PC5 + PC6 + PC7 + PC8 + PC9 + PC10", data=df1).fit()
                            if cal_p == True:
                                lgr_age = smf.logit("geno_binom ~ AGE_grp + Batch + PC1 + PC2 + PC3 + PC4 + PC5 + PC6 + PC7 + PC8 + PC9 + PC10", data=df1).fit()
                                w_age = lgr_age.llf
                                w_o_age = lgr.llf
                                likelihood_ratio = -2*(w_o_age - w_age)
                                p = scipy.stats.chi2.sf(likelihood_ratio, df=1)

                            for i in list(set(df1[age_grp])):
                                temp = df1[df1[age_grp] == i]
                                xtest = temp[['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10', "Batch"]]
                                yhat = lgr.predict(xtest)
                                temp["predict"] = yhat
                                exp = sum(temp["predict"])/len(temp)
                                expect.append(exp) 
                        except np.linalg.LinAlgError:  
                            expect = [0.0005]*8                

                else:
                    expect.append('None')

                if cal_p == True:
                    out_list.append([snpID, rsID, freq, se, expect, n_allele, n_haplotypes, p])
                else:
                    out_list.append([snpID, rsID, freq, se, expect, n_allele, n_haplotypes])


    return out_list


                



def plot_freq(data_list, n_col, n_row, predict=False, reg_tbl=None, Color = 'steelblue', height_times=2.5, width_times=2.8, xylab=False, filename=False):
    """
    data_list: the list output by 'get_freq_se_list' function
    n_col: number of columns
    n_row: number of rows
    predict: True if predict = True in get_freq_se_list
    reg_tbl: the output table of logistic regression, if available
    height_times: height scale (float)
    width_times: width scale (float)
    """
    n = 0
    m = 0
    column_num = n_col
    row_num = n_row
    height = row_num*height_times
    width = column_num*width_times
    fig, axes = plt.subplots(ncols=column_num, nrows=row_num, constrained_layout=True, figsize=(width, height), dpi=300, squeeze=False)

    empirical_p = None

    for i in range(len(data_list)):
        snpID = data_list[i][0]
        rsID = data_list[i][1]
        freq_list = data_list[i][2]
        se_list = data_list[i][3]

        if reg_tbl is None:
            pass
        else:
            if rsID in reg_tbl['ID'].to_list():
                snp_info_1 = reg_tbl[reg_tbl['ID'] == rsID]
                p_val = snp_info_1['pval'].iloc[0,]
            else:
                snp_info_1 = reg_tbl[reg_tbl['ChrPosID'] == snpID]
                p_val = snp_info_1['pval'].iloc[0,]
            empirical_p = str(format(p_val, '.1E'))

            if 'gene' in snp_info_1.columns:
                gene = snp_info_1['gene'].iloc[0,]
                if gene == 'intergenic':
                    axes[n][m].set_title(rsID)
                else:
                    if len(gene.split(';')) > 1:
                        gene = gene.split(';')[0]
                    axes[n][m].set_title(f"{rsID} (${gene}$)")
            else:
                axes[n][m].set_title(rsID)

        age_grp = list(range(1, len(data_list[i][2])+1))

        axes[n][m].errorbar(age_grp, freq_list, yerr=se_list, marker = "o", markersize=5, color = Color)

        ratio = 0.92
        x_left, x_right = axes[n][m].get_xlim()
        y_low, y_high = axes[n][m].get_ylim()
        axes[n][m].set_aspect(abs((x_right-x_left)/(y_low-y_high))*ratio)

        axes[n][m].set_xticks(list(np.arange(1.5, 8, 1)))
        axes[n][m].set_xticklabels(list(range(30, 65, 5)))

        if predict == True:
            expect = data_list[i][4]
            axes[n][m].plot(age_grp, expect, linestyle='dashed', color = "gray")

        if empirical_p is not None:
            max_index = freq_list.index(max(freq_list))
            if freq_list[0] > freq_list[len(freq_list)-2]:
                axes[n][m].text(x=8, y=max(freq_list)+se_list[max_index], s = 'p = ' + empirical_p, ha = 'right', va = 'top')
            else:
                axes[n][m].text(x=1, y=max(freq_list)+se_list[max_index], s = 'p = ' + empirical_p, ha = 'left', va = 'top')
        
        if xylab == True:
            axes[n][m].set_xlabel('Age', fontsize=11)
            axes[n][m].set_ylabel('Frequency', fontsize=11)
        else:
            pass

        m += 1 
        if m > (column_num-1):
            n += 1
            m = 0

    if filename == False:
        plt.show()
    else:
        plt.savefig(filename, dpi=300)




def find_trend(vcf, trait_tbl, age_grp_col):
    '''
    vcf: a vcf file
    trait_tbl: a table contains traits (e.g. batch, age group, PCs ...)
    age_grp_col: the name of "age group" column, a string
    This function recognizes the empirical freq trend of each snp, and group them in 'down', 'up', or 'U' lists.
    Output is a list: [[down], [up], # [U]]
    '''
    trend_list = []

    trait = trait_tbl
    age_grp = age_grp_col

    up = []
    down = []

    with(open(vcf, 'r')) as file:
        for line in file:
            if line.startswith("##"):
                continue
            elif line.startswith("#CHROM"):
                head = line.strip().split('\t')
            else:
                snp_i_vcf = line.strip().split('\t')
                snpID = 'chr' + str(snp_i_vcf[0]) + '_' + str(snp_i_vcf[1])
                rsID = snp_i_vcf[2]


                df = pd.DataFrame({
                    "ID": head,
                    "genotype": snp_i_vcf,
                    })

                
                conditions = [
                    (df['genotype'] == '0/0'),
                    (df['genotype'] == '0/1'),
                    (df['genotype'] == '1/1')]
                choices = [0, 1, 2]

                df['geno_binom'] = np.select(conditions, choices, default=-1)

                miss = df[df["geno_binom"] == -1].index
                df_f = df.drop(miss)
                df_f = df_f.reset_index(drop = True)



                df_f_mrg = pd.merge(df_f, trait, left_on="ID", right_on="ID", how = "inner")

                condition = df_f_mrg['geno_binom'] == 0
                tmp1 = df_f_mrg[condition]
            
                condition = df_f_mrg['geno_binom'] == 1
                tmp2 = df_f_mrg[condition]
                tmp2_1 = copy.deepcopy(tmp2)
                tmp2_1.loc[tmp2_1['geno_binom'] == 1, 'geno_binom'] = 0 

                condition = df_f_mrg['geno_binom'] == 2
                tmp3 = df_f_mrg[condition]
                tmp3.loc[tmp3['geno_binom'] == 2, 'geno_binom'] = 1 
                frames = [tmp1, tmp1, tmp2, tmp2_1, tmp3, tmp3]
                

                df_m = df[(df['genotype'] == '0') | (df['genotype'] == '1')]
                df_m.loc[:, 'geno_binom'] = df_m['genotype'].astype(int)
                df_m_mrg = pd.merge(df_m, trait, left_on="ID", right_on="ID", how = "inner")

                frames.append(df_m_mrg)

                df1 = pd.DataFrame()
                df1 = pd.concat(frames, ignore_index=True)


                grp_list = list(set(df1[age_grp]))
                first_grp = df1[df1[age_grp] == grp_list[0]]
                last_grp = df1[df1[age_grp] == grp_list[-1]]

                first_grp_frq = sum(first_grp['geno_binom'])/len(first_grp)
                last_grp_frq = sum(last_grp['geno_binom'])/len(last_grp)

                if first_grp_frq > last_grp_frq:
                    down.append(rsID)
                else:
                    up.append(rsID)

    trend_list = [down, up]
    return trend_list



def plot_freq_two(list_1, list_2, n_col, n_row, predict=False, reg_tbl_1=None, reg_tbl_2=None, height_times=2.5, width_times=2.8, col_1=None, col_2=None, SE=1, filename=False):
    """
    data_list: the list output by 'get_freq_se_list' function
    n_col: number of columns
    n_row: number of rows
    predict: True if predict = True in get_freq_se_list
    reg_tbl: the output table of logistic regression, if available
    height_times: height scale (float)
    width_times: width scale (float)
    """
    if col_1 is None:
        col_1 = ['steelblue', 'cadetblue']
    if col_2 is None:
        col_2 = ['firebrick', 'rosybrown']
    n = 0
    m = 0
    column_num = n_col
    row_num = n_row
    height = row_num*height_times
    width = column_num*width_times
    fig, axes = plt.subplots(ncols=column_num, nrows=row_num, constrained_layout=True, figsize=(width, height), dpi=300, squeeze=False)

    empirical_p_1 = None
    empirical_p_2 = None

    for i in range(len(list_1)):
        snpID = list_1[i][0]
        rsID = list_1[i][1]
        freq_list_1 = list_1[i][2]
        freq_list_2 = list_2[i][2]
        se_list_1 = list_1[i][3]
        se_list_2 = list_2[i][3]

        if len(list_1[i]) == 8:
            pval_1 = list_1[i][7]
            empirical_p_1 = str(format(pval_1, '.1E'))
        if len(list_1[i]) == 8:
            pval_2 = list_2[i][7]
            empirical_p_2 = str(format(pval_2, '.1E'))

        if reg_tbl_1 is None:
            axes[n][m].set_title(rsID)
        else:
            snp_info_1 = reg_tbl_1[reg_tbl_1['ID'] == rsID]

            if reg_tbl_2 != None:
                snp_info_2 = reg_tbl_2[reg_tbl_2['ID'] == rsID]

            if 'gene' in snp_info_1.columns:
                gene = snp_info_1['gene'].iloc[0,]
                if gene == 'intergenic':
                    axes[n][m].set_title(rsID)
                else:
                    if len(gene.split(';')) > 1:
                        gene = gene.split(';')[0]
                    axes[n][m].set_title(f"{rsID} (${gene}$)")
            else:
                axes[n][m].set_title(rsID)

        age_grp = list(range(1, len(list_1[i][2])+1))

        axes[n][m].errorbar(age_grp, freq_list_1, yerr=[e*SE for e in se_list_1], marker = "o", markersize=5, color = col_1[0])
        axes[n][m].errorbar(age_grp, freq_list_2, yerr=[e*SE for e in se_list_2], marker = "o", markersize=5, color = col_2[0])

        ratio = 0.92
        x_left, x_right = axes[n][m].get_xlim()
        y_low, y_high = axes[n][m].get_ylim()
        axes[n][m].set_aspect(abs((x_right-x_left)/(y_low-y_high))*ratio)

        axes[n][m].set_xticks(list(np.arange(1.5, 8, 1)))
        axes[n][m].set_xticklabels(list(range(30, 65, 5)))
        axes[n][m].set_xlabel('Age')
        axes[n][m].set_ylabel('Allele frequency')
        if predict == True:
            if list_1[i][4] != ['None']:
                expect_m = list_1[i][4]
                axes[n][m].plot(age_grp, expect_m, linestyle='dashed', color = col_1[1])
            if list_2[i][4] != ['None']:
                expect_f = list_2[i][4]
                axes[n][m].plot(age_grp, expect_f, linestyle='dashed', color = col_2[1])
          

        if (empirical_p_1 is not None) & (empirical_p_2 is not None):
            if max(freq_list_1) > max(freq_list_2):
                frq_list_max = freq_list_1
                se_list_max = se_list_1
            else:
                frq_list_max = freq_list_2
                se_list_max = se_list_2
            max_index = frq_list_max.index(max(frq_list_max))
            highest_point = max(frq_list_max)+se_list_max[max_index]

            if min(freq_list_1) < min(freq_list_2):
                frq_list_min = freq_list_1
                se_list_min = se_list_1
            else:
                frq_list_min = freq_list_2
                se_list_min = se_list_2
            min_index = frq_list_min.index(min(frq_list_min))
            lowest_point = min(frq_list_min) - se_list_min[min_index]
        
            total_y_height = highest_point - lowest_point

            if frq_list_max[0] > frq_list_max[len(frq_list_max)-2]:
                axes[n][m].text(x=8, y=highest_point, s = 'p = ' + empirical_p_1, ha = 'right', va = 'top', color=col_1[0])
                axes[n][m].text(x=8, y=highest_point-total_y_height/10, s = 'p = ' + empirical_p_2, ha = 'right', va = 'top', color=col_2[0])
            else:
                axes[n][m].text(x=1, y=highest_point, s = 'p = ' + empirical_p_1, ha = 'left', va = 'top', color=col_1[0])
                axes[n][m].text(x=1, y=highest_point-total_y_height/10, s = 'p = ' + empirical_p_2, ha = 'left', va = 'top', color=col_2[0])
        else:
            pass

        m += 1
        if m > (column_num-1):
            n += 1
            m = 0
    if filename == False:
        plt.show()
    else:
        plt.savefig(filename, dpi=300)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        prog='plt_empirical_frq.py',
        description='Plot empirical allele-frequency trajectories across age groups.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--vcf', required=True,
                        help='VCF file with genotype calls (one SNP per data line).')
    parser.add_argument('--trait', required=True,
                        help='CSV trait table with columns ID, <age-grp-col>, '
                             'and optionally SEX, Batch, PC1–PC10 (required for --predict).')
    parser.add_argument('--snps', required=True, nargs='+', metavar='rsID',
                        help='One or more rsIDs to extract and plot.')
    parser.add_argument('--output', required=True,
                        help='Output file path (PDF or PNG; extension determines format).')
    parser.add_argument('--age-grp-col', default='AGE_grp', metavar='COL',
                        help='Column name for age group in the trait table (default: AGE_grp).')
    parser.add_argument('--predict', action='store_true',
                        help='Overlay logistic-regression–predicted frequencies '
                             '(requires SEX, Batch, PC1–PC10 in the trait table).')
    parser.add_argument('--reg-tbl', default=None, metavar='CSV',
                        help='Regression output CSV; adds p-value annotations and gene labels.')
    parser.add_argument('--n-col', type=int, default=1, metavar='N',
                        help='Number of subplot columns (default: 1).')
    parser.add_argument('--n-row', type=int, default=1, metavar='N',
                        help='Number of subplot rows (default: 1).')
    parser.add_argument('--color', default='steelblue',
                        help='Matplotlib color for the frequency line (default: steelblue).')
    parser.add_argument('--xylab', action='store_true',
                        help='Add "Age" / "Frequency" axis labels to each subplot.')

    args = parser.parse_args()

    trait_tbl = pd.read_csv(args.trait, header=0)

    reg_tbl = None
    if args.reg_tbl is not None:
        reg_tbl = pd.read_csv(args.reg_tbl, header=0)

    data_list = []
    data_list = get_freq_se_list(
        args.vcf, trait_tbl, args.age_grp_col,
        data_list, args.predict, args.snps,
    )

    if not data_list:
        sys.exit(f'No matching SNPs found in {args.vcf} for: {args.snps}')

    plot_freq(
        data_list,
        n_col=args.n_col,
        n_row=args.n_row,
        predict=args.predict,
        reg_tbl=reg_tbl,
        Color=args.color,
        xylab=args.xylab,
        filename=args.output,
    )


if __name__ == '__main__':
    main()
