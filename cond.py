'''
Author: Kesar-Chi && momiji0613@163.com
Date: 2024-11-20 11:15:04
LastEditors: Kesar-Chi && momiji0613@163.com
LastEditTime: 2026-10-16 11:51:42
FilePath: \BIGCodem:\progress\ANGPTL3\submit\NC\code-checklist\condGWAS_revised.py
Description: Institute: Beijing Institute Of Genomics, CAS.

Copyright (c) 2024 by ${momiji0613@163.com}, All Rights Reserved. 

python iter_cond_regenie.py \
    -sg start.regenie \
    -g /path/to/pgen_prefix \
    -p pheno.txt \
    -l pred.list \
    -tn BMI \
    -c /pmaster/shaoqi/data/UKB/regenie_bmi.cov \
    -cl genotype_array,inferred_sex,age_visit,PC1,PC2,PC3,PC4,PC5,PC6,PC7,PC8,PC9,PC10,assessment_center,age_squared \
    -af 0.01 \
    -m 1 \
    -b 1000 \
    -tt qt \
    -pthresh 1e-6 \
    -o ./condGWAS
'''

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import subprocess
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


def run_cmd(cmd):
    print("[RUN]", " ".join(map(str, cmd)), flush=True)
    subprocess.run(cmd, check=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Iterative conditional GWAS with regenie"
    )
    parser.add_argument(
        "-sg", "--startgwas",
        type=str,
        required=True,
        help="start GWAS result file"
    )
    parser.add_argument(
        "-g", "--geno",
        type=str,
        required=True,
        help="genotype file root/prefix for regenie --pgen"
    )
    parser.add_argument(
        "-p", "--pheno",
        type=str,
        required=True,
        help="pheno file"
    )
    parser.add_argument(
        "-l", "--list",
        type=str,
        required=True,
        help="regenie step1 pred_list file"
    )
    parser.add_argument(
        "-tn", "--traitname",
        type=str,
        required=True,
        help="trait name used by regenie output"
    )
    parser.add_argument(
        "-c", "--cov",
        type=str,
        default="/pmaster/shaoqi/data/UKB/regenie_bmi.cov",
        help="covariate file"
    )
    parser.add_argument(
        "-cl", "--covlist",
        type=str,
        default="genotype_array,inferred_sex,age_visit,PC1,PC2,PC3,PC4,PC5,PC6,PC7,PC8,PC9,PC10,assessment_center,age_squared",
        help="covariate list"
    )
    parser.add_argument(
        "--cat-covlist",
        type=str,
        default="genotype_array,inferred_sex,assessment_center",
        help="categorical covariate list"
    )
    parser.add_argument(
        "-af", "--afreq",
        type=float,
        default=0.01,
        help="af threshold"
    )
    parser.add_argument(
        "-maf", "--maf",
        action="store_true",
        help="use minor allele frequency (min(A1FREQ, 1-A1FREQ)) for filtering"
    )
    parser.add_argument(
        "-m", "--mac",
        type=int,
        default=1,
        help="minor allele count threshold"
    )
    parser.add_argument(
        "-k", "--keep",
        type=str,
        default=None,
        help="keep sample id file"
    )
    parser.add_argument(
        "-b", "--bsize",
        type=int,
        default=1000,
        help="regenie block size"
    )
    parser.add_argument(
        "-tt", "--traittype",
        type=str,
        default="qt",
        help="trait type: qt or bt"
    )
    parser.add_argument(
        "-bt", "--bt",
        action="store_true",
        help="binary trait"
    )
    parser.add_argument(
        "-pthresh", "--pthresh",
        type=float,
        default=1e-6,
        help="p-value threshold"
    )
    parser.add_argument(
        "-hd", "--header",
        type=str,
        default=None,
        help="header prefix, e.g. # for #CHROM"
    )
    parser.add_argument(
        "--exclude-log10p",
        type=float,
        default=None,
        help="optionally append SNPs with LOG10P < this threshold to exclude file; set 1 to mimic old behavior"
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=8,
        help="threads for regenie"
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=100,
        help="maximum number of conditional iterations"
    )
    parser.add_argument(
        "-o", "--out",
        type=str,
        default="./condGWAS",
        help="output prefix"
    )
    return parser.parse_args()


def read_table(path):
    return pd.read_csv(path, sep=r"\s+", engine="python")


def resolve_trait_mode(args):
    tt = str(args.traittype).lower().strip()

    if args.bt:
        return "bt"

    if tt in {"bt", "binary"}:
        return "bt"

    if tt in {"qt", "quantitative", ""}:
        return "qt"

    raise ValueError(f"Unknown trait type: {args.traittype}")


def get_chrom_col(df, header_prefix=None):
    candidates = []
    if header_prefix:
        candidates.append(f"{header_prefix}CHROM")
    candidates.extend(["CHROM", "#CHROM", "chrom", "#chrom"])

    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"Cannot find chromosome column. Available columns: {list(df.columns)}")


def get_pos_col(df):
    for c in ["GENPOS", "POS", "BP", "GENPOS38"]:
        if c in df.columns:
            return c
    raise ValueError(f"Cannot find position column. Available columns: {list(df.columns)}")


def select_lead_variant(df, args, exclude_ids=None): 
    required = {"ID", "LOG10P", "A1FREQ"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    d = df.copy()
    d = d.loc[d["ID"].notna()].copy()
    d["ID"] = d["ID"].astype(str).str.strip()
    d["LOG10P"] = pd.to_numeric(d["LOG10P"], errors="coerce")
    d["A1FREQ"] = pd.to_numeric(d["A1FREQ"], errors="coerce")

    if args.maf:
        maf = np.minimum(d["A1FREQ"], 1 - d["A1FREQ"])
    else: 
        maf = d["A1FREQ"]
    d = d[
        d["ID"].ne("")  
        & ~d["ID"].str.lower().isin(["nan", "none", "na"])
        & d["LOG10P"].notna()
        & maf.notna()
        & (maf >= args.afreq)
    ].reset_index(drop=True)

    if exclude_ids:
        d = d.loc[~d["ID"].isin(set(exclude_ids))].reset_index(drop=True)

    if d.empty:
        return None, None

    best_idx = d["LOG10P"].idxmax()
    best_log10p = float(d.loc[best_idx, "LOG10P"])

    if best_log10p < -np.log10(args.pthresh):
        return None, best_log10p

    snpid = str(d.loc[best_idx, "ID"])
    return snpid, best_log10p


def write_ids(path, ids, mode="w"):
    ids = [str(x) for x in ids if pd.notna(x) and str(x) != ""]
    with open(path, mode) as f:
        for x in ids:
            f.write(f"{x}\n")


def build_regenie_cmd(args, startpos, endpos, chrom, exclude_file, cnt, trait_mode):
    cmd = [
        "regenie",
        "--threads", str(args.threads),
        "--step", "2",
        "--pgen", args.geno,
        "--minMAC", str(args.mac),
        "--covarFile", args.cov,
        "--covarColList", args.covlist,
        "--phenoFile", args.pheno,
        "--catCovarList", args.cat_covlist,
        "--maxCatLevels", "30",
        "--pred", args.list,
        "--range", f"{chrom}:{startpos}-{endpos}",
        "--bsize", str(args.bsize),
        "--out", f"{args.out}-{cnt}",
        "--condition-list", f"{args.out}.condlist",
    ]

    if args.keep:
        cmd.extend(["--keep", args.keep])

    if exclude_file is not None and Path(exclude_file).exists():
        cmd.extend(["--exclude", exclude_file])

    if trait_mode == "bt":
        cmd.extend(["--bt", "--approx", "--firth"])
    else:
        cmd.append("--qt")

    return cmd


def run_gwas(args, startpos, endpos, chrom, exclude_file, cnt, trait_mode):
    cmd = build_regenie_cmd(
        args=args,
        startpos=startpos,
        endpos=endpos,
        chrom=chrom,
        exclude_file=exclude_file,
        cnt=cnt,
        trait_mode=trait_mode,
    )
    run_cmd(cmd)

    out_file = f"{args.out}-{cnt}_{args.traitname}.regenie"
    if not Path(out_file).exists():
        raise FileNotFoundError(f"Regenie output not found: {out_file}. Check --traitname and REGENIE output prefix.")
    return out_file


def maybe_append_exclude(df, exclude_file, exclude_log10p):
    if exclude_log10p is None:
        return 0

    if "LOG10P" not in df.columns or "ID" not in df.columns:
        return 0

    d = df.copy()
    d["LOG10P"] = pd.to_numeric(d["LOG10P"], errors="coerce")
    ids = (
        d.loc[d["LOG10P"] < exclude_log10p, "ID"]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .tolist()
    )
    if ids:
        write_ids(exclude_file, ids, mode="a")
    return len(ids)


def main():
    args = parse_args()
    trait_mode = resolve_trait_mode(args)

    out_prefix = Path(args.out)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    if not 0 < args.pthresh <= 1 or not 0 <= args.afreq <= 0.5 or args.max_iter < 1:
        raise ValueError("Require 0 < --pthresh <= 1, 0 <= --afreq <= 0.5, and --max-iter >= 1")

    start_df = read_table(args.startgwas)
    if start_df.empty:
        raise ValueError("Start GWAS result is empty")
    chrom_col = get_chrom_col(start_df, args.header)
    pos_col = get_pos_col(start_df)

    pos_num = pd.to_numeric(start_df[pos_col], errors="coerce")
    if pos_num.isna().all():
        raise ValueError(f"All positions are NA in column {pos_col}")
    if start_df[chrom_col].isna().any() or start_df[chrom_col].astype(str).nunique() != 1:
        raise ValueError("Start GWAS must contain exactly one chromosome with no missing chromosome values")

    start_pos = int(pos_num.min()) - 1
    end_pos = int(pos_num.max()) + 1
    chrom = str(start_df[chrom_col].iloc[0])

    lead_snp, lead_log10p = select_lead_variant(start_df, args)
    if lead_snp is None:
        print(f"[STOP] No lead SNP in start GWAS passes p < {args.pthresh} after MAF >= {args.afreq}")
        return

    condlist_file = f"{args.out}.condlist"
    exclude_file = f"{args.out}.excludesnp"

    write_ids(condlist_file, [lead_snp], mode="w")
    Path(exclude_file).write_text("")

    condition_snps = [lead_snp]

    print(f"[INIT] region = {chrom}:{start_pos}-{end_pos}")
    print(f"[INIT] first lead SNP from startgwas = {lead_snp} (LOG10P={lead_log10p:.4f})")
    print(f"[INIT] trait_mode = {trait_mode}")

    for cnt in range(1, args.max_iter + 1):
        print(f"\n========= Iteration {cnt} =========")
        print(f"[INFO] current condition SNPs: {','.join(condition_snps)}")

        dffile = run_gwas(
            args=args,
            startpos=start_pos,
            endpos=end_pos,
            chrom=chrom,
            exclude_file=exclude_file,
            cnt=cnt,
            trait_mode=trait_mode,
        )

        df = read_table(dffile)

        n_ex = maybe_append_exclude(df, exclude_file, args.exclude_log10p)
        if args.exclude_log10p is not None:
            print(f"[INFO] appended {n_ex} SNPs to exclude file using LOG10P < {args.exclude_log10p}")

        snpid, best_log10p = select_lead_variant(df, args, exclude_ids=condition_snps)

        if snpid is None:
            print(f"[STOP] No additional SNP passes p < {args.pthresh} after conditioning.")
            break

        write_ids(condlist_file, [snpid], mode="a")
        condition_snps.append(snpid)
        print(f"[ADD] new condition SNP = {snpid} (LOG10P={best_log10p:.4f})")

    else:
        print(f"[WARN] Reached max_iter = {args.max_iter}; further signals may remain, and the final SNP has not been conditioned on in another iteration.")

    print("\n[DONE] independent signals:")
    for i, s in enumerate(condition_snps, 1):
        print(f"  {i}. {s}")
    print(f"[DONE] condition list file: {condlist_file}")
    print(f"[DONE] exclude file: {exclude_file}")


if __name__ == "__main__":
    main()


# import pandas as pd
# import argparse
# import subprocess
# from glob import glob
# import numpy as np
# import warnings
# warnings.filterwarnings('ignore')


# def run_cmd(args):
#     subprocess.run(args, shell=True, check=True, stdout=subprocess.DEVNULL)
    
    
# def parse_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-sg', '--startgwas', type=str, help='start gwas result file', required=True)
#     parser.add_argument('-g', '--geno', type=str, help='genotype file root and prefix', required=True)
#     parser.add_argument('-p', '--pheno', type=str, help='pheno file', required=True)
#     parser.add_argument('-l', '--list', type=str, help='step1 pred_list file', required=True)
#     parser.add_argument('-tn', '--traitname', type=str, help='trait name', required=True)
#     parser.add_argument('-c', '--cov', type=str, help='covariate file', default='/pmaster/shaoqi/data/UKB/regenie_bmi.cov')
#     parser.add_argument('-cl', '--covlist', type=str, help='covariate list', default='genotype_array,inferred_sex,age_visit,PC1,PC2,PC3,PC4,PC5,PC6,PC7,PC8,PC9,PC10,assessment_center,age_squared')
#     parser.add_argument('-af', '--afreq', type=float, help='allele frequency threshold', default=0.01)
#     parser.add_argument('-m', '--mac', type=int, help='minor allele count threshold', default=1)
#     parser.add_argument('-k', '--keep', type=str, help='keep sample id file')
#     parser.add_argument('-b', '--bsize', type=int, help='block size', default=1000)
#     parser.add_argument('-tt', '--traittype', type=str, help='trait type', default='qt')
#     parser.add_argument('-bt', '--bt', action='store_true', help='binary trait, default is quantitative trait')
#     parser.add_argument('-pthresh', '--pthresh', type=float, help='p-value threshold', default=1e-6)
#     parser.add_argument('-hd', '--header', type=str, help='start header, default is empty, perhaps #', default=None)
#     parser.add_argument('-o', '--out', type=str, help='output file', default='./condGWAS')
    
#     return parser.parse_args()


# def run_gwas(args, startpos, endpos, chrom, ex, cnt):
#     if args.keep:
#         cmd = f"regenie --threads 8 --step 2 --pgen {args.geno} --minMAC {args.mac} --covarFile {args.cov} --covarColList {args.covlist} --phenoFile {args.pheno} --catCovarList genotype_array,inferred_sex,assessment_center --maxCatLevels 30 --pred {args.list} --range {chrom}:{startpos}-{endpos} --{args.traittype} --bsize {args.bsize} --out {args.out}-{cnt} --keep {args.keep} --condition-list {args.out}.condlist --exclude {ex}"
#     else:
#         cmd = f"regenie --threads 8 --step 2 --pgen {args.geno} --minMAC {args.mac} --covarFile {args.cov} --covarColList {args.covlist} --phenoFile {args.pheno} --catCovarList genotype_array,inferred_sex,assessment_center --maxCatLevels 30 --pred {args.list} --range {chrom}:{startpos}-{endpos} --{args.traittype} --bsize {args.bsize} --out {args.out}-{cnt} --condition-list {args.out}.condlist --exclude {ex}"
    
#     if args.bt:
#         cmd += " --bt --approx --firth"
#     else:
#         cmd += " --qt"
#     run_cmd(cmd)
    
#     return f"{args.out}-{cnt}_{args.traitname}.regenie"
    
    
# def filterDF(df, args):
#     df = df[df['A1FREQ']>args.afreq].reset_index(drop=True)
#     if df['LOG10P'].max() < -np.log10(args.pthresh):
#         print(f"No variants remain with p-value < {args.pthresh}, exit.")
#         return None
#     snpid = df.loc[df['LOG10P'].idxmax(), 'ID']
#     return snpid


# def iterCond(args, startpos, endpos, chrom, condlist, exclude, cnt=1):
#     print(f"=========Iteration {cnt}========")
#     print(f"Condition List: {condlist}")
#     dffile = run_gwas(args, startpos, endpos, chrom, exclude, cnt)
#     df = pd.read_csv(dffile, sep=' ')

#     # run_cmd(f"echo {'\n'.join(df.loc[df['LOG10P']<1, 'ID'].tolist())} >> {exclude}")
#     wt = '\n'.join(df.loc[df['LOG10P']<1, 'ID'].tolist())
#     # run_cmd(f"echo {wt} >> {exclude}")
#     with open(exclude, 'a') as f:
#         f.write(wt)

#     snpid = filterDF(df, args)
#     if snpid is None:
#         exit()
#     run_cmd(f"echo {snpid} >> {args.out}.condlist")
#     condlist.append(snpid)
#     iterCond(args, startpos, endpos, chrom, condlist, exclude, cnt+1)


# if __name__ == '__main__':
#     args = parse_args()
#     df = pd.read_csv(args.startgwas, sep=' ')
#     start_pos, end_pos = df['GENPOS'].min()-1, df['GENPOS'].max()+1
#     chromh = args.header + 'CHROM' if args.header else 'CHROM'
#     chrom = df[chromh].values[0]
    
#     snpid = filterDF(df, args)
#     condlist = [snpid]
#     run_cmd(f"echo {snpid} > {args.out}.condlist")
#     run_cmd(f"touch {args.out}.excludesnp")
#     iterCond(args, start_pos, end_pos, chrom, condlist, f'{args.out}.excludesnp')
    
    
    