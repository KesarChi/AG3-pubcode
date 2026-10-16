'''
Author: Kesar-Chi && momiji0613@163.com
Date: 2024-12-18 21:24:14
LastEditors: Kesar-Chi && momiji0613@163.com
LastEditTime: 2026-07-12 15:27:40
FilePath: /BIGCode/pipeline/burden/regenie.py
Description: Institute: Beijing Institute Of Genomics, CAS.

Copyright (c) 2024 by ${momiji0613@163.com}, All Rights Reserved.

# Step 1
python ~/BIGCode/pipeline/burden/regenie.py \
  -pgen /path/to/genotype/ukb_qc \
  -pheno /path/to/pheno.tsv \
  -phenoCol T2D \
  -covar /path/to/covar.tsv \
  -step 1 \
  -bt \
  -out T2D_step1 \
  -threads 32

# Step 2 GWAS
for chr in {1..22}
do
  python ~/BIGCode/pipeline/burden/regenie.py \
    -pgen /path/to/genotype/ukb_chr${chr} \
    -pheno /path/to/pheno.tsv \
    -phenoCol T2D \
    -covar /path/to/covar.tsv \
    -step 2 \
    -bt \
    -gwas \
    -pred T2D_step1_pred.list \
    -chr ${chr} \
    -out T2D_chr${chr}_gwas \
    -threads 16
done
 
'''

import pandas as pd
import argparse
import subprocess
from glob import glob
import numpy as np
import multiprocessing
import sys
import warnings
warnings.filterwarnings('ignore')


def run_cmd(args):
    subprocess.run(args, shell=True, check=True, stdout=subprocess.DEVNULL)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-pgen', '--pgen', type=str, help='plink pgen file', required=True)
    parser.add_argument('-pheno', '--pheno', type=str, help='pheno file', required=True)
    parser.add_argument('-step', '--step', type=int, help='step', required=True)
    parser.add_argument('-out', '--out', type=str, help='output file', required=True)
    parser.add_argument('-covar', '--covar', type=str, help='covar file', required=True)
    parser.add_argument('-covarCol', '--covarCol', type=str, help='covar column list', default="genotype_array,inferred_sex,age_visit,PC1,PC2,PC3,PC4,PC5,PC6,PC7,PC8,PC9,PC10,assessment_center,age_squared")
    parser.add_argument('-phenoCol', '--phenoCol', type=str, help='pheno column', default=None)
    parser.add_argument('-keep', '--keep', type=str, help='keep sample file', default=None)
    parser.add_argument('-remove', '--remove', type=str, help='remove sample file', default=None)
    parser.add_argument('-extract', '--extract', type=str, help='extract snp file', default=None)
    parser.add_argument('-exclude', '--exclude', type=str, help='exclude snp file', default=None)
    parser.add_argument('-catCovar', '--catCovar', type=str, help='categorical covar list', default="genotype_array,inferred_sex,assessment_center")
    parser.add_argument('-maxCatLevels', '--maxCatLevels', type=int, help='max categorical levels', default=30)
    parser.add_argument('-pred', '--pred', type=str, help='pred list file', default=None)
    parser.add_argument('-range', '--range', type=str, help='range', default=None)
    parser.add_argument('-qt', '--qt', action='store_true', help='quantitative trait')
    parser.add_argument('-bt', '--bt', action='store_true', help='binary trait')
    parser.add_argument('-bsize', '--bsize', type=int, help='block size', default=1000)
    parser.add_argument('-pThresh', '--pThr', type=float, help='p-value threshold', default=0.01)
    parser.add_argument('-chr', '--chr', type=int, help='chromosome', default=None)
    parser.add_argument('-minMAC', '--minMAC', type=str, help='minor allele count threshold', default='5')
    parser.add_argument('-threads', '--threads', type=int, help='threads', default=8)  
    parser.add_argument('-condlist', '--condlist', type=str, help='condition list', default=None)
    parser.add_argument('-aafs', '--aafs', type=str, help='aaf bins', default="0.001,0.0001,0.00001")
    parser.add_argument('-anno', '--anno', type=str, help='annotation file', default=None)
    parser.add_argument('-setlist', '--setlist', type=str, help='set list file', default=None)
    parser.add_argument('-maskdef', '--maskdef', type=str, help='mask def file', default=None)
    parser.add_argument('-buildmask', '--buildmask', type=str, help='build mask file', default="sum")
    parser.add_argument('-vctest', '--vctest', type=str, help='vc test', default=None)
    parser.add_argument('-vcmacthr', '--vcmacthr', type=int, help='vc mac threshold', default=1)
    parser.add_argument('-gwas', '--gwas', action='store_true', help='gwas')
    parser.add_argument('-int', '--int', action='store_true', help='int transform')
    
    return parser.parse_args()


def general(cmd, args):
    cmd += f" --pred {args.pred}"
    cmd += f" --minMAC {args.minMAC}"
    cmd += f" --vc-MACthr {args.vcmacthr}"
    
    if args.keep:
        cmd += f" --keep {args.keep}"
    elif args.remove:
        cmd += f" --remove {args.remove}"
        
    if args.extract:
        cmd += f" --extract {args.extract}"
    elif args.exclude:
        cmd += f" --exclude {args.exclude}"
        
    if args.range:
        cmd += f" --range {args.range}"
    elif args.chr:
        cmd += f" --chr {args.chr}"
    
    if args.condlist:
        cmd += f" --condition-list {args.condlist}"
    
    return cmd


def step2(cmd, args):
    cmd += f" --anno-file {args.anno}"
    cmd += f" --set-list {args.setlist}"
    cmd += f" --mask-def {args.maskdef}"
    cmd += f" --build-mask {args.buildmask}"
    cmd += f" --aaf-bins {args.aafs}"
    cmd += " --write-mask-snplist"
    cmd += f" --pThresh {args.pThr}"
    
    if args.vctest:
        cmd += f" --vc-test {args.vctest}"
    
    cmd = general(cmd, args)
    
    return cmd


if __name__ == '__main__':
    args = get_args()
    print(' '.join(sys.argv))
    
    cmd = "regenie"
    
    cmd += f" --pgen {args.pgen} \
    --phenoFile {args.pheno} \
    --step {args.step} \
    --out {args.out} \
    --covarFile {args.covar} \
    --covarColList {args.covarCol} \
    --threads {args.threads} \
    --catCovarList {args.catCovar} \
    --maxCatLevels {args.maxCatLevels} \
    --bsize {args.bsize} "
    
    if args.qt:
        cmd += "--qt"
        if args.int:
            cmd += " --apply-rint"
    elif args.bt:
        cmd += "--bt --approx --firth"
        
    if args.step == 2 and not args.gwas:
        cmd = step2(cmd, args)
    elif args.step == 2 and args.gwas:
        cmd = general(cmd, args)
    elif args.step == 1:
        cmd += f" --pThresh {args.pThr}"
    
    run_cmd(cmd)
    