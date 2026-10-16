'''
Author: Kesar-Chi && momiji0613@163.com
Date: 2025-01-27 11:38:24
LastEditors: Kesar-Chi && momiji0613@163.com
LastEditTime: 2026-10-16 11:31:16
FilePath: \BIGCodem:\progress\ANGPTL3\submit\NC\code-checklist\phewas.py
Description: Institute: Beijing Institute Of Genomics, CAS.

Copyright (c) 2025 by ${momiji0613@163.com}, All Rights Reserved. 
'''


import pandas as pd
import numpy as np
import statsmodels.api as sm
import scipy.stats as stats
from glob import glob
import argparse
import os
import sys
sys.path.append('/home/brush')
from brush import statools
import seaborn as sns
import matplotlib.pyplot as plt
from adjustText import adjust_text
import warnings
warnings.filterwarnings('ignore')


def load_data(input_file, cov_file, pheno_root, keep_file=None, exclude_file=None):
    """
    加载数据并合并。

    参数:
    - input_file: str，输入文件路径，包含特征数据。
    - cov_file: str，协变量文件路径。
    - pheno_root: str，表型文件根目录。
    - keep_file: str，保留样本文件路径。
    - exclude_file: str，排除样本文件路径。

    返回:
    - df: DataFrame，合并后的数据。
    - pheno_files: list，表型文件列表。
    """
    df = pd.read_csv(input_file, sep='\s+')
    colnames = df.columns.tolist()
    colnames[0] = 'eid'
    df.columns = colnames
    if 'eid' not in df.columns:
        raise ValueError("输入文件的第一列必须是 'eid'。")
    xcols = df.columns[1:].tolist()

    cov = pd.read_csv(cov_file, sep='\t', compression='gzip')
    colnames = cov.columns.tolist()
    colnames[0] = 'eid'
    cov.columns = colnames
    if 'eid' not in cov.columns:
        raise ValueError("协变量文件的第一列必须是 'eid'。")

    df = pd.merge(df, cov, on='eid', how='inner')

    if keep_file:
        keep_samples = pd.read_csv(keep_file, header=None)[0].tolist()
        df = df[df['eid'].isin(keep_samples)].reset_index(drop=True)
    if exclude_file:
        exclude_samples = pd.read_csv(exclude_file, header=None)[0].tolist()
        df = df[~df['eid'].isin(exclude_samples)].reset_index(drop=True)

    pheno_files = glob(os.path.join(pheno_root, '*.feather'))
    if not pheno_files:
        raise ValueError(f"未找到表型文件：{pheno_root}")

    return df, pheno_files, xcols


def run_phewas(df, pheno_files, cov_list, x_list, save_dir, surv=False, mr=False, incident=False):
    """
    运行 PheWAS 分析。

    参数:
    - df: DataFrame，合并后的数据。
    - pheno_files: list，表型文件列表。
    - cov_list: list，协变量列表。
    - x_list: list 或 'all'，特征列表。

    返回:
    - results: DataFrame，PheWAS 结果。
    """
    results = []
    for pheno_file in pheno_files:
        try:
            pheno = pd.read_feather(pheno_file)
            if 'eid' not in pheno.columns:
                raise ValueError(f"表型文件 {pheno_file} 的第一列必须是 'eid'。")
            if not surv:
                if not incident:
                    pheno = pheno[['eid', 'event']]
                else:
                    print(f"incident event: {pheno_file}")
                    pheno = pheno[['eid', 'incident', 'prevalent']]
                    pheno = pheno[(pheno['prevalent'] == 0) | (pheno['prevalent'].isna())].reset_index(drop=True)
                    pheno.drop('prevalent', axis=1, inplace=True)
                    pheno.columns = ['eid', 'event']
                    
                if len(pheno['event'].unique()) == 1:
                    print(f"{pheno_file} has only one unique value")
                    continue
                
                pheno['eid'] = pheno['eid'].astype(int)
                df_pheno = pd.merge(df, pheno, on='eid', how='inner')
                
                if not mr:
                    for x in x_list:
                        res_tmp = statools.stat_model(df_pheno, 'event', x, cov_list, 'logit', verbose=False, save=f"{save_dir}/{x}/{pheno_file.split('/')[-1].split('.')[0]}.model")
                        results.append({
                            'Trait': pheno_file.split('/')[-1].split('.')[0],
                            'variable': x,
                            'beta': res_tmp.params[x],
                            'p': res_tmp.pvalues[x]
                        })
                else:
                    res_tmp = statools.stat_model(df_pheno, 'event', x_list, cov_list, 'logit', verbose=False, save=f"{save_dir}/{x}/{pheno_file.split('/')[-1].split('.')[0]}.model")
                    results.append({
                        'Trait': pheno_file.split('/')[-1].split('.')[0],
                        'variable': x_list[0],
                        'beta': res_tmp.params[x_list[0]],
                        'p': res_tmp.pvalues[x_list[0]]
                    })
            else:
                pheno = pheno[['eid', 'prevalent', 'incident', 'survTime']]
                pheno = pheno[(pheno['prevalent'] == 0) | (pheno['prevalent'].isna())].reset_index(drop=True)
                if len(pheno['incident'].unique()) == 1:
                    print(f"{pheno_file} has only one unique value")
                    continue
                pheno['eid'] = pheno['eid'].astype(int)
                pheno.drop('prevalent', axis=1, inplace=True)
                pheno.columns = ['eid', 'event', 'survTime']
                df_pheno = pd.merge(df, pheno, on='eid', how='inner')
                for x in x_list:
                    res_tmp = statools.surv_model(df_pheno, ['survTime', 'event'], x, cov_list, verbose=False, onehot_verbose=False)
                    results.append({
                        'Trait': pheno_file.split('/')[-1].split('.')[0],
                        'variable': x,
                        'beta': res_tmp.loc[x, 'coef'],
                        'p': res_tmp.loc[x, 'p']
                    })
                    res_tmp.to_csv(f"{save_dir}/{x}/{pheno_file.split('/')[-1].split('.')[0]}.model", sep='\t')
                
        except Exception as e:
            print(f"{pheno_file}, Error: {e}")
            continue

    return pd.DataFrame(results)


def create_dir(root, xcols, mr):
    if not mr:
        for x in xcols:
            os.makedirs(os.path.join(root, x), exist_ok=True)
    else:
        os.makedirs(os.path.join(root, 'MR2x2'), exist_ok=True)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='phewas')
    parser.add_argument('-i', '--input', type=str, help='input file, 1st col is eid, others are x', required=True)
    parser.add_argument('-o', '--output', type=str, help='output root', default='./')
    parser.add_argument('-xl', '--xlist', type=str, help='x list from --input, default=all', default='all')
    parser.add_argument('-c', '--cov', type=str, help='covariate file', default="/pmaster/shaoqi/data/UKB/noProt_UKB_drug.txt.gz")
    parser.add_argument('-cl', '--covlist', type=str, help='covariate list', default='age,sex,PC1,PC2,PC3,PC4,PC5,PC6,PC7,PC8,PC9,PC10')
    parser.add_argument('-p', '--pheno', type=str, help='pheno file or root', default='/pmaster/shaoqi/data/UKB/phewas/')
    parser.add_argument('-k', '--keep', type=str, help='keep samples file, one col, no header')
    parser.add_argument('-e', '--exclude', type=str, help='exclude samples file, one col, no header')
    parser.add_argument('-surv', '--surv', action='store_true', help='survival analysis')
    parser.add_argument('-scale', '--scale', action='store_true', help='scale')
    parser.add_argument('-firth', '--firth', action='store_true', help='firth')
    parser.add_argument('-res', '--residual', action='store_true', help='residual')
    parser.add_argument('-rl', '--reslist', type=str, help='residual list', default='PC1,PC2,PC3,PC4,PC5,PC6,PC7,PC8,PC9,PC10')
    parser.add_argument('-closep', '--closep', action='store_true', help='close plot')
    parser.add_argument('-incident', '--incident', action='store_true', help='incident event')
    parser.add_argument('-map', '--map', action='store_true', help='map icd10 to name')
    args = parser.parse_args()
    print(' '.join(sys.argv))
    
    if not args.onlyplot:
        df, pheno_files, xcols = load_data(args.input, args.cov, args.pheno, args.keep, args.exclude)
        print(f'Loaded data: {df.shape}')
        cov_list = args.covlist.split(',')
        x_list = xcols if args.xlist == 'all' else args.xlist.split(',')
        print(f'Variables: {x_list}')
        create_dir(args.output, x_list, args.mr2by2)
        
        if args.residual:
            print(f'Calculating residuals: {args.reslist}')
            for x in x_list:
                df[x] = statools.get_resid(df, x, args.reslist.split(','))
            
        if args.scale:
            print('Scaling...')
            for x in x_list:
                df[x] = (df[x] - df[x].mean()) / df[x].std()
            
        results = run_phewas(df, pheno_files, cov_list, x_list, args.output, args.surv, args.mr2by2, args.incident)
        mp_icd = pd.read_csv('/home/pipeline/phewas/ICD10-map.txt', sep='\t')
        results['name'] = results['Trait'].map(mp_icd.set_index('ICD')['Name'])
        results['name2'] = results['Trait'] + '.' + results['name']
        output_file = os.path.join(args.output, 'phewas_results.csv')
        results.to_csv(output_file, index=False, sep='\t')
    else:
        results = pd.read_csv(args.input, sep='\s+')
        if args.map:
            mp_icd = pd.read_csv('/home/pipeline/phewas/ICD10-map.txt', sep='\t')
            results['name'] = results['Trait'].map(mp_icd.set_index('ICD')['Name'])
            results['name2'] = results['Trait'] + '.' + results['name']
            output_file = os.path.join(args.output, 'phewas_results_ICDname.tsv')
            results.to_csv(output_file, index=False, sep='\t')