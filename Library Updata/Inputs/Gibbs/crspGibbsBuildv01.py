""" paper-Joel Hasbrouck(2009)
    Python code to build the CRSP Gibbs sampler
    The program first builds the crsp sample (market spilts and changes in 
    listing exchanges),and then passes the crsp data to the estimation rounties.

    note:RollGibbsLibrary02.py must be run prior to this program, to compile the IML subroutines.
"""
import pandas as pd
import numpy as np
from datetime import datetime
from scipy.stats import truncnorm
import scipy.linalg as la
from scipy.special import expit
from RollGibbsLibrary02 import RollGibbsBeta

# Data preprocessing functions
def compute_exchange_changes(dsenames):
    # Detect exchange changes
    dsenames.sort_values(['permno', 'exchcd', 'namedt'], inplace=True)
    dsenames.rename(columns={'exchcd': 'exchcd0', 'namedt': 'startDate'}, inplace=True)
    dsenames['endDate'] = dsenames.groupby('permno')['namedt'].shift(-1)
    exch = dsenames[dsenames.groupby(['permno', 'exchcd0'])['namedt'].transform('idxmin') == dsenames.index]
    exch = exch[exch.groupby(['permno', 'exchcd0'])['namedt'].transform('idxmax') == exch.index]
    return exch[['permno', 'year', 'startDate', 'endDate', 'exchcd']]

def compute_splits(dsf):
    # Detect splits
    dsf.sort_values(['permno', 'date'], inplace=True)
    dsf['cfacpr_lag'] = dsf.groupby('permno')['cfacpr'].shift()
    dsf['ratio'] = dsf['cfacpr'] / dsf['cfacpr_lag']
    splits = dsf[(dsf['ratio'] > 1.2) | (dsf['ratio'] < 0.8)]
    return splits[['permno', 'year', 'date']]

def compute_trade_direction(dsf):
    # Compute trade direction indicator
    dsf.sort_values(['permno', 'date'], inplace=True)
    dsf['prc_abs'] = np.abs(dsf['prc'])
    dsf['prev_prc_abs'] = dsf.groupby('permno')['prc_abs'].shift()
    dsf['prc_change'] = dsf['prc_abs'] - dsf['prev_prc_abs']
    dsf['q'] = np.sign(dsf['prc_change'])
    dsf.loc[dsf['prc_change'] == 0, 'q'] = 1
    dsf['q'] = dsf['q'].fillna(1)
    return dsf

def main():
    np.random.seed(12345)

# Load CRSP datasets
    dsenames = pd.read_csv('crsp_dsenames.csv')
    dsf = pd.read_csv('crsp_dsf.csv')
    dsi = pd.read_csv('crsp_dsi.csv')
    
    # Compute exchange changes
    exch = compute_exchange_changes(dsenames)
    exch['year'] = exch['startDate'].dt.year
    
    # Compute splits
    splits = compute_splits(dsf)
    splits['year'] = splits['date'].dt.year
    
    # Merge data
    dsf['year'] = dsf['date'].dt.year
    dsf['kSample'] = 1
    dsf = pd.merge(dsf, exch.rename(columns={'startDate': 'date', 'exchcd': 'exchcd0'}), 
                   on=['permno', 'year', 'date'], how='left')
    dsf = pd.merge(dsf, splits.rename(columns={'date': 'split_date'}), 
                   on=['permno', 'year'], how='left')
    
    # Compute index data (pm)
    dsi['year'] = dsi['date'].dt.year
    dsf = pd.merge(dsf, dsi[['date', 'pm']], on='date', how='left')
    
    # Compute trade direction indicator
    dsf = compute_trade_direction(dsf)
    
    # Preprocess and group data
    grouped = dsf.groupby(['permno', 'year', 'kSample'])
    
    results = []
    for group_name, group_data in grouped:
        permno, year, kSample = group_name
        p = group_data['p'].values
        pm = group_data['pm'].values
        q = group_data['q'].values
        
        # Run Gibbs sampler
        parmOut = RollGibbsBeta(p, pm, q, nSweeps=1000, nDrop=200)
        c, beta, varu = parmOut.mean(axis=0)
        sdu = np.sqrt(varu)
        
        # Append results
        results.append({
            'permno': permno,
            'year': year,
            'kSample': kSample,
            'c': c,
            'beta': beta,
            'varu': varu,
            'sdu': sdu,
        })
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Descriptive statistics
    descriptive_stats = dsf.groupby(['permno', 'year', 'kSample']).agg(
        exchcd=('exchcd', 'first'),
        shrcd=('shrcd', 'first'),
        firstDate=('date', 'min'),
        lastDate=('date', 'max'),
        nDays=('date', 'count'),
        nTradeDays=('prc', lambda x: (x != 0).sum()),
    ).reset_index()
    
    # Merge results with descriptive statistics
    results_merged = pd.merge(results_df, descriptive_stats, 
                             on=['permno', 'year', 'kSample'], how='left')
    
    # Export to CSV
    results_merged.to_csv('crspgibbs.csv', index=False)

if __name__ == "__main__":
    main()

















