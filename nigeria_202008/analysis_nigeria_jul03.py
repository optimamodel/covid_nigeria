'''
Simulate COVID-19 in Nigeria
'''

#%% Imports and settings
import sciris as sc
import covasim as cv
import numpy as np
import pylab as pl
import pandas as pd
import covasim.utils as cvu

# Settings
seed = 1
verbose = 1

# Load scenarios
scens = sc.loadobj('nigeria.scens')
lockdown_ends = scens.sims[0][0].day('2020-05-04')
count_from = scens.sims[0][0].day('2020-06-30')

for indicator in ['infections', 'quarantined', 'diagnoses']:
    baseline = scens.results[f'cum_{indicator}']['noscreen']['best']
    baseline_no = baseline[-1] - baseline[count_from]

    eoy_totals = {name: val['best'][-1] - val['best'][count_from] for name, val in scens.results[f'cum_{indicator}'].items()}
    pct_changes = {name: (baseline_no-eoy_totals[name])/baseline_no for name in scens.results[f'cum_{indicator}'].keys()}
    df = pd.DataFrame.from_dict(pct_changes, orient='index')
    df.to_excel(f'{indicator}.xlsx')

