'''
Simulate COVID-19 in Nigeria
'''

#%% Imports and settings
import sciris as sc
import covasim as cv
import numpy as np
import pylab as pl

# Settings
seed = 1
verbose = 1

 # Define parameters
pars = {'pop_size': 20e3,
        'pop_type': 'hybrid',
        'location': 'nigeria', # Load age structure and household sizes
        'start_day':'2020-04-01',
        'n_days': 180,
        'pop_scale': 1000,
        'pop_infected': 100,
        'rescale': 1,
        'rescale_threshold': 0.2,
        'rescale_factor': 2,
        'n_beds': 0,
        'OR_no_treat': 100.0,
        'rel_symp_prob': 1.2,
        'rel_severe_prob': 4.5,
        'rel_crit_prob': 5., #
        'rel_death_prob': 1.0
        }

sim = cv.Sim(pars=pars)
#sim.run()
#sc.tic(); sim.initialize(); sc.toc()

# Define the scenarios
ineffective = {
    'baseline_ineffective': {
        'name': 'Ineffective lockdown, no hospital capacity',
        'pars': {
        }
    },
    'fullcare_ineffective': {
        'name': 'Ineffective lockdown, hospital scale-up',
        'pars': {
            'n_beds': np.inf,
            'rel_crit_prob': 1.55,
            'rel_death_prob': 0.6
        }
    },
}

effective = {
    'baseline_effective': {
        'name': 'Effective lockdown, no hospital capacity',
        'pars': {
            'beta_layer': {'h': 1.2, 's': 0.1, 'w': 0.1, 'c': 0.1},
        }
    },
    'fullcare_effective': {
        'name': 'Effective lockdown, hospital scale-up',
        'pars': {
            'beta_layer': {'h': 1.2, 's': 0.1, 'w': 0.1, 'c': 0.1},
            'n_beds': np.inf,
            'rel_crit_prob': 1.55,
            'rel_death_prob': 0.6
        }
    },
}

metapars = {'n_runs': 3}

to_plot = [
    'cum_infections',
    'new_infections',
    'cum_symptomatic',
    'cum_severe',
    'cum_critical',
    'cum_deaths',
]

#fig_args = dict(figsize=(24, 40)) # If all
fig_args = dict(figsize=(24, 20))
all = sc.mergedicts(ineffective,effective)
scens = cv.Scenarios(sim=sim, scenarios=all)
scens.run(verbose=verbose)
scens.plot(do_save=1, do_show=0, to_plot=to_plot, fig_path=f'results/nigeria_scenarios_all', n_cols=2, fig_args=fig_args)


#fig_args = dict(figsize=(24, 16))
#for lockdown_label,lockdown in zip(['ineffective','effective'],[ineffective, effective]):
#    scens = cv.Scenarios(sim=sim, scenarios=lockdown)
#    scens.run(verbose=verbose)
#    scens.plot(do_save=1, do_show=0, to_plot=to_plot, fig_path=f'results/nigeria_scenarios_{lockdown_label}', fig_args=fig_args)



