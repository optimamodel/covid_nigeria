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

def make_sim():
     # Define parameters
    pars = {'pop_size': 20e3,
            'pop_type': 'hybrid',
            'location': 'nigeria', # Load age structure and household sizes
            'start_day':'2020-04-01',
            'n_days': 365,
            'pop_scale': 1000,
            'pop_infected': 100,
            'rescale': 1,
            'rescale_threshold': 0.2,
            'rescale_factor': 2,
            'n_beds': 1e-6,
            'OR_no_treat': 100.0,
            'rel_symp_prob': 1.2,
            'rel_severe_prob': 4.5,
            'rel_crit_prob': 5., #
            'rel_death_prob': 1.0
            }

    sim = cv.Sim(pars=pars)

    return sim


metapars = {'n_runs': 3}

to_plot = [
    'cum_infections',
    'new_infections',
    'cum_symptomatic',
    'cum_severe',
    'cum_critical',
    'cum_deaths',
]

fig_args = dict(figsize=(24, 20))


def basic_lockdown():
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
                'interventions': [clips],
            }
        },
        'fullcare_effective': {
            'name': 'Effective lockdown, hospital scale-up',
            'pars': {
                'interventions': [clips],
                'n_beds': np.inf,
                'rel_crit_prob': 1.55,
                'rel_death_prob': 0.6
            }
        },
    }

    sim = make_sim()
    scenarios = sc.mergedicts(ineffective, effective)
    scens = cv.Scenarios(sim=sim, scenarios=scenarios, metapars=metapars)
    df = scens.run(verbose=verbose)
    scens.plot(do_save=1, do_show=0, to_plot=to_plot, fig_path=f'results/nigeria_lockdown_scenarios', n_cols=2,
               fig_args=fig_args)

    return df, scens


def lift_lockdown():
    scenarios = {
        f'{int(d)}daylockdown': {
            'name': f'{int(d)} day lockdown',
            'pars': {
                'interventions': [cv.clip_edges(start_day='2020-04-01', end_day=d, change={'w': 0.3, 's': 0.0}), # Reduce work by 70%
                                  cv.change_beta([1, d], [0.3, 1], layers='c')] # Reduce community by 70%
            }
        } for d in range(30,180,15)
    }

    sim = make_sim()
    scens = cv.Scenarios(sim=sim, scenarios=scenarios, metapars=metapars)
    df = scens.run(verbose=verbose)
    scens.plot(do_save=1, do_show=0, to_plot=to_plot, fig_path=f'results/nigeria_scenarios_70', n_cols=2, fig_args=fig_args)


    return df, scens


df, scens = lift_lockdown()

