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
            'n_days': 92,
            'pop_scale': 1000,
            'pop_infected': 10,
            'beta': 0.015,
            'rand_seed': 1,
#            'rescale': 1,
#            'rescale_threshold': 0.2,
#            'rescale_factor': 2,
            'n_beds': 1e-6,
            'OR_no_treat': 100.0,
            'rel_symp_prob': 1.2,
            'rel_severe_prob': 4.5,
            'rel_crit_prob': 5., #
            'rel_death_prob': 1.0,
            'asymp_factor': 0.5, # Ferguson
            'quar_eff': {'h': 1, 's': 0.5, 'w': 0.0, 'c': 0.1},
            'quar_period': 14
     }

    sim = cv.Sim(pars=pars)

    mapping = {
         0: 0.2,
         20: 0.9,
         40: 1.0,
         70: 2.5,
         80: 5.0,
         90: 10.0,
     }
    #sim.run()

    #for age, val in mapping.items():
    #     sim.people.rel_sus[sim.people.age > age] = val

    sim['contacts']['s'] = 0 # No school
    sim['contacts']['c'] = 5  # Reduced community

    return sim


metapars = {'n_runs': 1}
to_plot = [
    'cum_infections',
    'new_infections',
    'cum_symptomatic',
    'cum_severe',
    'cum_critical',
    'cum_deaths',
    'new_diagnoses',
    'new_quarantined',
]
fig_args = dict(figsize=(28, 20))


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


def lift_lockdown_endday():
    scenarios = {
        f'{int(d)}daylockdown': {
            'name': f'{int(d)} day lockdown',
            'pars': {
                'interventions': [cv.clip_edges(start_day='2020-04-01', end_day=d, change={'w': 0.3, 's': 0.0}), # Reduce work by 70%
                                  #cv.change_beta([1, d], [0.0, 1], layers='s'),
                                  #cv.change_beta([1, d], [0.3, 1], layers='w'),
                                  cv.change_beta([1, d], [0.3, 1], layers='c')] # Reduce community by 70%
            }
        } for d in [35, 42, 56, 70, 84, 98] # lockdowns of 4,6,8,10,12,14 weeks
    }

    sim = make_sim()
    scens = cv.Scenarios(sim=sim, scenarios=scenarios, metapars=metapars)
    df = scens.run(verbose=verbose)
    scens.plot(do_save=1, do_show=0, to_plot=to_plot, fig_path=f'results/nigeria_scenarios_6_1000', n_cols=2, fig_args=fig_args)


    return df, scens


def lift_lockdown_howmuch():
    scenarios = {
        f'reopen_{po}pc': {
            'name': f'Reopen {int(po*100)}% of work',
            'pars': {
                'interventions': [cv.clip_edges(start_day='2020-04-01', end_day=35, change={'w': 0.3, 's': 0.0}), # Reduce work by 70%
                                  cv.clip_edges(start_day='2020-04-01', end_day=35, change={'w': po, 's': 0.0}),
                                  cv.change_beta([1, 180], [0.3, 1], layers='c')] # Reduce community by 70%
            }
        } for po in [0.3, 0.4, 0.5, 0.6, 0.7] # Reopen work -
    }

    sim = make_sim()
    scens = cv.Scenarios(sim=sim, scenarios=scenarios, metapars={'n_runs':3})
    df = scens.run(verbose=verbose)
    scens.plot(do_save=1, do_show=0, to_plot=to_plot, fig_path=f'results/nigeria_scenarios_howmuch', n_cols=2, fig_args=fig_args)


    return df, scens



def lift_lockdown_paper():

    pop_scale = 1000
    test_kwargs = {'sympt_test': 100.0, 'quar_test': 10.0, 'sensitivity': 1.0, 'test_delay': 0, 'loss_prob': 0}
    testnos = [500,2000,20000]

    scenarios_in = {
        f'{tn}_ineff': {
            'name': f'{tn} tests/day, ineffective face-masks',
            'pars': {
                'interventions': [
                    cv.clip_edges(start_day='2020-04-01', end_day='2020-05-04', change={'w': 0.3}), # Reduce work by 70% til May 4
                    cv.test_num(daily_tests=np.array([200]*180) / pop_scale, start_day='2020-04-01', end_day='2020-05-04', **test_kwargs),
                    cv.contact_tracing(start_day='2020-04-01', end_day='2020-05-04',
                                       trace_probs={'h': 1, 's': 0, 'w': 0.8, 'c': 0.1},
                                       trace_time={'h': 1, 's': 7, 'w': 7, 'c': 7}),
                    cv.test_num(daily_tests=np.array([tn]*180) / pop_scale, start_day='2020-05-04', **test_kwargs),
                    cv.contact_tracing(start_day='2020-05-04', end_day='2020-09-30',
                                       trace_probs={'h': 1, 's': 0, 'w': 0.8, 'c': 0.1},
                                       trace_time={'h': 1, 's': 7, 'w': 7, 'c': 7})]
            }
        } for tn in testnos
    }

    scenarios_eff = {
        f'{tn}_eff': {
            'name': f'{tn} tests/day, effective face-masks',
            'pars': {
                'interventions': [
                    cv.clip_edges(start_day='2020-04-01', end_day='2020-05-04', change={'w': 0.3}), # Reduce work by 70% til May 4
                    cv.test_num(daily_tests=np.array([200]*180) / pop_scale, start_day='2020-04-01', end_day='2020-05-04', **test_kwargs),
                    cv.contact_tracing(start_day='2020-04-01', end_day='2020-05-04',
                                       trace_probs={'h': 1, 's': 0, 'w': 0.8, 'c': 0.1},
                                       trace_time={'h': 1, 's': 7, 'w': 7, 'c': 7}),
                    cv.test_num(daily_tests=np.array([tn]*180) / pop_scale, start_day='2020-05-04', **test_kwargs),
                    cv.contact_tracing(start_day='2020-05-04', end_day='2020-09-30',
                                       trace_probs={'h': 1, 's': 0, 'w': 0.8, 'c': 0.1},
                                       trace_time={'h': 1, 's': 7, 'w': 7, 'c': 7}),
                    cv.change_beta([35, 182], [0.9, 1], layers='w'),
                    cv.change_beta([35, 182], [0.9, 1], layers='c')]  # Reduce community by 70%
    }
        } for tn in testnos
    }

    scenarios = sc.mergedicts(scenarios_eff, scenarios_in)
    sim = make_sim()
    scens = cv.Scenarios(sim=sim, scenarios=scenarios, metapars=metapars)
    df = scens.run(verbose=verbose)
    scens.plot(do_save=1, do_show=0, to_plot=to_plot, fig_path=f'results/nigeria_scenarios_paper', n_cols=2, fig_args=fig_args)


    return df, scens


df, scens = lift_lockdown_paper()
#sim = make_sim()
#sim.plot(do_save=1, do_show=0)
