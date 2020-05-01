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

def make_sim(beta=None):

    if beta is None: beta = 0.015
     # Define parameters
    pars = {'pop_size': 20e3,
            'pop_type': 'hybrid',
            'location': 'nigeria', # Load age structure and household sizes
            'start_day':'2020-04-01',
            'n_days': 92,
            'pop_scale': 1000,
            'pop_infected': 10,
            'beta': beta,
            'rand_seed': 1,
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

    sim['contacts']['s'] = 0 # No school
    sim['contacts']['c'] = 5  # Reduced community

    return sim

class quarantine_severe(cv.Intervention):
    '''Quarantine people with severe symptoms'''
    def __init__(self, start_day=0, end_day=None):
        self.start_day   = start_day
        self.end_day     = end_day
        self._store_args()

        return

    def initialize(self, sim):
        ''' Fix the dates '''
        self.start_day = sim.day(self.start_day)
        self.end_day   = sim.day(self.end_day)
        self.days      = [self.start_day, self.end_day]
        return

    def apply(self, sim):
        t = sim.t
        if t < self.start_day:
            return
        elif self.end_day is not None and t > self.end_day:
            return

        severe_inds = cvu.true(sim.people.severe)
        sim.people.quarantine(severe_inds)
        sim.results['new_quarantined'][t] += len(severe_inds)

#        print(f'{severe_inds} put into quarantine due to severe symptoms.')

        return


class screen(cv.Intervention):
    '''
    Implement screening, with people sent into quarantine if they screen positive
    '''
    def __init__(self, daily_screens, sensitivity=0.5, prob_quar_after_fever=0.7,
                 start_day=0, end_day=None, do_plot=None):
        super().__init__(do_plot=do_plot)
        self.daily_screens = daily_screens  # Should be a list of length matching time
        self.sensitivity  = sensitivity     # Prevalence of fever in symptomatic COVID patients
        self.prob_quar_after_fever = prob_quar_after_fever # Probability that you'll quarantine after you've had your temp checked and have a fever (but not a confirmed diagnosis)
        self.start_day   = start_day
        self.end_day     = end_day
        self._store_args()
        self.screen_quarantined = 0

        return

    def initialize(self, sim):
        ''' Fix the dates '''
        self.start_day = sim.day(self.start_day)
        self.end_day   = sim.day(self.end_day)
        self.days      = [self.start_day, self.end_day]
        return

    def apply(self, sim):
        t = sim.t
        if t < self.start_day:
            return
        elif self.end_day is not None and t > self.end_day:
            return

        adults = sim.people.age > 18
        adults_inds = sc.findinds(adults)
        screen_inds = cvu.choose(len(adults_inds), self.daily_screens[t]) # Who will we screen today - untargeted
        screen_inds = np.unique(screen_inds)

        screened_adult_inds = adults_inds[screen_inds]
        is_symptomatic = cvu.itruei(sim.people.symptomatic, screened_adult_inds)
        pos_screen     = cvu.n_binomial(self.sensitivity, len(is_symptomatic))
        is_sym_pos     = is_symptomatic[pos_screen]

        not_diagnosed = is_sym_pos[np.isnan(sim.people.date_diagnosed[is_sym_pos])]
        will_quar     = cvu.n_binomial(self.prob_quar_after_fever, len(not_diagnosed))
        final_inds    = not_diagnosed[will_quar]

        sim.people.quarantine(final_inds)
        sim.results['new_quarantined'][t] += len(final_inds)

        print(f'{final_inds} put into quarantine by screening.')

        return


def lift_lockdown_paper_screening_beta():

    pop_scale = 1000
    test_kwargs = {'sympt_test': 0.0, 'quar_test': 100.0, 'sensitivity': 1.0, 'test_delay': 0, 'loss_prob': 0}
    tn = 2000

    pre_lockdown_interventions = [cv.clip_edges(start_day='2020-04-01', end_day='2020-05-04', change={'w': 0.3}), # Reduce work by 70% til May 4
                                  cv.test_num(daily_tests=np.array([200]*180) / pop_scale, start_day='2020-04-01', end_day='2020-05-04', **test_kwargs),
                                  cv.contact_tracing(start_day='2020-04-01', end_day='2020-05-04',
                                                     trace_probs={'h': 1, 's': 0, 'w': 0.8, 'c': 0.1},
                                                     trace_time={'h': 1, 's': 7, 'w': 7, 'c': 7})]

    scenarios = {
        f'noscreen': {
            'name': f'No symptom screening',
            'pars': {
                'interventions': pre_lockdown_interventions + [
                    # new post-lockdown interventions
                    quarantine_severe(start_day='2020-05-04'),
                    cv.test_num(daily_tests=np.array([tn] * 180) / pop_scale, start_day='2020-05-04',
                                end_day='2020-09-30', **test_kwargs),
                    cv.contact_tracing(start_day='2020-05-04', end_day='2020-09-30',
                                       trace_probs={'h': 1, 's': 0, 'w': 0.8, 'c': 0.1},
                                       trace_time={'h': 1, 's': 7, 'w': 7, 'c': 7})]
            }
        },
       f'screen': {
            'name': f'Symptom screening; ineffective face-masks',
            'pars': {
                'interventions': pre_lockdown_interventions + [
                    # new post-lockdown interventions
                    screen(daily_screens=np.array([5e6]*180) / pop_scale,
                           start_day='2020-05-04'),
                    quarantine_severe(start_day='2020-05-04'),
                    cv.test_num(daily_tests=np.array([tn] * 180) / pop_scale, start_day='2020-05-04',
                                end_day='2020-09-30', **test_kwargs),
                    cv.contact_tracing(start_day='2020-05-04', end_day='2020-09-30',
                                       trace_probs={'h': 1, 's': 0, 'w': 0.8, 'c': 0.1},
                                       trace_time={'h': 1, 's': 7, 'w': 7, 'c': 7})]
            }
        },
    }

    metapars = {'n_runs': 1}
    to_plot = [
        'cum_infections',
        'new_infections',
        'cum_symptomatic',
        'cum_severe',
        'cum_critical',
        'cum_deaths',
        #    'cum_diagnoses',
        #    'cum_quarantined',
    ]
    fig_args = dict(figsize=(28, 20))

    allres = {}

    betascens = {'low':0.01, 'medium': 0.015, 'high': 0.018}

    for name, beta in betascens.items:
        sim = make_sim(beta=beta)
        scens = cv.Scenarios(sim=sim, scenarios=scenarios, metapars=metapars)
        df = scens.run(verbose=verbose, debug=False)
        scens.plot(do_save=1, do_show=0, to_plot=to_plot, fig_path=f'results/nigeria_scenarios_paper', n_cols=2, fig_args=fig_args)

        allres[name] = scens.results


    return allres, scens



allres, scens = lift_lockdown_paper_screening_beta()
