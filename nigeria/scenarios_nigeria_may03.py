'''
Simulate COVID-19 in Nigeria
'''

#%% Imports and settings
import matplotlib
matplotlib.use('Agg')
import sciris as sc
import covasim as cv
import numpy as np
import pylab as pl
import pandas as pd
import covasim.utils as cvu

# Settings
seed = 1
verbose = 1


class quarantine_severe(cv.Intervention):
    '''Quarantine people with severe symptoms'''
    def __init__(self, start_day=0, end_day=None, do_plot=None):
        super().__init__(do_plot=do_plot)
        self.start_day   = start_day
        self.end_day     = end_day
        #self.do_plot     = do_plot
        self._store_args()
        self.initialized     = False

        return

    def initialize(self, sim):
        ''' Fix the dates '''
        self.start_day = sim.day(self.start_day)
        self.end_day   = sim.day(self.end_day)
        self.days      = [self.start_day, self.end_day]
        self.initialized = True
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
        self.initialized     = False
        self.screen_quarantined = 0

        return

    def initialize(self, sim):
        ''' Fix the dates '''
        self.start_day = sim.day(self.start_day)
        self.end_day   = sim.day(self.end_day)
        self.days      = [self.start_day, self.end_day]
        self.initialized = True
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

        #print(f'{final_inds} put into quarantine by screening.')

        return


def lift_lockdown_paper_screening_scens():

    sim = sc.loadobj('/Users/robynstuart/Documents/git/covid_apps/nigeria/nigeria_lockdown_may03_long.sim')

    n_adults = sim['pop_size']/2
    pop_scale = sim['pop_scale']

    test_kwargs = {'symp_test': 0.0, 'quar_test': 100.0, 'sensitivity': 1.0, 'test_delay': 0, 'loss_prob': 0}
    tn = 2000

    number_screened = [int(n_adults*x/10) for x in range(1,6,1)] # Number that could be screened - 10-50% of the population
    efficacy = [x/10 for x in range(1,6,1)]

    number_screened = [int(n_adults*.5)]
    efficacy = [0.5]

    scenarios = {
       f'screen{sc}_eff{ef}': {
            'name': f'Screen {sc} with {ef} efficacy',
            'pars': {
                'interventions': [
                    # new post-lockdown interventions
                    screen(daily_screens=np.array([sc]*180) / pop_scale,
                           sensitivity=ef,
                           start_day='2020-05-04'),
                    quarantine_severe(start_day='2020-05-04'),
                    cv.test_num(daily_tests=np.array([tn] * 180) / pop_scale, start_day='2020-05-04',
                                end_day='2020-09-30', **test_kwargs),
                    cv.contact_tracing(start_day='2020-05-04', end_day='2020-09-30',
                                       trace_probs={'h': 1, 's': 0, 'w': 0.8, 'c': 0.1},
                                       trace_time={'h': 1, 's': 7, 'w': 7, 'c': 7})]
            }
        } for sc in number_screened for ef in efficacy
    }

    baseline = {
       f'noscreen': {
            'name': f'No screening',
            'pars': {
                'interventions': [
                    # new post-lockdown interventions
                    quarantine_severe(start_day='2020-05-04'),
                    cv.test_num(daily_tests=np.array([tn] * 180) / pop_scale, start_day='2020-05-04',
                                end_day='2020-09-30', **test_kwargs),
                    cv.contact_tracing(start_day='2020-05-04', end_day='2020-09-30',
                                       trace_probs={'h': 1, 's': 0, 'w': 0.8, 'c': 0.1},
                                       trace_time={'h': 1, 's': 7, 'w': 7, 'c': 7})]
            }
        }
    }

    metapars = {'n_runs': 1}
    allscenarios = sc.mergedicts(baseline, scenarios)
    scens = cv.Scenarios(sim=sim, scenarios=allscenarios, metapars=metapars)
    df = scens.run(verbose=verbose, debug=True)

    return df, scens



df, scens = lift_lockdown_paper_screening_scens()
#scens.save('nigeria.scens') # Save for analysis script


to_plot = [
    'cum_infections',
    'new_infections',
    'cum_symptomatic',
    'cum_severe',
    'cum_critical',
    'cum_deaths',
    'cum_diagnoses',
    'cum_quarantined',
]
fig_args = dict(figsize=(28, 20))

scens.plot(do_save=1, do_show=0) #, to_plot=to_plot, fig_path='nigeria_scenarios_paper_may05.png', n_cols=2, fig_args=fig_args)
