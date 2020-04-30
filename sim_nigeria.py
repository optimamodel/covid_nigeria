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

                    cv.screen_test_trace(daily_tests=np.array([tn]*180) / pop_scale, start_day='2020-05-04', **test_kwargs),
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
                    # #old interventions
                    cv.clip_edges(start_day='2020-04-01', end_day='2020-05-04', change={'w': 0.3}), # Reduce work by 70% til May 4
                    cv.test_num(daily_tests=np.array([200]*180) / pop_scale, start_day='2020-04-01', end_day='2020-05-04', **test_kwargs),
                    cv.contact_tracing(start_day='2020-04-01', end_day='2020-05-04',
                                       trace_probs={'h': 1, 's': 0, 'w': 0.8, 'c': 0.1},
                                       trace_time={'h': 1, 's': 7, 'w': 7, 'c': 7}),

                    # new post-lockdown interventions
                    cv.screen_test_trace(daily_tests=np.array([tn]*180) / pop_scale,
                                         daily_screens=np.array([20000]*180) / pop_scale,
                                         severe_test=100.0,
                                         screen_test=4.,
                                         start_day='2020-05-04', **test_kwargs),
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




class screen_test_trace(cv.Intervention):
    '''
    Implement targeted testing in combination with symptom screening
    '''
    def __init__(self, daily_screens, daily_tests, severe_test=100.0, screen_test=4., quar_test=1.0, sensitivity=1.0, loss_prob=0,
                 test_delay=0, fever_prev=0.02, fever_prev_covid=0.5, prob_quar_after_fever=0.7, start_day=0, end_day=None, do_plot=None):
        super().__init__(do_plot=do_plot)
        self.daily_screens = daily_screens  # Should be a list of length matching time
        self.daily_tests = daily_tests      # Should be a list of length matching time
        self.severe_test  = severe_test     # Probability of getting tested if you have severe symptoms
        self.screen_test = screen_test      # Probability of getting tested if you've had a temperature check and have a fever
        self.quar_test   = quar_test        # Probability of getting tested if you're in quarantine

        self.fever_prev_covid  = fever_prev_covid    # Prevalence of fever in symptomatic COVID patients
        self.fever_prev  = fever_prev                # Prevalence of fever overall
        self.screen_trace_probs = screen_trace_probs # Probability you'll call your contacts and tell them you have a fever (but not a confirmed diagnosis)
        self.screen_trace_time = screen_trace_time   # Time it'll take you to get round to making those calls
        self.prob_quar_after_fever = prob_quar_after_fever # Probability that you'll quarantine after you've had your temp checked and have a fever (but not a confirmed diagnosis)

        self.sensitivity = sensitivity      # Sensitivity of the PCR test
        self.loss_prob   = loss_prob        # Probability of LTFU
        self.test_delay  = test_delay       # Delay in getting test results

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

        # Process daily tests -- has to be here rather than init so have access to the sim object
        if isinstance(self.daily_tests, (pd.Series, pd.DataFrame)):
            start_date = sim['start_day'] + dt.timedelta(days=self.start_day)
            end_date = self.daily_tests.index[-1]
            dateindex = pd.date_range(start_date, end_date)
            self.daily_tests = self.daily_tests.reindex(dateindex, fill_value=0).to_numpy()

        # Check that there are still tests
        rel_t = t - self.start_day
        if rel_t < len(self.daily_tests):
            n_tests = self.daily_tests[rel_t]  # Number of tests for this day
            if not (n_tests and pl.isfinite(n_tests)): # If there are no tests today, abort early
                return
            else:
                sim.results['new_tests'][t] += n_tests
        else:
            return

        # Set the prevalence of fever in the population. Assume that some proportion of people with COVID symptoms have fever
        # and that some proportion of the rest of the population will record temperatures above 37.5
        has_fever   = np.full(sim.n, False, dtype=bool)

        symptomatic_covid_inds  = cvu.true(sim.people.symptomatic)
        symptomatic_covid_with_fever = cvu.n_binomial(self.fever_prev_covid, len(symptomatic_covid_inds))
        symptomatic_covid_with_fever_inds = symptomatic_covid_inds[symptomatic_covid_with_fever]

        extra_fever = cvu.choose(self.fever_prev*n, sim.n) # Prevalence of fever generally, not necessarily connected to COVID
        has_fever[symptomatic_covid_with_fever_inds] = True
        has_fever[extra_fever] = True

        # Screen people and figure out who screens positive
        screened = np.full(sim.n, False, dtype=bool) # No-one has been screened this timestep
        screen_inds = cvu.choose(daily_screens[t], sim.n) # Who will we screen today - untargeted
        screened[screen_inds] = True
        screened_pos_inds  = cvu.true(screened_people & has_fever)

        # Do some light quarantine and contact tracing for those who've screened positive
        will_quarantine = cvu.binomial_filter(self.prob_quar_after_fever, arr)
        sim.people.quarantine(will_quarantine)
        sim.people.trace(screened_pos_inds, self.screen_trace_probs, self.screen_trace_time)

        test_probs = np.zeros(sim.n) # Begin by assigning zero testing probability to everyone

        severe_inds  = cvu.true(sim.people.severe)
        quar_inds    = cvu.true(sim.people.quarantined)
        diag_inds    = cvu.true(sim.people.diagnosed)

        test_probs[screened_pos_inds]  = self.screen_test # Probability of being tested if you were screened (i.e., got your temp checked and had fever)
        test_probs[severe_inds] = self.severe_test
        test_probs[quar_inds]   = self.quar_test
        test_probs[diag_inds]   = 0.

        test_inds = cvu.choose_w(probs=test_probs, n=n_tests, unique=False)

        sim.people.test(test_inds, self.sensitivity, loss_prob=self.loss_prob, test_delay=self.test_delay) # This sets their diagnosed date


        return

