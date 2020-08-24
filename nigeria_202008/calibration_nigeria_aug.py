'''
Create model for Lagos

Data scraped from

https://opendata.ecdc.europa.eu/covid19/casedistribution/csv

on 2020-06-03 using the Covasim data scraper.
'''

import sciris as sc
import pylab as pl
import covasim as cv
import numpy as np
import datetime as dt

do_multi = True
do_plot = False
do_save = True


# good fit (nearly) with
total_pop = 20e6
pop_size = 100e3
pop_scale = int(total_pop/pop_size) #2.0
pop_infected = 150
beta = 0.0111
rel_death_prob = 2.
symp_dates = [cv.date('2020-03-01'), cv.date('2020-04-01'), cv.date('2020-05-01'), cv.date('2020-06-01'),
              cv.date('2020-07-01'), cv.date('2020-08-01'), cv.date('2020-09-01')]
#symp_probs = [0.01, 0.02, 0.03, 0.04, 0.06, 0.06, 0.035]#[0.01, 0.015, 0.03, 0.04, 0.04, 0.04, 0.03]
#daily_tests = np.linspace(0,5000,182)
daily_tests = np.concatenate([np.linspace(0,30,31), # March
                              np.linspace(30,200,30), # April
                              np.linspace(200,600,31), # May
                              np.linspace(600,1200,30), # June
                              np.linspace(1200,1000,31), # July
                              np.linspace(1000,600,31)]) # August
beta_change = 0.6
which = ['lockdown', 'nolockdown'][0]
length = ['aug', 'dec'][0]

if length == 'aug': end_day = '2020-08-22'
elif length == 'dec': end_day = '2020-12-31'

pars = dict(
    pop_size = pop_size,
    pop_scale = pop_scale,
    pop_infected = pop_infected,
    start_day = '2020-03-01',
    end_day=end_day,
    rand_seed = 111,
    beta = beta,
    verbose = 0.1,
    location = 'nigeria',
    pop_type = 'hybrid',
    rel_death_prob = rel_death_prob,
    rescale      = True,
)

#testints = [cv.test_prob(symp_prob=symp_probs[i], start_day=symp_dates[i], end_day=symp_dates[i+1]-dt.timedelta(days=1), test_delay=4, do_plot=False) for i in range(6)]
testints = [cv.test_num(daily_tests=daily_tests, test_delay=3, do_plot=False)]
pars['interventions'] = testints + [cv.contact_tracing(start_day='2020-03-01',
                                                       trace_probs={'h': 0.5, 's': 0.2, 'w': 0.2, 'c': 0.0},
                                                       trace_time={'h': 3, 's': 10, 'w': 10, 'c': 14}, do_plot=False)]

if which=='lockdown':
    pars['interventions'] +=  [
#        cv.change_beta(days=['2020-03-29', '2020-07-06'], changes=[0, 0.9], layers=['s'], do_plot=beta_change<1.0),
        cv.change_beta(days=['2020-03-29'], changes=[0.2], layers=['s'], do_plot=beta_change<1.0),
        #cv.change_beta(days=['2020-03-29'], changes=[0.75], layers=['w'], do_plot=beta_change < 1.0),
        #cv.change_beta(days=['2020-03-29'], changes=[0.75], layers=['c'], do_plot=beta_change < 1.0),
        cv.change_beta(days=['2020-03-29','2020-05-04'], changes=[beta_change, 0.75], layers=['c'], do_plot=beta_change<1.0),
        cv.change_beta(days=['2020-03-29','2020-05-04'], changes=[beta_change, 0.75], layers=['w'], do_plot=beta_change<1.0),

    ]

# Create sim and run
sim = cv.Sim(pars=pars, datafile='Lagos.csv')
for col in ['new_deaths', 'cum_deaths']:
    total_deaths = 992
    lagos_deaths = 201
    factor = lagos_deaths/total_deaths # Adjust for Lagos vs. Nigeria, from https://covid19.ncdc.gov.ng/
    sim.data.loc[:, col] = factor*sim.data.loc[:, col]
if not do_multi: sim.run()

if do_multi:
    msim = cv.MultiSim(base_sim=sim)
    msim.run(n_runs=100, reseed=True, noise=0)
    #msim.reduce()


# Plotting
to_plot = sc.objdict({
    'Diagnoses': ['cum_diagnoses'],
    'Daily diagnoses': ['new_diagnoses'],
    'Deaths': ['cum_deaths'],
    'Daily deaths': ['new_deaths'],
    'Total infections': ['cum_infections'],
#    'Cumulative tests': ['cum_tests'],
    'New infections per day': ['new_infections'],
#    'New tests': ['new_tests'],
    })


if do_plot:
    if do_multi:
        msim.plot(to_plot=to_plot, do_save=True, do_show=False, fig_path=f'nigeria_calibration_{which}_{length}.png',
             legend_args={'loc': 'upper left'}, axis_args={'hspace':0.4}, interval=28)
    else:
        sim.plot(to_plot=to_plot, do_save=True, do_show=False, fig_path=f'nigeria_calibration_{which}_{length}.png',
             legend_args={'loc': 'upper left'}, axis_args={'hspace':0.4}, interval=28)

if do_save:
    if do_multi: msim.save(f'nigeria_{which}_{length}.sim')
    else:         sim.save(f'nigeria_{which}_{length}.sim')
