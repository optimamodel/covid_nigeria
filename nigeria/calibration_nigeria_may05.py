'''
Create model for Lagos

Data scraped from

https://opendata.ecdc.europa.eu/covid19/casedistribution/csv

on 2020-05-03 using the Covasim data scraper.
'''

import sciris as sc
import pylab as pl
import covasim as cv
import numpy as np

cv.check_version('0.30.3')
cv.git_info('covasim_version.json')

do_save = False

pop_size = 500e3
pop_scale = 2.0
pop_infected = 100
beta = 0.012
rel_crit_prob = 1.
symp_prob = 0.006
beta_change = 0.6
which = ['lockdown', 'nolockdown'][0]
length = ['short', 'long'][1]

if length == 'short': end_day = '2020-05-04'
elif length == 'long': end_day = '2020-05-31'

pars = dict(
    pop_size = pop_size,
    pop_scale = pop_scale,
    pop_infected = pop_infected,
    start_day = '2020-03-01',
    end_day=end_day,
    rand_seed = 111,
    beta = beta,
    location = 'nigeria',
    pop_type = 'hybrid',
    rel_crit_prob = rel_crit_prob,
)

pars['interventions'] = [
    cv.test_prob(symp_prob=symp_prob, asymp_prob=0, start_day=0, end_day='2020-05-04', do_plot=False),
    cv.contact_tracing(start_day='2020-03-01',
                       trace_probs={'h': 1, 's': 0, 'w': 0.8, 'c': 0.1},
                       trace_time={'h': 1, 's': 7, 'w': 7, 'c': 7})]

if which=='lockdown':
    pars['interventions'] +=  [
        cv.test_prob(symp_prob=symp_prob, asymp_prob=0, start_day=0, end_day='2020-05-04', do_plot=False),
        cv.change_beta(days=['2020-03-29'], changes=[beta_change], layers=['s'], do_plot=beta_change<1.0),
        cv.change_beta(days=['2020-03-29','2020-05-04'], changes=[beta_change, 0.95], layers=['c'], do_plot=beta_change<1.0),
        cv.change_beta(days=['2020-03-29','2020-05-04'], changes=[beta_change, 1.00], layers=['w'], do_plot=beta_change<1.0)]


# Create sim and run
sim = cv.Sim(pars=pars, datafile='nigeria_data.csv')
for col in ['new_diagnoses', 'cum_diagnoses', 'new_deaths', 'cum_deaths']:
    total_deaths = 51
    lagos_deaths = 21
    factor = lagos_deaths/total_deaths # Adjust for Lagos vs. Nigeria, from https://covid19.ncdc.gov.ng/
    sim.data.loc[:, col] = factor*sim.data.loc[:, col]
sim.run()

if length=='short':
    msim = cv.MultiSim(base_sim=sim)
    msim.run(n_runs=6, noise=0.1)
    msim.reduce()
    sim = msim.base_sim


# Plotting
to_plot = sc.objdict({
    'Diagnoses': ['cum_diagnoses'],
    'Deaths': ['cum_deaths'],
    'Total infections': ['cum_infections'],
    'New infections per day': ['new_infections'],
    })
sim.plot(to_plot=to_plot, do_save=True, do_show=False, fig_path=f'nigeria_calibration_{which}_{length}.png',
         legend_args={'loc': 'upper left'}, axis_args={'hspace':0.4})

if do_save:
    sim.save(f'nigeria_{which}_{length}.sim')
