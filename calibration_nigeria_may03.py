'''
Calibration to Nigeria/Lagos data.

Data scraped from

https://opendata.ecdc.europa.eu/covid19/casedistribution/csv

on 2020-05-03 using the Covasim data scraper.
'''

import sciris as sc
import pylab as pl
import covasim as cv

cv.check_version('0.30.3')
cv.git_info('covasim_version.json')

do_save = True

# Calibration parameters -- "default" uses default sim values, "calibrated" uses Nigeria-specific ones

rel_crit_prob = 5.0
diag_factor = 0.8
pop_size = 200e3
pop_scale = 5.0
pop_infected = 40
symp_prob = 0.004
beta = 0.0125
which = ['lockdown', 'nolockdown'][0]
if which == 'lockdown':
    h_beta_change = 1.2 # or 1.0
    s_beta_change = 0.2 # or 0.5
    w_beta_change = 0.5 # or 0.5
    c_beta_change = 0.8 # or 0.5
elif which == 'nolockdown':
    h_beta_change = 1.0
    s_beta_change = 1.0
    w_beta_change = 1.0
    c_beta_change = 1.0


# Other parameters
pars = dict(
    pop_size = pop_size,
    pop_scale = pop_scale,
    rescale = False,
    start_day = '2020-03-01',
    end_day = '2020-05-04',
    pop_infected = pop_infected,
    interventions = [
        cv.test_prob(symp_prob=symp_prob, asymp_prob=0, start_day=0, do_plot=False),
        cv.change_beta(days=['2020-03-29'], changes=[h_beta_change], layers=['h'], do_plot=h_beta_change<1.0),
        cv.change_beta(days=['2020-03-29'], changes=[s_beta_change], layers=['s'], do_plot=s_beta_change<1.0),
        cv.change_beta(days=['2020-03-29'], changes=[w_beta_change], layers=['w'], do_plot=w_beta_change<1.0),
        cv.change_beta(days=['2020-03-29'], changes=[c_beta_change], layers=['c'], do_plot=c_beta_change<1.0),
        ],
    rand_seed = 1,
    beta = beta,
    location = 'nigeria',
    pop_type = 'hybrid',
    )

# Create sim and run
sim = cv.Sim(pars=pars, datafile='nigeria_data.csv')
for col in ['new_diagnoses', 'cum_diagnoses', 'new_deaths', 'cum_deaths']:
    total_deaths = 51
    lagos_deaths = 21
    factor = lagos_deaths/total_deaths # Adjust for Lagos vs. Nigeria, from https://covid19.ncdc.gov.ng/
    sim.data.loc[:, col] = factor*sim.data.loc[:, col]

msim = cv.MultiSim(base_sim=sim)
msim.run(n_runs=6, noise=0.1)
msim.reduce()
sim = msim.base_sim

# sim.run()

# Plotting
to_plot = sc.objdict({
    'Diagnoses': ['cum_diagnoses'],
    'Deaths': ['cum_deaths'],
    'Total infections': ['cum_infections'],
    'New infections per day': ['new_infections'],
    })
sim.plot(to_plot=to_plot, do_save=False, do_show=True, legend_args={'loc': 'upper left'}, axis_args={'hspace':0.4})

if do_save:
    pl.savefig(f'nigeria_calibration_{which}_may03.png', dpi=150)
    sim.save(f'nigeria_{which}_may03.sim')

