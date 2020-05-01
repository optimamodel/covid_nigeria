'''
Calibration to Nigeria/Lagos data.

Data scraped from

https://opendata.ecdc.europa.eu/covid19/casedistribution/csv

using the Covasim data scraper.
'''

import covasim as cv

# Calibration parameters -- "default" uses default sim values, "calibrated" uses Nigeria-specific ones
which = ['default', 'calibrated'][1]
if which == 'default':
    symp_prob = 0.0015
    beta_change = 1.0
    beta = 0.015
    pop_infected = 20
elif which == 'calibrated':
    symp_prob = 0.005
    beta_change = 0.6
    beta = 0.011
    pop_infected = 100

# Other parameters
pars = dict(
    pop_size = 500e3,
    pop_scale = 2.0,
    start_day = '2020-03-01',
    pop_infected = pop_infected,
    interventions = [
        cv.test_prob(symp_prob=symp_prob, asymp_prob=0, do_plot=False),
        cv.change_beta(days='2020-03-29', changes=beta_change, layers=['s','w','c'], do_plot=beta_change<1.0),
        ],
    n_days = 65,
    rand_seed = 111,
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
sim.run()

# Plotting
to_plot = cv.get_sim_plots()
to_plot['Diagnoses'] = ['cum_diagnoses']
to_plot['Deaths'] =  ['cum_deaths']
to_plot.remove('Health outcomes')
sim.plot(to_plot=to_plot, use_grid=False)