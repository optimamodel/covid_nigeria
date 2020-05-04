'''
Calibration to Nigeria/Lagos data.

Data scraped from

https://opendata.ecdc.europa.eu/covid19/casedistribution/csv

on 2020-05-03 using the Covasim data scraper.
'''

import covasim as cv

cv.check_version('0.30.3')
cv.git_info('covasim_version.json')

# Calibration parameters -- "default" uses default sim values, "calibrated" uses Nigeria-specific ones
which = ['default', 'calibrated'][1]
if which == 'default':
    symp_prob = 0.0015
    beta_change = 1.0
    beta = 0.015
    pop_infected = 20
elif which == 'calibrated':
    symp_prob = 0.005
    beta_change = 0.4
    beta = 0.014
    pop_infected = 50

# Other parameters
pars = dict(
    pop_size = 500e3,
    pop_scale = 2.0,
    rescale = False,
    start_day = '2020-03-01',
    end_day = '2020-05-04',
    pop_infected = pop_infected,
    interventions = [
        cv.test_prob(symp_prob=symp_prob, asymp_prob=0, start_day=0, do_plot=False),
        cv.change_beta(days=['2020-03-29','2020-05-05'], changes=[beta_change, 0.95], layers=['s','w','c'], do_plot=beta_change<1.0),
        ],
    rand_seed = 3,
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
sim.plot(to_plot=to_plot, do_save=False, do_show=True)
sim.save('nigeria.sim')

