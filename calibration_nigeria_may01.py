import covasim as cv

# Calibration parameters
symp_prob = 0.005
beta_change = 1.0# 0.5
beta = 0.015 # 0.010

# Other parameters
pars = dict(
    pop_size = 250e3,
    pop_scale = 2.0,
    start_day = '2020-02-25',
    pop_infected = 10, # 120,
    interventions = [
        cv.test_prob(symp_prob=symp_prob, asymp_prob=0, do_plot=False),
        cv.change_beta(days='2020-03-29', changes=beta_change, layers=['s','w','c']),
        ],
    n_days = 65,
    rand_seed = 2048,
    beta = beta,
    location = 'nigeria',
    pop_type = 'hybrid',
    )

to_plot = cv.get_sim_plots()
to_plot['Diagnoses'] = ['cum_diagnoses']
to_plot['Deaths'] =  ['cum_deaths']
to_plot.remove('Health outcomes')


sim = cv.Sim(pars=pars, datafile='nigeria_data.csv')
# Adjust for Lagos vs. Nigeria, from https://covid19.ncdc.gov.ng/
for col in ['new_diagnoses', 'cum_diagnoses', 'new_deaths', 'cum_deaths']:
    total_deaths = 51
    lagos_deaths = 21
    factor = lagos_deaths/total_deaths
    sim.data.loc[:, col] = factor*sim.data.loc[:, col]
sim.run()
sim.plot(to_plot=to_plot, use_grid=False)