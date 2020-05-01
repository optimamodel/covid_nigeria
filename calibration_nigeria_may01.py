import covasim as cv

# Calibration parameters
symp_prob = 0.007
beta_change = 0.5
beta = 0.012

# Other parameters
pars = dict(
    pop_size = 300e3,
    pop_scale = 1.0,
    start_day = '2020-02-25',
    pop_infected = 200,
    interventions = [
        cv.test_prob(symp_prob=symp_prob, asymp_prob=0),
        cv.change_beta(days='2020-03-29', changes=beta_change, layers=['s','w','c']),
        ],
    n_days = 65,
    rand_seed = 298347,
    beta = beta,
    location = 'nigeria',
    pop_type = 'hybrid',
    )

to_plot = cv.get_sim_plots()
to_plot['Diagnoses'] = ['cum_diagnoses']
to_plot['Deaths'] =  ['cum_deaths']
to_plot.remove('Health outcomes')


sim = cv.Sim(pars=pars, datafile='nigeria_data.csv')
sim.run()
sim.plot(to_plot=to_plot)