import covasim as cv

pars = dict(
    pop_size = 500e3,
    pop_scale = 0.5,
    start_day = '2020-02-25',
    pop_infected = 50,
    interventions = cv.test_prob(symp_prob=0.01, asymp_prob=0),
    n_days = 65,
    rand_seed = 2,
    # beta = 0.012,s
    location = 'nigeria',
    pop_type = 'hybrid',
    )

# Simple
# pars = dict(pop_size=200e3,
#             start_day = '2020-02-15',
#             pop_scale = 1,
#               n_days=60,
#               rand_seed=1,
#               pop_type = 'hybrid',
#               )

to_plot = cv.get_sim_plots()
to_plot['Diagnoses'] = ['cum_diagnoses']
to_plot['Deaths'] =  ['cum_deaths']
to_plot.remove('Health outcomes')


sim = cv.Sim(pars=pars, datafile='nigeria_data.csv')
sim.run()
sim.plot(to_plot=to_plot)
