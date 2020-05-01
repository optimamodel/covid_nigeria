import covasim as cv

pars = dict(
    pop_size = 50e3,
    pop_scale = 1,
    start_day = '2020-02-15',
    # pop_infected = 5,
    # interventions = cv.test_prob(symp_prob=0.01, asymp_prob=0),
    n_days = 75,
    # rand_seed = 2,
    # beta = 0.012,
    # location = 'nigeria',
    # pop_type = 'hybrid',
    )

# Works
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
