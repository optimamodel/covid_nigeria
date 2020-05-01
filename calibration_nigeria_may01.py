import covasim as cv

pars = dict(
    pop_size = 20e3,
    pop_scale = 10,
    start_day = '2020-03-01',
    pop_infected = 5,
    interventions = cv.test_prob(symp_prob=0.01, asymp_prob=0),
    n_days = 60,
    )

sim = cv.Sim(pars=pars, datafile='nigeria_data.csv')
sim.run()
sim.plot()
