import covasim as cv

pars = dict(
    pop_size = 20e3,
    start_day = '2020-02-15',
    pop_infected = 10,
    interventions = cv.test_prob(symp_prob=0.01, asymp_prob=0),
    n_days = 80,
    )

sim = cv.Sim(pars=pars, datafile='nigeria_data.csv')
sim.run()
sim.plot()
