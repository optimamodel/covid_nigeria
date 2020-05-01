import covasim as cv

sim = cv.Sim(datafile='nigeria_data.csv')
sim.run()
sim.plot()
