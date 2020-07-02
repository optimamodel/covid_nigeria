import pandas as pd
import numpy as np
import pylab as pl
import sciris as sc
import covasim as cv
import optuna as op


def create_sim(x):

    beta = x[0]
    #beta_change = x[1]
    rel_crit_prob = x[1]
    #symp_prob = x[2]
    symp_prob_prelockdown = x[2]
    symp_prob_lockdown = x[3]
    symp_prob_postlockdown = x[4]

    start_day = '2020-03-01'
    lockdown_start = '2020-03-29'
    lockdown_end = '2020-05-04'
    end_day   = '2020-06-30'
    datafile = 'Lagos.csv'

    # Set the parameters
    total_pop    = 20e6 # Lagos population size
    pop_size     = 100e3 # Actual simulated population
    pop_scale    = int(total_pop/pop_size) #2.0
    pop_infected = 150
    pop_type     = 'hybrid'

    pars = sc.objdict(
        pop_size     = pop_size,
        pop_scale    = pop_scale,
        pop_infected = pop_infected,
        pop_type     = pop_type,
        start_day    = start_day,
        end_day      = end_day,
        beta         = beta,
        rel_crit_prob=rel_crit_prob,
        rescale      = True,
        verbose      = 0.1,
    )

    pars['interventions'] = [
        cv.test_prob(symp_prob=symp_prob_prelockdown,   asymp_prob=0, start_day=start_day,      end_day=lockdown_start, do_plot=False),
        cv.test_prob(symp_prob=symp_prob_lockdown,      asymp_prob=0, start_day=lockdown_start, end_day=lockdown_end,   do_plot=False),
        cv.test_prob(symp_prob=symp_prob_postlockdown,  asymp_prob=0, start_day=lockdown_end,                           do_plot=False),
        #cv.test_prob(symp_prob=symp_prob,  asymp_prob=0, start_day=0, do_plot=False),
        cv.contact_tracing(start_day=start_day, trace_probs={'h': 1, 's': 0, 'w': 0.8, 'c': 0.0}, trace_time={'h': 1, 's': 7, 'w': 7, 'c': 7}, do_plot=False),
        cv.change_beta(days=[lockdown_start], changes=[0], layers='s', do_plot=False),
        cv.change_beta(days=[lockdown_start, lockdown_end], changes=[0.5, 0.8], layers='w', do_plot=False),
        cv.change_beta(days=[lockdown_start, lockdown_end], changes=[0.5, 0.8], layers='c', do_plot=False),
    ]

    # Create the baseline simulation
    sim = cv.Sim(pars=pars, datafile=datafile, location='nigeria')

    # Rescale deaths from Nigeria total to Lagos total
    for col in ['new_deaths', 'cum_deaths']:
        total_deaths = 573
        lagos_deaths = 127
        factor = lagos_deaths / total_deaths  # Adjust for Lagos vs. Nigeria, from https://covid19.ncdc.gov.ng/
        sim.data.loc[:, col] = factor * sim.data.loc[:, col]

    return sim



def objective(x):
    ''' Define the objective function we are trying to minimize '''

    # Create and run the sim
    sim = create_sim(x)
    sim.run()
    fit = sim.compute_fit()

    return fit.mismatch


def get_bounds():
    ''' Set parameter starting points and bounds '''
    pdict = sc.objdict(
        beta         = dict(best=0.012, lb=0.008, ub=0.015),
        #beta_change = dict(best=0.5,  lb=0.3,   ub=0.7),
        #symp_prob = dict(best=0.01,  lb=0.005,   ub=0.015),
        rel_crit_prob = dict(best=1.1,  lb=1.,   ub=1.6),
        symp_prob_prelockdown = dict(best=0.01,  lb=0.000,   ub=0.03),
        symp_prob_lockdown = dict(best=0.02,  lb=0.005,   ub=0.1),
        symp_prob_postlockdown = dict(best=0.04,  lb=0.005,   ub=0.15),
    )

    # Convert from dicts to arrays
    pars = sc.objdict()
    for key in ['best', 'lb', 'ub']:
        pars[key] = np.array([v[key] for v in pdict.values()])

    return pars, pdict.keys()


#%% Calibration

name      = 'covasim_nigeria_calibration'
storage   = f'sqlite:///{name}.db'
n_trials  = 100
n_workers = 8

pars, pkeys = get_bounds() # Get parameter guesses


def op_objective(trial):

    pars, pkeys = get_bounds() # Get parameter guesses
    x = np.zeros(len(pkeys))
    for k,key in enumerate(pkeys):
        x[k] = trial.suggest_uniform(key, pars.lb[k], pars.ub[k])

    return objective(x)


def worker():
    study = op.load_study(storage=storage, study_name=name)
    return study.optimize(op_objective, n_trials=n_trials)


def run_workers():
    return sc.parallelize(worker, n_workers)


def make_study():
    try: op.delete_study(storage=storage, study_name=name)
    except: pass
    return op.create_study(storage=storage, study_name=name)


def calibrate():
    ''' Perform the calibration '''
    make_study()
    run_workers()
    study = op.load_study(storage=storage, study_name=name)
    output = study.best_params
    return output, study


def savejson(study):
    dbname = 'calibrated_parameters_nigeria'

    sc.heading('Making results structure...')
    results = []
    failed_trials = []
    for trial in study.trials:
        data = {'index':trial.number, 'mismatch': trial.value}
        for key,val in trial.params.items():
            data[key] = val
        if data['mismatch'] is None:
            failed_trials.append(data['index'])
        else:
            results.append(data)
    print(f'Processed {len(study.trials)} trials; {len(failed_trials)} failed')

    sc.heading('Making data structure...')
    keys = ['index', 'mismatch'] + pkeys
    data = sc.objdict().make(keys=keys, vals=[])
    for i,r in enumerate(results):
        for key in keys:
            data[key].append(r[key])
    df = pd.DataFrame.from_dict(data)

    order = np.argsort(df['mismatch'])
    json = []
    for o in order:
        row = df.iloc[o,:].to_dict()
        rowdict = dict(index=row.pop('index'), mismatch=row.pop('mismatch'), pars={})
        for key,val in row.items():
            rowdict['pars'][key] = val
        json.append(rowdict)
    sc.savejson(f'{dbname}.json', json, indent=2)

    return


if __name__ == '__main__':

    do_save = True

    to_plot = ['cum_infections', 'new_infections', 'cum_tests', 'new_tests', 'cum_diagnoses', 'new_diagnoses', 'cum_deaths', 'new_deaths']

    # # Plot initial
    print('Running initial...')
    pars, pkeys = get_bounds() # Get parameter guesses
    sim = create_sim(pars.best)
    sim.run()
    sim.plot(to_plot=to_plot, do_save=True, do_show=False, fig_path=f'nigeria_calibration_optuna.png')
    pl.gcf().axes[0].set_title('Initial parameter values')
    objective(pars.best)
    pl.pause(1.0) # Ensure it has time to render

    # Calibrate
    print('Starting calibration for {state}...')
    T = sc.tic()
    pars_calib, study = calibrate()
    sc.toc(T)

    # Plot result
    print('Plotting result...')
    sim = create_sim([pars_calib['beta'], pars_calib['rel_crit_prob'], pars_calib['symp_prob_prelockdown'], pars_calib['symp_prob_lockdown'], pars_calib['symp_prob_postlockdown']])
    #sim = create_sim([pars_calib['beta'], pars_calib['beta_change'], pars_calib['symp_prob_prelockdown'], pars_calib['symp_prob_lockdown'], pars_calib['symp_prob_postlockdown']])
    #sim = create_sim([pars_calib['beta'], pars_calib['beta_change'], pars_calib['symp_prob']])
    sim.run()
    sim.plot(to_plot=to_plot, do_save=True, do_show=False, fig_path=f'nigeria_calibration_optuna.png')
    pl.gcf().axes[0].set_title('Calibrated parameter values')

    if do_save:
        savejson(study)


print('Done.')