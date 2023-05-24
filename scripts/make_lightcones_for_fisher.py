import py21cmfast as p21c
import os, shutil
import glob
import numpy as np
import time
from joblib import Parallel, delayed
import argparse
import configparser
import multiprocessing

import py21cmfish as p21fish

import logging
logger = logging.getLogger("21cmFAST")
logger.setLevel(logging.INFO)

print(f"21cmFAST version is {p21c.__version__}")

# ==============================================================================
# python make_lightcones_for_fisher.py ../21cmFAST_config_files/Park19.config --dry_run
# TODO =====
# Took ---- Finished making lightcones, took 15.86 hours ---- for ETHOS.
# Took 11 mins to make PS
#
#
# python scripts/make_lightcones_for_fisher.py 21cmFAST_config_files/ETHOS.config --num_cores 2 --h_PEAK 0 --random_seed $r
# ==============================================================================
# ==============================================================================
#
# Script to create set of 21cmFAST simulations for Fisher matrix analysis.
#   Loads a configuration file of default parameters, and parameters to vary
#
# ==============================================================================
# ==============================================================================
#
# Import config files
config = configparser.ConfigParser(delimiters=':')
config.optionxform = str

# Managing arguments with argparse (see http://docs.python.org/howto/argparse.html)
parser = argparse.ArgumentParser()
# ---- required arguments ---- :
parser.add_argument("config_file", type=str, help="Path to config file")
# ---- optional arguments ----
parser.add_argument("--h_PEAK", type=float, help="h_PEAK for ETHOS model, only used if USE_ETHOS = True [default = vary]")
parser.add_argument("--N_THREADS", type=int, help="Number of threads for 21cmFAST [default = 1, clogs memory if you use too many]")
parser.add_argument("--num_cores", type=int, help="Number of cores to run on [default = n_cpu - 1]")
parser.add_argument("--q_scale", type=float, help="Percentage step for the parameters [default = 3%]")
parser.add_argument("--random_seed", type=int, help="Random seed [default = 12345]")
# ---- flags ------
parser.add_argument("--save_Tb", action='store_true', help="Save BrightnessTemp boxes [default = False]")
parser.add_argument("--fix_astro_params", action='store_true', help="Fix astro params (only vary k_peak, h_peak for ETHOS runs) [default = False]")
parser.add_argument("--test_linear", action='store_true', help="Test linearity of PS derivatives by creating lightcones on a wider grid of parameters [default = False]")
parser.add_argument("--clobber", action='store_true', help="make new lightcones [default = False]")
parser.add_argument("--dry_run", action='store_true', help="Just print the parameters, don't run anything [default = False]")

args = parser.parse_args()
# ==============================================================================
# Run Parameters
num_cores = multiprocessing.cpu_count() - 1
if args.num_cores:
    num_cores  = args.num_cores
logger.info(f'Running on {num_cores} cores')

N_THREADS = 1
if args.N_THREADS:
    N_THREADS  = args.N_THREADS
logger.info(f'Running on {N_THREADS} threads')

q_scale = 3
if args.q_scale:
    q_scale  = args.q_scale
logger.info(f'Calculating derivatives at {q_scale} percent from fiducial')

if args.h_PEAK is not None:
    h_PEAK  = args.h_PEAK
    fix_h_PEAK = True
    h_peaks = [h_PEAK]
    logger.info(f'Running with fixed h_peak = {h_PEAK}')
else:
    fix_h_PEAK = False
    h_PEAK = 0. # default
    h_peaks = np.arange(0., 1.1, 0.1)
    logger.info(f'Running with varied h_peak [if USE_ETHOS = True]')
logger.info(f'Will make lightcones for h_peak={h_peaks}')

save_Tb = False
if args.save_Tb:
    save_Tb = True
    logger.info(f'Saving BrightnessTemp coeval boxes')

vary_array = np.array([-1,1])
if args.test_linear:
    vary_array = np.arange(-10,11)
    vary_array = np.delete(vary_array,np.where(vary_array==0))
    logger.info(f'Testing linearity of derivatives on a larger grid +/-{q_scale*np.max(vary_array)}% of fiducial')

fix_astro_params = False
if args.fix_astro_params:
    fix_astro_params = True
    logger.info(f'Fixing astro params')

clobber = False
if args.clobber:
    clobber = True
    logger.info(f'Clobber = True - making new lightcones')

random_seed = 12345
if args.random_seed:
    random_seed  = args.random_seed
logger.info(f'Using random_seed = {random_seed}')

# ==============================================================================
# Get config
config_file = args.config_file
assert os.path.exists(config_file), f'{config_file} does not exist!'
print(config_file)
logger.info(f'Running {config_file}...')
config.read(config_file)
logger.info(f'Running with {config.get("run","name")}...')

# ==============================================================================
output_dir = config.get('run','output_dir')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
logger.info(f'Loading from cache at {output_dir}')
p21c.config['direc'] = output_dir

# --------------------------------------
lightcone_quantities = ("brightness_temp", 'density')
global_quantities    = ("brightness_temp", 'density', 'xH_box')

# ==================================
# parameters

# Fidicual parameters
user_params = dict(config.items('user_params'))
user_params = {key:p21fish.read_config_params(user_params[key]) for key in user_params}
user_params["N_THREADS"] = N_THREADS

flag_options = dict(config.items('flag_options'))
flag_options = {key:p21fish.read_config_params(flag_options[key]) for key in flag_options}

astro_params_fid = dict(config.items('astro_params'))
astro_params_fid = {key:float(astro_params_fid[key]) for key in astro_params_fid}

if fix_astro_params:
    astro_params_vary = []
else:
    astro_params_vary = config.get('vary','astro_params_vary').split('\n')
    astro_params_vary = list(filter(None, astro_params_vary))

# ==================================

min_redshift = float(config.get('redshifts','min'))
max_redshift = float(config.get('redshifts','max'))

HII_DIM = user_params["HII_DIM"]
BOX_LEN = user_params["BOX_LEN"]

logger.info(f'Making lightcone from z={min_redshift}-{max_redshift}')
logger.info(f'Box HII_DIM={HII_DIM}, BOX_LEN={BOX_LEN}')

# Clean up types
if save_Tb:
    clear_kind = ['IonizedBox','TsBox']
else:
    clear_kind = ['IonizedBox','TsBox','BrightnessTemp', 'PerturbedField']

# ==================================
# Make dictionary of sets of parameters for each run
astro_params_run_all = {}

# Set up parameters for fisher runs
if flag_options['USE_ETHOS'] is True:
    dict_prefix = f'h_PEAK_{h_PEAK:.1f}_'
else:
    dict_prefix = ''

astro_params_run_all[f'{dict_prefix}fid'] = astro_params_fid

for param in astro_params_vary:
    p_fid = astro_params_fid[param]

    # Make smaller for L_X
    if param == 'L_X':
        q = 0.001*vary_array
    else:
        q = q_scale/100*vary_array

    if p_fid == 0.:
        p = q
    else:
        p = p_fid - q*p_fid

    astro_params_run = astro_params_fid.copy()

    for i,pp in enumerate(p):
        astro_params_run[param] = pp
        if param == 'L_X': # change L_X and L_X_MINI at the same time
            astro_params_run['L_X_MINI'] = pp
        astro_params_run_all[f'{dict_prefix}{param}_{q[i]}'] = astro_params_run.copy()

# TODO nicer for not ETHOS runs
if flag_options['USE_ETHOS'] is True:
    # Vary k_peak and h_peak
    # inv_k_peak = np.array([0.01, 0.03])
    # inv_k_peak = np.array([1e-4, 0.001, 0.002, 0.003])
    # inv_k_peak = np.array([1e-8, 1e-6, 1e-4])
    # inv_k_peak = np.array([1e-8, 1e-6, 0.002, 0.003])
    # inv_k_peak = np.array([1e-5, 5e-5, 1e-4, 5e-4, 1e-3])
    inv_k_peak = np.array([1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 5e-5, 5e-4, 1e-3]) # test for convergence
#    inv_k_peak = np.array([1e-5, 5e-5, 5e-4]) # this was default for h_peak = 0?
    for h_peak in h_peaks:
        for inv_k in inv_k_peak:
            log_k_peak = np.log10(1/inv_k)
            astro_params_run = astro_params_fid.copy()
            astro_params_run['log10_k_PEAK'] = log_k_peak
            astro_params_run['h_PEAK'] = h_peak
            astro_params_run_all[f'h_PEAK_{h_peak:.1f}_inv_k_PEAK_{inv_k}'] = astro_params_run.copy()

logger.info(f'Going to make {len(astro_params_run_all)} lightcones')

if 'ALPHA_ESC_-0.03' in astro_params_run_all:
    assert astro_params_run_all['ALPHA_ESC_-0.03']['ALPHA_ESC'] != astro_params_run_all['ALPHA_ESC_0.03']['ALPHA_ESC'],\
            'Parameters havent changed between fisher runs!!!'

if 'ALPHA_STAR_MINI_-0.03' in astro_params_run_all:
    assert astro_params_run_all['ALPHA_STAR_MINI_-0.03']['ALPHA_STAR_MINI'] != astro_params_run_all['ALPHA_STAR_-0.03']['ALPHA_STAR'],\
        'ALPHA_STAR and ALPHA_STAR_MINI messed up!!!'

if args.dry_run:
    for key in astro_params_run_all:
        print(key,':')
        logger.info(f'',astro_params_run_all[key])

else:
    # ==================================
    # Initial Conditions
    logger.info(f'Making initial conditions')

    initial_conditions = p21c.initial_conditions(user_params=user_params,
                                                 random_seed=random_seed,
                                                 direc=output_dir)

    # Find ICs and perturbed fields
    PerturbedField_files = glob.glob(f'{output_dir}PerturbedField*')
    IC_files = glob.glob(f'{output_dir}InitialConditions*')

    logger.info(f'Loaded or made initial conditions')
    logger.info(f'{initial_conditions.user_params}')

    # Will not write more boxes
    # p21c.config['write'] = False

    # ==================================
    # Run each filter

    def make_lightcone(astro_params_key):
        """
        Make lightcone for a given set of astroparams
        """

        # Save output for each parameter to a new directory
        # if save_Tb:
        output_dir_lc = f'{output_dir}_{astro_params_key}'
        if not os.path.exists(output_dir_lc):
            os.makedirs(output_dir_lc)

        # put PerturbedFields in output_dir_lc
        if len(PerturbedField_files) > 0:
            for PF in PerturbedField_files:
                PF_file = PF.split('/')[-1]
                linked_file = f'{output_dir_lc}/{PF_file}'
                if not os.path.exists(linked_file):
                    # os.symlink(PF, linked_file)
                    shutil.copyfile(PF, linked_file)

        for IC in IC_files:
            IC_file = IC.split('/')[-1]
            linked_file = f'{output_dir_lc}/{IC_file}'
            if not os.path.exists(linked_file):
                # os.symlink(IC, linked_file)
                shutil.copyfile(IC, linked_file)
        direc = output_dir_lc
        # else:
        #     direc = None

        # Lightcone filename
        suffix = f'HIIDIM={HII_DIM}_BOXLEN={BOX_LEN}_fisher_{astro_params_key}'
        lightcone_filename = f'LightCone_z{min_redshift:.1f}_{suffix}_r{random_seed}.h5'
        logger.info(f'Will save lightcone to {lightcone_filename}')

        t1 = time.time()
        logger.info(f'{user_params}')

        if not os.path.exists(f'{output_dir}{lightcone_filename}'):
            
            lightcone = p21c.run_lightcone(
                                        redshift = min_redshift,
                                        max_redshift = max_redshift,
                                        lightcone_quantities=lightcone_quantities,
                                        global_quantities=global_quantities,
                                        init_box = initial_conditions,
                                        user_params  = initial_conditions.user_params,
                                        flag_options = flag_options,
                                        astro_params = astro_params_run_all[astro_params_key],
                                        random_seed = random_seed,
                                        direc=direc,
                                        write=save_Tb
                                        )

            # save in main dir
            lightcone_save = lightcone.save(fname=lightcone_filename, direc=output_dir, clobber=True)
            logger.info(f'Saved lightcone to {lightcone_save}')
        else:
            logger.info(f'{lightcone_filename} already exists, skipping...')


        # Clean up
        for kind in clear_kind:
            logger.info(f'Clearing cache')
            p21c.cache_tools.clear_cache(direc=output_dir_lc, kind=kind)
        os.system(f"rm -rf {output_dir_lc}")

        t2 = time.time()
        logger.info(f'Done with {astro_params_key}, took {(t2-t1)/3600:.2f} hours')

        return

    t1 = time.time()

    if num_cores == 1:
        print(astro_params_run_all.keys())
        for key in astro_params_run_all.keys():
            logger.info(f'Saved making lightcone for {key}')
            make_lightcone(key)
    else:
        Parallel(n_jobs=num_cores)(delayed(make_lightcone)(key) for key in astro_params_run_all.keys())

    t2 = time.time()
    logger.info(f'---- Finished making lightcones, took {(t2-t1)/3600:.2f} hours')
