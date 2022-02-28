import py21cmfast as p21c
import os, glob
import numpy as np
import time
import itertools as it
from joblib import Parallel, delayed
import argparse
import configparser
import multiprocessing

import py21cmfish as p21fish

print(f"21cmFAST version is {p21c.__version__}")

import logging
logger = logging.getLogger("21cmFAST")
logger.setLevel(logging.INFO)

# ==============================================================================
# python make_lightcones_examples.py ../21cmFAST_config_files/EoS_mini_kpeak.config --h_PEAK 0.0 --astro_param F_STAR10 --astro_param_value -1.5 --dry_run
# TODO =====
# Took ---- Finished making lightcones, took 15.86 hours ---- for ETHOS.
# Took 11 mins to make PS
#
# ==============================================================================
# Import config files
config = configparser.ConfigParser(delimiters=':')
config.optionxform = str

# Managing arguments with argparse (see http://docs.python.org/howto/argparse.html)
parser = argparse.ArgumentParser()
# ---- required arguments ---- :
parser.add_argument("config_file", type=str, help="Path to config file")
# ---- optional arguments ----
parser.add_argument("--N_THREADS", type=int, help="Number of threads for 21cmFAST [default = 1, clogs memory if you use too many]")
parser.add_argument("--num_cores", type=int, help="Number of cores to run on [default = n_cpu - 1]")
parser.add_argument("--q_scale", type=float, help="Percentage step for the parameters [default = 3%]")
parser.add_argument("--random_seed", type=int, help="Random seed [default = 12345]")
parser.add_argument("--h_PEAK", type=float, help="h_PEAK for ETHOS model, only used if USE_ETHOS = True [default = vary]")
parser.add_argument("--astro_param", type=str, help="astro param to vary [default = don't change any")
parser.add_argument("--astro_param_value", type=float, help="astro param value to vary [default = don't change any")
# ---- flags ------
parser.add_argument("--save_Tb", action='store_true', help="Save BrightnessTemp boxes [default = False]")
parser.add_argument("--clobber", action='store_true', help="make new lightcones [default = False]")
parser.add_argument("--dry_run", action='store_true', help="Just print the parameters, don't run anything [default = False]")

args = parser.parse_args()
# ==============================================================================
# Get config
config_file = args.config_file
config.read(config_file)
logger.info(f'Running with {config.get("run","name")}...')

# Fidicual parameters
user_params = dict(config.items('user_params'))
user_params = {key:p21fish.read_config_params(user_params[key]) for key in user_params}

flag_options = dict(config.items('flag_options'))
flag_options = {key:p21fish.read_config_params(flag_options[key]) for key in flag_options}

astro_params_fid = dict(config.items('astro_params'))
astro_params_fid = {key:float(astro_params_fid[key]) for key in astro_params_fid}

astro_params_vary = config.get('vary','astro_params_vary').split('\n')
astro_params_vary = list(filter(None, astro_params_vary))

# ==================================
# Run Parameters
num_cores = multiprocessing.cpu_count() - 1
if args.num_cores:
    num_cores  = args.num_cores
logger.info(f'Running on {num_cores} cores')

N_THREADS = 1
if args.N_THREADS:
    N_THREADS  = args.N_THREADS
logger.info(f'Running on {N_THREADS} threads')
user_params["N_THREADS"] = N_THREADS

if args.h_PEAK is not None:
    h_PEAK  = args.h_PEAK
    fix_h_PEAK = True
    h_peaks = [h_PEAK]
    logger.info(f'Running with h_peak = {h_PEAK}')
else:
    fix_h_PEAK = False
    h_PEAK = 1.
    h_peaks = np.arange(0., 1.1, 0.1)
    logger.info(f'Running with varied h_peak [if USE_ETHOS = True]')

if args.astro_param:
    if args.astro_param not in astro_params_fid.keys():
        print(astro_params_fid.keys())
        logger.error(f'{args.astro_param} not in astro_param list')
    elif args.astro_param_value:
        astro_params_fid[args.astro_param] = args.astro_param_value
        logger.info(f'Changed {args.astro_param} to {args.astro_param_value}')
        save_suffix = f'{args.astro_param}={args.astro_param_value}'
    else:
        logger.warning(f'{args.astro_param} value not set')
else:
    save_suffix = 'fid'

save_Tb = False
if args.save_Tb:
    save_Tb = True
    logger.info(f'Saving BrightnessTemp coeval boxes')

clobber = False
if args.clobber:
    clobber = True
    logger.info(f'Clobber = True - making new lightcones')

random_seed = 12345
if args.random_seed:
    random_seed  = args.random_seed
logger.info(f'Using random_seed = {random_seed}')

# ==============================================================================
output_dir = config.get('run','output_dir')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
logger.info(f'Loading from cache at {output_dir}')

# --------------------------------------
lightcone_quantities = ("brightness_temp", 'density')
global_quantities    = ("brightness_temp", 'density', 'xH_box')

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

astro_params_run_all[f'{dict_prefix}{save_suffix}'] = astro_params_fid

logger.info(f'Going to make {len(astro_params_run_all)} lightcones')

if args.dry_run:
    for key in astro_params_run_all:
        logger.info(f'Running {key} with {astro_params_run_all}')

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

    # ==================================
    # Run each filter

    def make_lightcone(astro_params_key):
        """
        Make lightcone for a given set of astroparams
        """

        # Save output for each parameter to a new directory
        output_dir_lc = f'{output_dir}_{astro_params_key}'
        if not os.path.exists(output_dir_lc):
            os.makedirs(output_dir_lc)

        # put PerturbedFields in output_dir_lc
        if len(PerturbedField_files) > 0:
            for PF in PerturbedField_files:
                PF_file = PF.split('/')[-1]
                linked_file = f'{output_dir_lc}/{PF_file}'
                if not os.path.exists(linked_file):
                    os.symlink(PF, linked_file)

        for IC in IC_files:
            IC_file = IC.split('/')[-1]
            linked_file = f'{output_dir_lc}/{IC_file}'
            if not os.path.exists(linked_file):
                os.symlink(IC, linked_file)

        # Lightcone filename
        suffix = f'HIIDIM={HII_DIM}_BOXLEN={BOX_LEN}_fisher_{astro_params_key}'
        lightcone_filename = f'LightCone_z{min_redshift:.1f}_{suffix}_r{random_seed}.h5'
        logger.info(f'Will save lightcone to {lightcone_filename}')

        t1 = time.time()

        if not os.path.exists(f'{output_dir}{lightcone_filename}'):
            lightcone = p21c.run_lightcone(
                                        redshift = min_redshift,
                                        max_redshift = max_redshift,
                                        lightcone_quantities=lightcone_quantities,
                                        global_quantities=global_quantities,
                                        # init_box = initial_conditions,
                                        user_params  = user_params,
                                        flag_options = flag_options,
                                        astro_params = astro_params_run_all[astro_params_key],
                                        random_seed = random_seed,
                                        direc=output_dir_lc,
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
