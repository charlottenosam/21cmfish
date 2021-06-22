import py21cmfast as p21c
import os
import numpy as np
import time
import itertools as it
from joblib import Parallel, delayed
import argparse
import configparser

print(f"21cmFAST version is {p21c.__version__}")

import logging
logger = logging.getLogger("21cmFAST")
logger.setLevel(logging.INFO)

# ==============================================================================
# python make_lightcones_for_fisher.py ../21cmFAST_config_files/ETHOS.config --h_PEAK 1. --dry_run
# python make_ETHOS_bigboxes_for_fisher.py --h_PEAK 1. --fix_astro_params

# ==============================================================================
# Import config files
# config = configparser.ConfigParser(os.environ, delimiters=':')
config = configparser.ConfigParser(delimiters=':')
config.optionxform = str

# Managing arguments with argparse (see http://docs.python.org/howto/argparse.html)
parser = argparse.ArgumentParser()
# ---- required arguments ---- :
parser.add_argument("config_file", type=str, help="Path to config file")
# ---- optional arguments ----
parser.add_argument("--h_PEAK", type=float, help="h_PEAK for ETHOS model [default = vary]")
parser.add_argument("--N_THREADS", type=int, help="Number of threads for 21cmFAST [default = 4]")
parser.add_argument("--num_cores", type=int, help="Number of cores to run on [default = 2]")
parser.add_argument("--q_scale", type=float, help="Percentage step for the parameters [default = 3%]")
# ---- flags ------
parser.add_argument("--save_Tb", action='store_true', help="Save BrightnessTemp boxes [default = False]")
parser.add_argument("--fix_astro_params", action='store_true', help="Fix astro params (only vary k_peak, h_peak) [default = False]")
parser.add_argument("--dry_run", action='store_true', help="Just print the parameters, don't run anything [default = False]")

args = parser.parse_args()
# ==============================================================================
# Run Parameters
N_THREADS = 4
if args.N_THREADS:
    N_THREADS  = args.N_THREADS
print(f'    Running on {N_THREADS} threads')

num_cores  = 2
if args.num_cores:
    num_cores  = args.num_cores
print(f'    Running on {num_cores} cores')

q_scale = 3
if args.q_scale:
    q_scale  = args.q_scale
print(f'    Calculating derivatives at {q_scale} percent from fiducial')

if args.h_PEAK:
    h_PEAK  = args.h_PEAK
    fix_h_PEAK = True
    h_peaks = [h_PEAK]
    print(f'    Running with h_peak = {h_PEAK}')
else:
    fix_h_PEAK = False
    h_PEAK = 1.
    h_peaks = np.arange(0., 1.1, 0.1)
    print(f'    Running with varied h_peak')

save_Tb = False
if args.save_Tb:
    save_Tb = True
    print(f'    Saving BrightnessTemp coeval boxes')

fix_astro_params = False
if args.fix_astro_params:
    fix_astro_params = True
    print(f'    Fixing astro params')

# ==============================================================================
# Get config
config_file = args.config_file
config.read(config_file)
print('    Running with %s...' % config.get('run','name'))

# ==============================================================================
random_seed = 12345

base_path  = os.path.abspath(os.path.dirname(__file__))
# output_dir = base_path+f'/../21cmFAST_notebooks/_cache/big_box/fisher'
output_dir = config.get('run','output_dir')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
print(f'    Loading from cache at {output_dir}')

# --------------------------------------
min_redshift = 5.
max_redshift = 35.
print(f'    Making lightcone from z={min_redshift}-{max_redshift}')

HII_DIM = 200
BOX_LEN = 400
print(f'    Box HII_DIM={HII_DIM}, BOX_LEN={BOX_LEN}')

lightcone_quantities = ("brightness_temp", 'density')
global_quantities    = ("brightness_temp", 'density', 'xH_box')

# ==================================
# parameters

# Fidicual parameters
user_params = {"HII_DIM":int(config.get('user_params','HII_DIM')),
                "BOX_LEN":int(config.get('user_params','HII_DIM')),
                "USE_FFTW_WISDOM": config.getboolean('user_params','USE_FFTW_WISDOM'),
                'USE_INTERPOLATION_TABLES': config.getboolean('user_params','USE_INTERPOLATION_TABLES'),
                "FAST_FCOLL_TABLES": config.getboolean('user_params','FAST_FCOLL_TABLES'),
                "USE_RELATIVE_VELOCITIES": config.getboolean('user_params','USE_RELATIVE_VELOCITIES'),
                "POWER_SPECTRUM":int(config.get('user_params','POWER_SPECTRUM')),
                "N_THREADS":N_THREADS}

astro_params_fid = dict(config.items('astro_params'))
astro_params_fid = {key:float(astro_params_fid[key]) for key in astro_params_fid}

# Make dictionary of sets of parameters for each run
astro_params_run_all = {}

if fix_astro_params:
    astro_params_vary = []
else:
    astro_params_vary = config.get('vary','astro_params_vary').split('\n')
    astro_params_vary = list(filter(None, astro_params_vary))

flag_options = {'INHOMO_RECO':config.getboolean('flag_options','INHOMO_RECO'),
                'USE_MASS_DEPENDENT_ZETA':config.getboolean('flag_options','USE_MASS_DEPENDENT_ZETA'),
                'USE_TS_FLUCT':config.getboolean('flag_options','USE_TS_FLUCT'),
                'USE_MINI_HALOS':config.getboolean('flag_options','USE_MINI_HALOS'),
                'FIX_VCB_AVG':config.getboolean('flag_options','FIX_VCB_AVG'),
                'USE_ETHOS':config.getboolean('flag_options','USE_ETHOS'),
                'FILTER':int(config.get('flag_options','FILTER'))}

# ==================================
# Set up parameters for fisher runs
astro_params_run_all[f'h_PEAK_{h_PEAK:.1f}_fid'] = astro_params_fid

for param in astro_params_vary:
    p_fid = astro_params_fid[param]

    # Make smaller for L_X
    if param == 'L_X':
        q = 0.001*np.array([-1,1])
    else:
        q = q_scale/100*np.array([-1,1])

    if p_fid == 0.:
        p = q
    else:
        p = p_fid - q*p_fid

    astro_params_run = astro_params_fid.copy()

    for i,pp in enumerate(p):
        astro_params_run[param] = pp
        if param == 'L_X': # change L_X and L_X_MINI at the same time
            astro_params_run['L_X_MINI'] = pp
        astro_params_run_all[f'h_PEAK_{h_PEAK:.1f}_{param}_{q[i]}'] = astro_params_run.copy()

# TODO nicer for not ETHOS runs
if flag_options['USE_ETHOS'] is True:
    # Vary k_peak and h_peak
    # inv_k_peak = np.array([0.01, 0.03])
    # inv_k_peak = np.array([1e-4, 0.001, 0.002, 0.003])
    # inv_k_peak = np.array([1e-8, 1e-6, 1e-4])
    # inv_k_peak = np.array([1e-8, 1e-6, 0.002, 0.003])
    # inv_k_peak = np.array([1e-5, 5e-5, 1e-4, 5e-4, 1e-3])
    inv_k_peak = np.array([1e-5, 5e-5, 5e-4])
    for h_peak in h_peaks:
        for inv_k in inv_k_peak:
            log_k_peak = np.log10(1/inv_k)
            astro_params_run = astro_params_fid.copy()
            astro_params_run['log10_k_PEAK'] = log_k_peak
            astro_params_run['h_PEAK'] = h_peak
            astro_params_run_all[f'h_PEAK_{h_PEAK:.1f}_inv_k_PEAK_{inv_k}'] = astro_params_run.copy()

print(f'    Going to make {len(astro_params_run_all)} lightcones')

if 'ALPHA_ESC_-0.03' in astro_params_run_all:
    assert astro_params_run_all['ALPHA_ESC_-0.03']['ALPHA_ESC'] != astro_params_run_all['ALPHA_ESC_0.03']['ALPHA_ESC'],\
            'Parameters havent changed between fisher runs!!!'

if 'ALPHA_STAR_MINI_-0.03' in astro_params_run_all:
    assert astro_params_run_all['ALPHA_STAR_MINI_-0.03']['ALPHA_STAR_MINI'] != astro_params_run_all['ALPHA_STAR_-0.03']['ALPHA_STAR'],\
        'ALPHA_STAR and ALPHA_STAR_MINI messed up!!!'

if args.dry_run:
    for key in astro_params_run_all:
        print(key,':')
        print('    ',astro_params_run_all[key])

else:
    # ==================================
    # Initial Conditions
    initial_conditions = p21c.initial_conditions(user_params=user_params,
                                                 random_seed=random_seed,
                                                 direc=output_dir)

    # ==================================
    # Run each filter

    def make_lightcone(astro_params_key):
        """
        Make lightcone for a given set of astroparams
        """

        # Lightcone filename
        suffix = f'HIIDIM={HII_DIM}_BOXLEN={BOX_LEN}_fisher_{astro_params_key}'
        lightcone_filename = f'LightCone_z{min_redshift:.1f}_{suffix}.h5'
        print('    Will save lightcone to',lightcone_filename)

        t1 = time.time()

        lightcone = p21c.run_lightcone(
                                    redshift = min_redshift,
                                    max_redshift = max_redshift,
                                    lightcone_quantities=lightcone_quantities,
                                    global_quantities=global_quantities,
                                    user_params  = user_params,
                                    flag_options = flag_options,
                                    astro_params = astro_params_run_all[astro_params_key],
                                    direc=output_dir,
                                    write=save_Tb
                                    )

        # Clean up
        if save_Tb:
            clear_kind = ['IonizedBox','TsBox']
        else:
            clear_kind = ['IonizedBox','TsBox','BrightnessTemp']

        for kind in clear_kind:
            p21c.cache_tools.clear_cache(direc=output_dir, kind=kind)

        lightcone_save = lightcone.save(fname=lightcone_filename, direc=output_dir, clobber=True)
        print('    Saved lightcone to',lightcone_save)

        t2 = time.time()
        print(f'    Done with {astro_params_key}, took {(t2-t1)/3600:.2f} hours')

        return

    t1 = time.time()

    Parallel(n_jobs=num_cores)(delayed(make_lightcone)(key) for key in astro_params_run_all.keys())


    t2 = time.time()
    print(f'---- Finished making lightcones, took {(t2-t1)/3600:.2f} hours ----')
