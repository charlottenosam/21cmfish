# useful functions
import configparser
import os

code_path = os.path.abspath(os.path.dirname(__file__))
base_path = code_path.strip(code_path.split('/')[-1])

def read_config_params(item):
    """
    Read ints and booleans from config files
    Use for user_params and flag_options only

    Parameters
    ----------
    item : str
        config dictionary item as a string

    Return
    ------
    config dictionary item as an int, bool or str

    """
    try:
        return int(item)
    except:
        if item == 'True':
            return True
        if item == 'False':
            return False
        else:
            return item


def get_params_fid(config_file):
    """
    Load Fisher parameter names and fiducials from config file
    """
    config = configparser.ConfigParser(delimiters=':')
    config.optionxform = str
    config.read(config_file)

    astro_params_fid = dict(config.items('astro_params'))
    astro_params_fid = {key:float(astro_params_fid[key]) for key in astro_params_fid}

    astro_params_vary = config.get('vary','astro_params_vary').split('\n')
    astro_params_vary = list(filter(None, astro_params_vary))

    return astro_params_vary, astro_params_fid


astro_params_labels = {'ALPHA_ESC': r'$\alpha_\mathrm{esc}^{II}$',
                        'F_ESC10' : r'$\log_{10}f_\mathrm{esc,10}$',
                        'ALPHA_STAR' : r'$\alpha_\star^{II}$',
                        'F_STAR10' : r'$\log_{10}f_{\star,10}$',
                        'F_STAR7_MINI' : r'$\log_{10}f_\mathrm{\star,7}$',
                        'ALPHA_STAR_MINI' : r'$\alpha_\star^{III}$',
                        'F_ESC7_MINI' : r'$\log_{10}f_\mathrm{esc,7}$',
                        'L_X' : r'$\log_{10}\frac{L_X/{\dot{M}_\star}}{\mathrm{erg}\,\mathrm{s}^{-1}\, M_\odot^{-1}\,\mathrm{yr}}$',
                        'NU_X_THRESH' : r'$E_0$/eV',
                        'A_LW' : r'$A_\mathrm{LW}$',
                        'M_TURN': r'$\log_{10} (M_\mathrm{turn}/M_\odot)$',
                         't_STAR': r'$t_\star$'}
