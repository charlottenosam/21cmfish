=====
Usage
=====

This is a basic walkthrough of how to do a Fisher matrix analysis with ``21cmFAST``
and ``21cmfish``.


Creating lightcones for your analysis
======================================

Start by creating a `.config` file for your runs. Have a look in
`21cmFAST_config_files/` for examples.

  - You can add any of the usual 21cmFAST AstroParams, UserParams and FlagOptions.
  - At the end of the config file list the AstroParams you want to vary
  - You will need to change the `output_dir` to the full path to where you want to save lightcones all other 21cmfish outputs.

To make the lightcones run:

.. code-block:: python

    python make_lightcones_for_fisher.py PATH_TO_YOUR_CONFIG_FILE.config --dry_run

I advise adding the `--dry_run` flag the first time you run to check the lightcones
it will make before it tries to run 21cmFAST! 21cmfish can also create lightcones
using multiprocessing and multithreading, which you can specify via (`--num_cores`
and `--N_THREADS`. By default, 21cmfish will run on n_cpus - 1.

Process lightcones for each parameter
======================================

The :func:`py21cmfish.Parameter` class loads lightcones for each parameter
and exports the 21cm global signal and power spectra for each lightcones, and
then calculates the derivatives of these quantities needed for the Fisher matrix
analysis.

Start by importing 21cmfish.

.. code-block:: python

    import py21cmfish

Parameters are loaded from the config file

.. code-block:: python

    astro_params_vary, astro_params_fid = p21fish.get_params_fid(config_file=data_dir+'21cmFAST_config_files/EoS_mini.config')

Each parameter is loaded as a separate :func:`py21cmfish.Parameter` object into a `parameters` dictionary. For example:

 .. code-block:: python

    output_dir_Park19 = data_dir+'examples/EoS_mini/'
    params_EoS = {}
    for param in astro_params_vary:
        params_EoS[param] = p21fish.Parameter(param=param, output_dir=output_dir)

If you are adding a new parameter from new lightcones, add `new=True` to generate new global signals and power spectra.


Create Fisher matrix
======================================

The Fisher matrix and its inverse are generated from the parameters dictionary:

 .. code-block:: python

    Fij_matrix_PS, Finv_PS= p21fish.make_fisher_matrix(params_EoS,
                                                        fisher_params=astro_params_vary,
                                                        hpeak=0.0, obs='PS',
                                                        k_min=0.1, k_max=1,
                                                        z_min=5.7, z_max=30.,
                                                        sigma_mod_frac=0.2,
                                                        add_sigma_poisson=True)

For more details see the examples notebook #TODO
