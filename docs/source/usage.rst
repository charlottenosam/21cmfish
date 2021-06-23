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

To make the lightcones run:

.. code-block:: python

    python make_lightcones_for_fisher.py YOUR_CONFIG_FILE.config --h_PEAK 1.

I advise adding the `--dry_run` flag the first time you run to check the lightcones
it will make before it tries to run 21cmFAST!

Process lightcones for each parameter
======================================

The :func:`py21cmfish.Parameter` class loads lightcones for each parameter
and exports the 21cm global signal and power spectra for each lightcones, and
then calculates the derivatives of these quantities needed for the Fisher matrix
analysis.

Start by importing 21cmfish.

.. code-block:: python

    import py21cmfish
