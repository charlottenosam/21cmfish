=====
Usage
=====

Start by importing 21cmfish.

.. code-block:: python

    import py21cmfish


Creating lightcones for your anaylsis
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
