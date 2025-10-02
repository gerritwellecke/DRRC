Genration of training data
==========================

Creating 1D Kuramoto-Sivashinsky Data 
-------------------------------------

To generate training data run

.. code:: bash

   python3 generate_ks_data.py <PARAMETER_YML_FILENAME> [--path <PATH_TO_PARAMETER_YML>]

The script comes with a command line help, execute :code:`python3 generate_ks_data.py -h`.
Currently the default value for :code:`--path` is :code:`Data/1D_KuramotoSivashinsky/`, if the config-YAML is in that directory a path may be omitted.
If a path is given, it must be given *relative to the git root*! 
The script finds the local path to the git root by utilising :code:`GitPython`.

We assume the Kuramoto-Sivashinsky equation in the following form

.. math::

    \partial_t u(x,t) = - \frac{1}{2} \nabla\left[ u^2(x,t)\right] - \nabla^2 u(x,t) - \nu \nabla^4 u(x,t) \,.

The config-YAML configures the following attributes:

.. code:: yaml

   Date: "25.07.2027"      # only for convenience, i.e. not used in code

   System:
     Name: "KuramotoSivashinsky"
     Parameters:
       # nu: 1
       L: 60
       nx: 128
       t0: 0
       tN: 30000
       dt: 0.25
     Method: "spectral"  # spectral or py-pde

   Saving:
     # this must be absolute paths with respect to the repository root
     Name: "KS_t"
     PlotPath: "Figures/"

   Jobscript:
     Name: "Example-2023"
     Datadir: "~"  # this has to be an absolute path on the executing system
     Cores: 4

   ParamScan:  # this is optional and will generate a parameter sweep
     nu: [0.1, 1.1, 0.2]


Creating 2D Data
----------------

The creation of Data (along with an animation of the Data) for implemented models is preformed in :code:`./2D_ExcitableMedia/` by 

.. code:: bash

  python3 CreateAnimation.py PATH_TO_PARAMETER_YML

where :code:`PATH_TO_PARAMETER_YML` is a path to the Parameter :code:`.yml` files.
These are currently stored at :code:`../Data/2D_MODEL_NAME/`.
The script expects the absolute path with respect to the git root, i.e. in this specific 
example that would be :code:`Data/2D_MODEL_NAME/`.

Model parameters or parameters describing the dataset (to be saved) can be changed by modification of the parameter files.

Please report all bugs to @l.fleddermann or fix them by yourself. 
Bugs will most likely occur if parameters will be changed considerably. 

Creation of FitzHugh-Nagumo and Aliev-Panfilov data is implemented and can easily be created.
If other models are desired, :code:`./2D_ExcitableMedia/SES.py` needs to be modified.

If you need more help you can contact L. Fleddermann.
