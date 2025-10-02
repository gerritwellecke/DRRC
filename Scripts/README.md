# Scripts

This is where the code goes that uses the package code from [Src](../Src/drrc/).

**The documentation below should work, but is not as actively maintained as the online version!**

## Creating 1D Kuramoto-Sivashinsky Data 
From `Src/1D_KS/` execute [^rel-import]
```bash
python3 generate_ks_data.py <PARAMETER_YML_FILENAME> [--path <PATH_TO_PARAMETER_YML>]
```
The script comes with a help, execute `python3 generate_ks_data.py -h`.
Currently the default value for `--path` is `Data/1D_KuramotoSivashinsky/`, if the config-YAML is in that directory a path may be omitted.
If a path is given, it must be given *relative to the git root*! 
The script finds the local path to the git root by utilising `GitPython`.

We assume the Kuramoto-Sivashinsky equation in the following form
```math
    \partial_t u(x,t) = - \frac{1}{2} \nabla\left[ u^2(x,t)\right] - \nabla^2 u(x,t) - \nu \nabla^4 u(x,t) \,.
```


## Creating 2D Excitable-Media Data

The creation of Data (along with an animation of the Data) for implemented models is preformed in `./2D_ExcitableMedia/` by 
```bash
python3 CreateAnimation.py PATH_TO_PARAMETER_YML
```
where `PATH_TO_PARAMETER_YML` is an **absolute path** to the Parameter `.yml` files. 
It can currently be given in two formats, either an absolute path on the host machine or an absolute path from the git root.
The latter format would be `Data/2D_<MODEL_NAME>/<CONFIG>.yml`.

Model parameters or parameters describing the dataset (to be saved) can be changed by modification of the parameter files.

Please report all bugs to @l.fleddermann or fix them by yourself. 
Bugs will most likely occur if parameters will be changed considerably. 

Creation of FitzHugh-Nagumo and Aliev-Panfilov data is implemented and can easily be created.
If other models are desired, `Src/drrc/spatially_extended_systems.py` needs to be modified.
(If you need help you can contact me - @l.fleddermann)


---
