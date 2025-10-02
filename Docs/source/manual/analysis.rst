Analysing raw cluster output
============================

The rough workflow is as follows:

1. Each cluster job produces a lot of output files which contain raw data. 
2. This raw data can then be aggregated into a :code:`DataFrame` object.
3. This object is then aggregated which reduces its size substantially. 
   We usually refer to these as *reduced data frames*.

The :code:`drrc` package contains a module :code:`drrc.analysis` which has helper functions for these steps.
Refer to the :code:`main` function in [this example](/Analysis/Visualise_ValidTime.py). 
