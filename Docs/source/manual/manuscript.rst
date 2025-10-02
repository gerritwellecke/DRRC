The Manuscript
==============

The manuscript for this project is published on the arXiv and currently under revision for publication in a journal.
Updates will follow at a later time.


Generating Plots for this Project
---------------------------------

This project comes with its own matplotlib style guide. 
For in depth information see `here <https://matplotlib.org/stable/users/explain/customizing.html>`_.

To use the style, you add the following to your plotting script 

.. code:: python

   from drrc.tools.plot_config import *

This exposes a function called :code:`set_plot_style` which will set the defaults for publication-style plots. 

Further it also exposes a set of variables which can and should be used in all plots. 
Generally these are labels, names of colormaps, etc.
