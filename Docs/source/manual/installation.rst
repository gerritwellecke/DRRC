Installing drrc
===============

Installing the Needed Packages
------------------------------

First generate a new virtual environment. 
I suggest you do so in the git root.

.. code:: bash

   python -m venv .venv 

I will generally assume that the venv is found at :code:`DRRC/.venv/`.
If you put this elsewhere, then please ensure to change the templates for jobscripts etc.

Thereafter activate the virtual environment and install all packages shipped with this 
project 

.. code:: bash

   source .venv/bin/activate 

   python -m pip install -r requirements.txt 

Keep in mind, that this project requires :code:`python 3.10`, as we make use of the newer
syntax for type annotations.

Also see :doc:`Contributing to DRRC<./contributing>`.


Editable Installation of Development-Stage Code
-----------------------------------------------

The project is set up such that part of the code is installable. 
From the git root (i.e. the directory that contains the file :code:`pyproject.toml`) you can run 

.. code:: bash

   pip install -e .

This will install the :code:`drrc` package.

.. note:: 

   You can perform the above installation within a virtual environment.
