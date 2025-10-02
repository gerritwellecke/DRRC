[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# DRRC
*DISCLAIMER*: This repo is still being finalised as the corresponding manuscript is also still in peer review.

This code belongs to work on dimensionality reduction in reservoir computing, thus short DRRC.

Note that the project is currently not actively maintained.

## Installation
The project is set up as an installable package. 
We recommend setting up a virtual environment for python using
```bash
python -m venv .venv 
source .venv/bin/activate
```
From within the project's root directory run
```bash
python -m pip install -r requirements.txt
python -m pip install .
```
Now all code should work as expected through `import drrc`.
Also see e.g. (analysis) scripts for examples.

For contributing we suggest an editable install
```bash
python -m pip install -e .
```

## Documentation
We follow the code-as-documentation philosophy, using sphinx to create the documentation. 
These docs are also hosted via GitHub pages.

### Structure of the project 

[Src](/Src/drrc/) contains the project's reusable code. 

[Scripts](/Scripts/) is all the code used for generating training data, evaluate reservoir performance, and submissions to the clusters that were available during the course of the project.
Except for the cluster submission workflow this should be reusable code.

[Docs](/Docs/) all documentation, automated and handwritten. 

[Data](/Data/) placeholder for all output data.

[Analysis](/Analysis/) scripts for data analysis.
This includes all visualisation / plotting.

----

For all further documentation / explanation, refer to the above--mentioned documentation or contact the authors of the corresponding manuscript.
