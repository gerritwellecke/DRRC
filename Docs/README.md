# Automated documentation

This repository is using sphinx to document "itself". 

To generate the documentation execute (from within [`/Docs/`](/Docs/)!)
```bash
   make html 
```
Then open [`/Docs/build/html/index.html`](/Docs/build/html/index.html) in a browser of your choice. 

The script [`/Docs/source/run_autodoc.py`](/Docs/source/run_autodoc.py) takes care of making the documentation. 
If you want to know more about this, I suggest the [sphinx documentation](https://www.sphinx-doc.org/en/master/).

So far I have only set this up for html documentation. 
If you want LaTeX or something else, you will have to set that up first. 

## Ensuring your code is documented 
Please adhere to **Google docstrings**. 
You can find some great examples [here](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html)
and some more explanation [here](https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html).
