Automated Testing
=================

We include automated testing with `pytest <https://docs.pytest.org/en/latest/>`_.

.. Important::

   Absolutely no PR should be accepted henceforth that does not include at least
   some basic tests of functionality!


Workflow and local testing
--------------------------

You should ensure that all tests run fine, before committing things!
It can also (sometimes!) be a good idea to write a test before writing the actual function, just to be sure that your function actually does what you originally indended.
This is commonly called *Test Driven Development* (or short TTD).

If you want to run the tests locally you can simply execute 

.. code:: bash

   pytest Tests/

from the git root.
Make sure that you install the updated :code:`requirements.txt` beforehand.

Beside the official documentation for pytest you might want to have a look at `this article <https://realpython.com/pytest-python-testing>`_.
