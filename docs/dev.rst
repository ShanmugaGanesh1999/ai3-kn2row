Developer Tools
================

Testing and Benchmarking
------------------------

The *run.py* script provides an easy interface for building, testing,
benchmarking and more.

    $ python run.py <command> <command> ...

Below are the possible commands and what they do.

To explicitly test a custom algorithm, add ``'custom'`` to the
algorithms to use list.

.. automodule:: run
   :members:

Installing
''''''''''

* *install*: Runs ``pip3 install .``
* *install.e*: Runs ``pip3 install --editable .``
* *install.ev*: Runs ``pip3 install --editable --verbose .``

Testing
'''''''
* *test*:
  Runs all of the following tests

Unit
""""

* *test.unit*:
  Runs all unit tests
* *test.unit.<operation>*:
  Runs unit tests for the operation

Integration
"""""""""""

#. Ensures outputs are equivalent before and after after swapping *conv2d*
   modules for the framework's implementations

   * *test.swap_conv2d*:
     Tests all models in |model_zoo|_
   * *test.swap_conv2d.<model>*:
     Tests the model from |model_zoo|_

#. Ensures outputs are equivalent before and after swapping all modules for the
   framework's implementations

   * *test.convert*:
     Tests all models in |model_zoo|_

   * *test.convert.<model>*:
     Tests the model from |model_zoo|_

Benchmarking
''''''''''''

* *bench*:
  Runs all of the following benchmarks

By Layer
""""""""

* *bench.layer*:
  Shows latency for all operations, both original and framework implementations

* *bench.layer.<layer>*:
  Shows latency for the specified operation, both original and framework implementations

By *DNN*
""""""""

#. Shows latencies before and after after swapping *conv2d*
   modules for the framework's implementations

   * *bench.swap_conv2d*:
     Shows latency for all models in |model_zoo|_
   * *bench.swap_conv2d.<model>*:
     Shows latency for the model from |model_zoo|_


#. Shows latencies before and after swapping all modules for the
   framework's implementations

   * *bench.convert*:
     Shows latency for all models in |model_zoo|_
   * *bench.convert.<model>*:
     Shows latency for the model from |model_zoo|_

Documentation
'''''''''''''
* *docs*: Generate the documentation in *HTML* format.
* *readme*: Generate the *README.rst*

Develop
'''''''
* *clangd*: Generate the *.clangd* file with the correct include paths
