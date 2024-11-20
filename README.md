Divergence of u tests
=====================

A small collection of dedalus scripts to test how tau polynomial choices affect the divergence of velocity (u) in Rayleigh-Benard convection.  The "best in class" scripts are `rb_second.py` and `rb_second_2.5d.py`.

To fully populate the plots in the included jupyter notebook, run the following cases:
```
mpirun -n 4 python rb_first.py
mpirun -n 4 python rb_first_rescale.py
mpirun -n 4 python rb_second.py
mpirun -n 4 python rb_second_2.5d.py
```
(These can be run serially, or likely on up to about `32` cores productively).

These scripts will run to `t=1e2` and create all required data products for the notebook.
