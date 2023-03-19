# A Confidence Machine for Sparse High-order Interaction Model


This package provides the code to reproduce the results in the supplementary rebuttal.
## Installation & Requirements

This package has the following requirements:

- [numpy](http://numpy.org)
- [scikit-learn](http://scikit-learn.org)
- [pandas](https://pandas.pydata.org)
- [python-intervals](https://pypi.org/project/python-intervals/)
- [matplotlib](https://matplotlib.org/)

We recommend installing or updating anaconda to the latest version and use Python 3 (We used Python 3.9.7).

All commands are run from the terminal.

## Reproducibility

**NOTE**: Due to the randomness of data generating process, we note that the results might be slightly different from the supplementary rebuttal. However, the overall results for interpretation will not change.

All the figure results are saved in folder "/results"


To reproduce the results using synthetic data please run the following scripts:

- Figure1 (3tc)
	```
	>> python ex1_3tc.py
	```

- Figure 2 (abc)
	```
	>> python ex1_abc.py
	```
	
- Figure 3 (friedman2)
	```
	>> python ex1_friedman2_continuous.py
	
- Figure 4 (bodyfat)
	```
	>> python ex1_bodyfat.py

	
- Figure 5 (friedman2, n=500)
	```
	>> python ex2_friedman2_continuous.py
	
- Figure 6 and 7 (Comparing CI lengths and coverages with jacknife and jacknife+)
	```
	>> python ex1_compare_stat.py

	
- Figure 8a and 8b (Comparing execution times with jacknife and jacknife+)
	```
	>> python ex1_compare_time.py
	```




