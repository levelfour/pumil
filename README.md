# A Convex Formulation for Multiple Instance Learning from Positive and Unlabeled Data

## Environment

Python 3.4.5 with the following modules:

+ numpy
+ scipy
+ scikit-learn
+ chainer
+ gurobipy

If you do not have Gurobi license, you can substitue `cvxopt` or `openopt` for it by rewriting
```python
_SOLVER = 'gurobi'
```
to each module in `MI/PU/SKC.py`.
Please refer to their official references for further information.

## Preparation

In order to prepare datasets, run

```sh
./prepare.sh
```

This script will download datasets from [here](http://www.cs.columbia.edu/~andrews/mil/datasets.html) and make necessary changes.

## Experiment

You can make experiments by `puskc.py` like

```sh
python puskc.py --dataset [musk1|musk2|elephant|fox|tiger] --prior [true class prior]
```
