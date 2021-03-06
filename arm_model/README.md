Description
---

This folder contains python scripts and Jupyter notebooks used for modeling and
simulation of a simplified arm model, that is both kinematicaly and dynamically
redundant.

Dependencies
---

The scripts are compatible with python 2. Dependencies can be installed through
python package manager (pip). The following libraries were used in the project:

- scipy (`pip install scipy`)
- pydy (`pip install pydy`)
- numpy (`pip install numpy`)
- sympy (`pip install sympy`)
- pandas (`pip install pandas`)
- seaborn (`pip install seaborn`)
- matplotlib (`pip install matplotlib`)
- pycddlib (`pip install pycddlib`)
- cython
- tqdm (`pip install tqdm`)

For the feasible muscle space analysis we use the command **convert** from
imagemagic to collect the results and generate the .gif file.


Demos
---

The user can execute the *main.py* script and choose between the different case
studies or use the interactive Jupyter notebooks (.ipynb) provided in the
folder. Folder results contains the results that are generated by the scripts
and the .html files are read only file based on the Jupyter notebooks.

- [Arm Model](model.ipynb) presents a case study using muscle space
  projection to study the response of segmental level reflexes

<!-- - [Muscle Space Projection](muscle_space_projection.ipynb) -->
<!--   demonstrates muscle space projection in the context of segmental level -->
<!--   (reflex) modeling -->

- [Feasible Muscle Forces](feasible_muscle_forces.ipynb) uses
  task space projection to simulate a simple hand movement, where the feasible
  muscle forces that satisfy this task are calculated and analyzed

- [Feasible Task Stiffness](feasible_task_stiffness.ipynb) calculates the
  feasible task stiffness of the simple arm model for an arbitrary movement
