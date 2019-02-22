Stiffness modulation of redundant musculoskeletal systems
===

[![DOI](https://zenodo.org/badge/157207358.svg)](https://zenodo.org/badge/latestdoi/157207358)


`git lfs install`

`git lfs clone https://github.com/mitkof6/musculoskeletal-redundancy.git`


Description
---

This project contains the source code related to the following publication:

Dimitar Stanev and Konstantinos Moustakas, Stiffness Modulation of
Redundant Musculoskeletal Systems, Journal of Biomechanics, vol. 85,
pp. 101-107, Mar. 2019, DOI:
https://doi.org/10.1016/j.jbiomech.2019.01.017

This work presents a framework for computing the limbs' stiffness using inverse
methods that account for the musculoskeletal redundancy effects. The
musculoskeletal task, joint and muscle stiffness are regulated by the central
nervous system towards improving stability and interaction with the environment
during movement. Many pathological conditions, such as Parkinson's disease,
result in increased rigidity due to elevated muscle tone in antagonist muscle
pairs, therefore the stiffness is an important quantity that can provide
valuable information during the analysis phase. Musculoskeletal redundancy poses
significant challenges in obtaining accurate stiffness results without
introducing critical modeling assumptions. Currently, model-based estimation of
stiffness relies on some objective criterion to deal with muscle redundancy,
which, however, cannot be assumed to hold in every context. To alleviate this
source of error, our approach explores the entire space of possible solutions
that satisfy the action and the physiological muscle constraints. Using the
notion of null space, the proposed framework rigorously accounts for the effect
of muscle redundancy in the computation of the feasible stiffness
characteristics. To confirm this, comprehensive case studies on hand movement
and gait are provided, where the feasible endpoint and joint stiffness is
evaluated. Notably, this process enables the estimation of stiffness
distribution over the range of motion and aids in further investigation of
factors affecting the capacity of the system to modulate its stiffness. Such
knowledge can significantly improve modeling by providing a holistic overview of
dynamic quantities related to the human musculoskeletal system, despite its
inherent redundancy.
    

Repository Overview
---

- arm_model: simulation of simple arm model and feasible task stiffness
- feasible_joint_stiffness: calculation of the feasible joint stiffness loads,
  by accounting for musculoskeletal redundancy effects
- docker: a self contained docker setup file, which installs all dependencies
  related to the developed algorithms


Demos
---

The user can navigate into the corresponding folders and inspect the source
code. The following case studies are provided in the form of interactive Jupyter
notebooks:

- [Arm Model](arm_model/model.ipynb) presents a case study using muscle space
  projection to study the response of segmental level reflexes

<!-- - [Muscle Space Projection](arm_model/muscle_space_projection.ipynb) -->
<!--   demonstrates muscle space projection in the context of segmental level -->
<!--   (reflex) modeling -->

- [Feasible Muscle Forces](arm_model/feasible_muscle_forces.ipynb) uses
  task space projection to simulate a simple hand movement, where the feasible
  muscle forces that satisfy this task are calculated and analyzed
  
- [Feasible Task Stiffness](arm_model/feasible_task_stiffness.ipynb) calculates
  the feasible task stiffness of the simple arm model for an arbitrary movement

- [Feasible Joint Stiffness](feasible_joint_stiffness/feasible_joint_stiffness.ipynb) calculates
  the feasible joint stiffness of an OpenSim model during walking

The .html files corresponding to the .ipynb notebooks included in the folders
contain the pre-executed results of the demos.


<a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img
alt="Creative Commons License" style="border-width:0"
src="https://i.creativecommons.org/l/by/4.0/88x31.png" /></a><br />This work is
licensed under a <a rel="license"
href="http://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution
4.0 International License</a>.
