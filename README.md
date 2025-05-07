# SuperHoles

**D. Michel Pino[^1][^2], Rubén Seoane Souto[^1], María José Calderón[^1], Ramón Aguado[^1], and José Carlos Abadillo-Uriel[^1], Theory of superconducting proximity effect in hole-based hybrid semiconductor-superconductor devices[^3]**

[^1]: Instituto de Ciencia de Materiales de Madrid (ICMM), Consejo Superior de Investigaciones Científicas (CSIC), Calle de Sor Juana Inés de la Cruz 3, 28049 Madrid, Spain
[^2]: dmichel.pino@csic.es
[^3]: ArXiv repository: https://arxiv.org/abs/2501.00088

### Contents of the repository

This repository contains various Python libraries that may help to reproduce all the results presented in the cited paper.

- lib_holes.py: This library contains basic functions such as the definition of effective 8KP, 6KP, 4KP, 2KP Hamiltonians in proximitized Germanium, both in 3D bulk and 2DHG models. There are also additional functions to find momentum values where zero-energy anticrossings lie.
- main_holes_figures.py: This library contains the extended code needed to reproduce all the plots shown in the cited paper.
- DOS.py: This library reproduces energy-resolved figures of the density of states of a 2DHG model, for a fixed magnetic field.
- DOS2.py: This library reproduces energy-resolved figures of the density of states of a 2DHG model, for varying magnetic fields.
- lib_artist_figures: This library contains additional functions used for plotting, preparing and saving figures.
