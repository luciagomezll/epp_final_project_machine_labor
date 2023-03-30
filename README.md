Paper Replication: "Seeing beyond the Trees: Using Machine Learning to Estimate the Impact of Minimum Wages on Labor Market Outcomes" (Cengiz, D., Dube, A., Lindner, A. and Zentler-Munro, D., 2022)
===============================

Effective Programming Practices for Economists (EPP) - Final Project - Winter Term 2022/23 - University of Bonn 
=========

Author: Lucia Gomez Llactahuamani
=========

Abstract:
=========

This project replicates the first half of the main results from @cengiz2022seeing that applies machine learning tools to predict who is affected by the policy of minimum wage changes. The code replicating the second part of the paper, i.e., implementing an event study using prominent minimum wage increases in the U.S. between 1979 and 2019, is still ongoing. The original code of the paper is written is Stata and R, the main advantage of replicating it in Python is to unify all the codes in just one programming language that is free and open source. This replication has put emphasis in applying concepts learned in the EPP course such as best programming practices, functional programming, Pytask, Pytest and docstrings.

Content:
=============

To navigate through the folders, the workflow is decomposed as follows:

`src` folder includes all the necessary code used in the analysis. 

* `data`: It contains part of the original data files used by @cengiz2022seeing.The paper uses the 1979-2019 CPS-Basic files and a subset of it, the 1979-2019 CPS-ORG. The full dataset contains around 24 million observations and the subset contains around 13 million observations. Given the constraint in my computer RAM, I randomly extracted 5% of the sample from the dataset 1979-2019 CPS-ORG by using Stata. I end up using 659,641 observations. I document it extensively in the $\LaTeX$ file of this project.
* `data_management`: It contains the code to clean the datasets for the analysis part.
* `analysis`: It contains the code for machine learning methods to predict who is affected by the policy of minimum wage changes
* `final`: It contains the code to generate the final tables and figures.

`tests` folder tests the functions in `data_management`and in `analysis`.

`paper` folder contains the $\LaTeX$ file.

Get started:
=============

For a local machine to run this project, it needs to have a Python and LaTeX distribution. The project was tested on Windows 10 operating system. 

The project environment includes all the packages needed to run the project.

To run this project in a local machine:

After cloning the repo, open a terminal in the root directory of the project. Create and activate the environment with


    $ conda env create -f environment.yml
    $ conda activate machine_labor

To generate the output files that will be stored in `bld` folder, in the root directory of your terminal type

    $ pytask

Credits
=====

This project was created with [cookiecutter](https://github.com/audreyr/cookiecutter) and the [econ-project-templates](https://github.com/OpenSourceEconomics/econ-project-templates).