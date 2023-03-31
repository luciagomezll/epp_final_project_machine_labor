Paper Replication: "Seeing beyond the Trees: Using Machine Learning to Estimate the Impact of Minimum Wages on Labor Market Outcomes" (Cengiz, D., Dube, A., Lindner, A. and Zentler-Munro, D., 2022)
===============================

Effective Programming Practices for Economists (EPP) - Final Project - Winter Term 2022/23 - University of Bonn 
=========

Author: Lucia Gomez Llactahuamani
=========

Abstract:
=========

This project replicates the first half of the main results from [Cengiz, D., Dube, A., Lindner, A., & Zentler-Munro, D. (2022)](https://www.journals.uchicago.edu/doi/abs/10.1086/718497).  Following the authors, I apply machine learning methods to identify the potencial workers who are actually affected by the minimum wage policy. Although the code for the second part of the paper – estimating the impact of the minimum wage on labor market outcomes – is still being developed, this project represents a significant advance. In contrast to the original code, which was written in Stata and R, I replicated the study in Python, streamlining all the code into a single programming language that is both free and open source. This replication project emphasizes the application of best programming practices and concepts learned in the EPP course, including functional programming, Pytask, Pytest, and docstrings.

Content:
=============

To navigate through the folders, the workflow is decomposed as follows:

`src` folder includes all the necessary code used in the analysis. 

* `data`: It contains part of the original data files used by [Cengiz, D., Dube, A., Lindner, A., & Zentler-Munro, D. (2022)](https://www.journals.uchicago.edu/doi/abs/10.1086/718497).The primary data used in this paper were obtained from the 1979-2019 Current Population Survey Outgoing Rotation Group (CPS-ORG), which comprises approximately 13 million observations. In order to facilitate the replication process, I extracted a random sample of 5% from the total number of observations, resulting in a sample size of 659,641. The process of obtaining the sample is documented extensively in the $\LaTeX$ file for this project.
* `data_management`: It contains the code to clean the datasets before the `analysis`.
* `analysis`: It contains the code to train the machine learning methods and to predict who is potencially affected by the minimum wage policy.
* `final`: It contains the code to generate the final tables and figures.

`tests` folder tests the functions in `data_management`and in `analysis`.

`paper` folder contains the $\LaTeX$ files.

Get started:
=============

For a local machine to run this project, it needs to have a Python and LaTeX distribution. The project was tested on Windows 10 operating system. 

The project environment includes all the packages needed to run the project.

To run this project in a local machine:

After cloning the repo, open a terminal in the root directory of the project. Create and activate the environment with

    $ conda env create -f environment.yml
    $ conda activate machine_labor

To generate the output files that will be stored in the `bld` folder, in the root directory of your terminal type

    $ pytask

To run the tests stored in the `tests` folder, in the root directory of your terminal type

    $ pytest

Warning:
=============

By replicating the paper, I obtained similar results, which led me to similar conclusions as the authors. However, there were slight differences resulting from the translation of the codes from Stata and R to Python:

* Stata uses 32-bit floats in many instances, while Python's default is 64-bit (i.e., double precision) floats. This difference is important when selecting workers who earn less than the minimum wage, as it is determined by a ratio. Although the number of observations may vary slightly when selecting workers in Python or in Stata, the results and conclusions of the paper remain unchanged. For more information on this topic, refer to [here](https://www.stata.com/support/faqs/data-management/float-data-type/).
* Most of the machine learning techniques applied in the project yielded results similar to those reported in the paper. However, the decision tree-model produced a markedly different result, which contradicts the literature that suggests it is not among the most effective learners. I used the same parameters as the paper, but the difference may be attributed to the software package used for the calculations.

Credits
=====

This project was created with [cookiecutter](https://github.com/audreyr/cookiecutter) and the [econ-project-templates](https://github.com/OpenSourceEconomics/econ-project-templates).