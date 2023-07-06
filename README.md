# Installation
The CSCSomics.py tool is tested in a miniconda environment with Python version 3.9. It is recommended to create a conda environment to have the correct python module versions without interference. 

Upon cloning the repository, execute the following code to create a conda environment. Assuming you have installed miniconda in your environment.
```
conda env create --name myenv -f install/python39.yml
```
Make sure to re-launch your terminal.

# CSCSomics example
In the test folder example files can be used to use the tool as follows:
```
python3 CSCSomics.py -M metabolomics -i features.tsv abundances.tsv -o [DIRECTORY] 
```
