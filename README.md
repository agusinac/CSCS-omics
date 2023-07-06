# Installation
The CSCSomics.py tool is tested in a miniconda environment with Python version 3.9. It is recommended to create a conda environment to have the correct python module versions without interference. 

Example code for installation:
```
conda create --name myenv python=3.9
pip install numpy pandas scipy matplotlib biopython seaborn psutil mkl scikit-learn scikit-bio
```
Make sure to re-launch your terminal.

# CSCSomics example
In the test folder example files can be used to use the tool as follows:
```
python3 CSCSomics.py -M metabolomics -i features.tsv abundances.tsv -o [DIRECTORY] 
```
