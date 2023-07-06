# Installation
The CSCSomics.py tool is tested in a miniconda environment with Python version 3.9. It is recommended to create a conda environment to have the correct python module versions without interference. 

Upon cloning the repository, execute the following code to create a conda environment. Assuming you have installed miniconda in your environment.
```
conda env create --name myenv -f install/python39.yml
```
Make sure to re-launch your terminal.

# CSCSomics example
In the test directory, test cases for metagenomics, proteomics and metabolomics can be found. 

The order of FASTA file and abundances after the -i argument is not important for metagenomics and proteomics.

Metagenomics example:
```
python3 CSCSomics.py -M metagenomics -i features.fasta abundances.tsv -o [DIRECTORY] 
```

Proteomics example:
```
python3 CSCSomics.py -M proteomics -i features.fasta abundances.tsv -o [DIRECTORY] 
```

The order of features and abundances **is** important for metabolomics. Make sure that the input files are in the order of features.tsv followed by abundances.tsv as shown in the example below.

Metabolomics example:
```
python3 CSCSomics.py -M metabolomics -i features.tsv abundances.tsv -o [DIRECTORY] 
```
