# Installation
The CSCSomics.py tool is tested on a Linux system in a miniconda environment with Python version 3.9. It is recommended to create a conda environment to have the correct python module versions without interference by following these steps:

* 1. Clone the repository:
```
git clone https://github.com/agusinac/CSCS-omics
cd CSCS-omics/
```
* 2. In the case conda is not installed:
```
wget https://repo.anaconda.com/miniconda/Miniconda3-py39_23.5.2-0-Linux-x86_64.sh
bash Miniconda3-py39_23.5.2-0-Linux-x86_64.sh
```
* 3. Create new environment (replace myenv with a name of your choice)
```
conda env create --name myenv -f install/python39.yml
conda install -c bioconda blast
```
Make sure to re-launch your terminal.

# CSCSomics example
In the test directory, test cases for metagenomics, proteomics and metabolomics can be found. 

The order of FASTA file and abundances after the -i argument is **not** important for metagenomics and proteomics.

* Metagenomics example:
```
python CSCSomics.py -m metagenomics -i features.fasta abundances.tsv -o [DIRECTORY] 
```

* Proteomics example:
```
python CSCSomics.py -m proteomics -i features.fasta abundances.tsv -o [DIRECTORY] 
```

The order of features and abundances **is** important for metabolomics. Make sure that the input files are in the order of features.tsv followed by abundances.tsv as shown in the example below.

* Metabolomics example:
```
python CSCSomics.py -m metabolomics -i features.tsv abundances.tsv -o [DIRECTORY] 
```
# Options
```
usage: CSCSomics.py [-h,    --help]
                    [-i,    --input INPUT_FILES]
                    [-o,    --output OUTPUT DIRECTORY]
                    [-md,   --metadata [INPUT_FILE COLUMN_ID COLUMN_GROUP]
                    [-m,    --mode MODE] 
                    [-n,    --normalise NORM] 
                    [-w,    --weight WEIGHT] 
                    [-s,    --seed SEED] 
                    [-it,   --iterations NUM_ITERS]
```

* In principel INPUT FILES require a features or abundances file, unless custom mode is selected. Then the following scenario is applied:
```
--input [DISTANCE_MATRIX] -m custom
```

* PERMANOVA and PCoA graphs requires metadata file and columns for samples ID and the grouping column.
```
--metadata [METADATA FILE] [COLUMN ID] [COLUMN_GROUP]
```

* Additional options:
- ```--normalise```, default is True, can be disabled via ``` -n False ```
- ```--weight```, default is True, can be disabled via ``` -w False ```
- ```--seed```, seed is random and can be specified via ``` -s 100 ```
- ```--iterations```, default is set at 1000 and can be adjusted via ``` -it 100 ```