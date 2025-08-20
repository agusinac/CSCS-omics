[![run with conda](http://img.shields.io/badge/run%20with-conda-3EB049?labelColor=000000&logo=anaconda)](https://anaconda.org/agusinac/cscsomics)

# Installation
The CSCSomics can be easily installed via `conda`, make sure `conda-forge` and `bioconda` channels are added.
```
conda create -n cscsomics -c agusinac cscsomics
```
## Usage 
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
usage: CSCSomics    [-h,    --help]
                    [-i,    --input INPUT_FILES]
                    [-o,    --output OUTPUT DIRECTORY]
                    [-md,   --metadata [INPUT_FILE COLUMN_ID COLUMN_GROUP]
                    [-m,    --mode MODE] 
                    [-n,    --normalise NORM] 
                    [-w,    --weight WEIGHT] 
                    [-s,    --seed SEED] 
                    [-it,   --iterations NUM_ITERS]
                    [-c,    --cores NUM_CORES]
                    [-t,    --threads MKL_THREADS]
```

* In principel INPUT FILES require a features or abundances file, unless custom mode is selected. Then the following scenario is applied:
```
--input [DISTANCE_MATRIX] -m custom
```

* PERMANOVA and PCoA graphs requires metadata file and columns for samples ID and the grouping column.
```
--metadata [METADATA FILE] [COLUMN ID] [COLUMN_GROUP]
```
