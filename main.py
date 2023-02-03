from Bio.Blast.Applications import NcbiblastnCommandline, NcbiblastpCommandline
#from Bio import SeqIO, Align, SeqRecord
from argparse import ArgumentParser
import numpy as np
import pandas as pd
import os, sys, time

#-------------------#
### Define Parser ###
#-------------------#

parser = ArgumentParser(description="Structural similarity identifier")
parser.add_argument("-i", action="store", dest="query_file", type=str, help="Provide input file as fasta format")
parser.add_argument("-o", action="store", dest="outdir", type=str, help="Provide input file as fasta format")

args = parser.parse_args()
infile = args.query_file
outdir = args.outdir

# TO DO:    Automatic detection of file format
#           Create gradient descent function or check documentation
#           Create eigenvalues function or check documentation
#           Create Bootstrap function or check documentation

#------------------#
### class virome ###
#------------------#

class virome():
    def __init__(self, infile, outdir):
        self.infile = infile
        self.filename = '.'.join(os.path.split(infile)[1].split('.')[:-1])
        self.outdir = os.path.join(os.path.realpath(os.path.dirname(__file__)), outdir)
        self.blastfile = os.path.join(self.outdir,self.filename + ".blast")
        if not os.path.exists(self.outdir):
            os.mkdir(self.outdir)
            print(f"Directory path made: {self.outdir}")

    def pairwise(self):
        cline = NcbiblastnCommandline(query = infile, subject=infile, outfmt=6, out=self.blastfile, max_hsps=1)
        return cline()

    def similarity_matrix(self):
        # extracting OTU labels and perc_identity scores
        matrix = np.loadtxt(self.blastfile, dtype=str, delimiter='\t', usecols=(0,1,2))
        matrix[:,2] = matrix[:,2].astype(float)*0.01
        print(matrix[:,2])
        df = pd.DataFrame(matrix, columns=['ID1' , 'ID2', 'score'])
        # pivot & index table
        pivot = pd.pivot_table(df,index='ID1',columns='ID2',values='score', aggfunc=lambda x: ' '.join(x))
        pivot_index = pivot.index.union(pivot.columns)
        self.table = pivot.reindex(index=pivot_index, columns=pivot_index, fill_value=0.0)
        self.matrix = self.table.to_numpy(na_value=0.0)
        print(self.matrix)

    def save_similarity_matrix(self):
        #os.remove(self.blastfile)
        return self.table.to_csv(os.path.join(self.outdir,self.filename + ".csv"), index=False, sep='\t')

class proteomics(virome):
    def __init__(self, infile, outdir):
        super().__init__(infile, outdir)

    def pairwise(self):
        cline = NcbiblastpCommandline(query = infile, subject=infile, outfmt=6, out=self.blastfile, max_hsps=1)
        return cline()

#----------#
### MAIN ###
#----------#

start_time = time.time()
test = proteomics(infile, outdir)
#test.pairwise()
test.similarity_matrix()
test.save_similarity_matrix()
print(f"Elapsed time: {(time.time()-start_time)} seconds")



# list of indexed names
#feature_index = np.asarray((np.unique(x, return_counts=True))).T

# size
#SIZE = len(feature_index)
#
## list of names
#feature_names = [feature_index[i][0] for i in range(SIZE)]
#
## empty score matrix:
#css_matrix = np.zeros(shape=[SIZE, SIZE], dtype=np.float32)

#for m in range(SIZE):
#    for n in range(int(feature_index[m][1])):
#        if ID[n] in feature_names:
#            css_matrix[m][feature_names.index(ID[n])] = score[feature_names.index(ID[n])]
#
#np.savetxt('OTU_scores.txt', css_matrix, fmt='%.2f')
