from Bio.Blast.Applications import NcbiblastnCommandline, NcbiblastpCommandline
#from Bio import SeqIO, Align, SeqRecord
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
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

class tools():
    def __init__(self, infile, outdir):
        self.infile = infile
        self.filename = '.'.join(os.path.split(infile)[1].split('.')[:-1])
        self.outdir = os.path.join(os.path.realpath(os.path.dirname(__file__)), outdir)
        self.blastfile = os.path.join(self.outdir,self.filename + ".blast")
        if not os.path.exists(self.outdir):
            os.mkdir(self.outdir)
            print(f"Directory path made: {self.outdir}")
        
    def similarity_matrix(self):
        # load and pivot table
        df = pd.read_csv(self.blastfile, sep='\t')
        df = df[df.columns[0:3]]
        df[df.columns[2]] = df[df.columns[2]].apply(lambda x: x*0.01)
        pivot = pd.pivot_table(df, index=df.columns[0], columns=df.columns[1], values=df.columns[2], fill_value=0.0)
        # saving object variables
        self.table = pivot
        self.matrix = self.table.to_numpy()

    def save_similarity_matrix(self):
        #os.remove(self.blastfile)
        return self.table.to_csv(os.path.join(self.outdir,self.filename + ".csv"), index=False, sep='\t')

    def PCOA(self):
        # Transforms data
        mds = MDS(random_state=0, metric=True)
        test = mds.fit_transform(self.table)

        # visualization
        plt.scatter(test[:,0], test[:,1])
        plt.xlabel('Coordinate 1')
        plt.ylabel('Coordinate 2')
        # TO DO: proper legenda with colors, do grouping of data into one label ?
        """
        for i, txt in enumerate(self.table.index):
            plt.annotate(txt, (test[:,0][i]+.3, test[:,1][i]))
        """
        return plt.show()

class virome(tools):
    def __init__(self, infile, outdir):
        super().__init__(infile, outdir)

    def pairwise(self):
        cline = NcbiblastnCommandline(query = infile, subject=infile, outfmt=6, out=self.blastfile, max_hsps=1)
        return cline()

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
test.PCOA()

#test.save_similarity_matrix()
print(f"Elapsed time: {(time.time()-start_time)} seconds")