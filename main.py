from Bio.Blast.Applications import NcbiblastnCommandline, NcbiblastpCommandline
import matplotlib.pyplot as plt
import seaborn
import skbio
from sklearn.metrics import euclidean_distances
from argparse import ArgumentParser
import numpy as np
import pandas as pd
import os, sys, time, random

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
#           Create Bootstrap function or check documentation

#---------------#
### Functions ###
#---------------#

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
        self.table = pd.pivot_table(df, index=df.columns[0], columns=df.columns[1], values=df.columns[2], fill_value=0.0)

    def save_similarity_matrix(self):
        #os.remove(self.blastfile)
        return self.table.to_csv(os.path.join(self.outdir,self.filename + ".csv"), index=False, sep='\t')

    def PCOA(self):
        # Transforms data
        dist_dissimilarity = euclidean_distances(1-self.table)
        pcoa_res = skbio.stats.ordination.pcoa(dist_dissimilarity)
        # plotting
        plt.scatter(pcoa_res.samples['PC1'], pcoa_res.samples['PC2'], s=5)
        Total = sum(np.square(pcoa_res.eigvals))
        PC1 = round((np.square(pcoa_res.eigvals[0])/Total)*100, 2) 
        PC2 = round((np.square(pcoa_res.eigvals[1])/Total)*100, 2) 
        plt.xlabel(f"PC1 ({PC1}%)")
        plt.ylabel(f"PC2 ({PC2}%)")
        plt.title(f"PCoA of before optimization")
        plt.savefig(os.path.join(self.outdir,self.filename + ".png"), format='png')
        
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
#test.PCOA()

#test.save_similarity_matrix()
print(f"Elapsed time: {(time.time()-start_time)} seconds")