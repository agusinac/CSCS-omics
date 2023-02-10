# BLAST related
from Bio.Blast.Applications import NcbiblastnCommandline, NcbiblastpCommandline
# plotting
import matplotlib.pyplot as plt
# scientific libraries
import skbio
import scipy.sparse as sparse
# command parser
import glob
import argparse
# basic libraries
import numpy as np
import pandas as pd
import os, sys, time, random

#-------------------#
### Define Parser ###
#-------------------#

parser = argparse.ArgumentParser(description="Structural similarity identifier")
parser.add_argument("-i", type=str, dest="input_files", nargs='+', help="Provide input file as fasta format")
parser.add_argument("-o", action="store", dest="outdir", type=str, help="Provide input file as fasta format")

args = parser.parse_args()
infile = args.input_files
outdir = args.outdir

# TO DO:    Optimization step with weights, abundance, css against eigenvectors
#           Optimize Blast file reading, remove IDs that are not covered in abundance

#---------------#
### Functions ###
#---------------#

class tools():
    def __init__(self, infile, outdir):
        fasta_format = ["fasta", "fna", "ffn", "faa", "frn", "fa"]
        for file in range(len(infile)):
            if os.path.split(infile[file])[1].split('.')[1] in fasta_format:
                self.file = infile[file]
            elif os.path.split(infile[file])[1].split('.')[1] == "tsv":
                self.counts = pd.read_csv(infile[file], index_col=0, sep="\t")
            elif os.path.split(infile[file])[1].split('.')[1] == "csv":
                self.counts = pd.read_csv(infile[file], index_col=0, sep=",")
            else:
                raise IOError("File format is not accepted!")

        self.filename = '.'.join(os.path.split(self.file)[1].split('.')[:-1])
        self.outdir = os.path.join(os.path.realpath(os.path.dirname(__file__)), outdir)
        self.blastfile = os.path.join(self.outdir,self.filename + ".blast")
        if not os.path.exists(self.outdir):
            os.mkdir(self.outdir)
            print(f"Directory path made: {self.outdir}")
        
    def similarity_matrix(self):
        feature_ids = {str(id):it for it, id in enumerate(list(self.counts.index))}
        self.css_matrix = sparse.dok_matrix((len(feature_ids), len(feature_ids)), dtype=np.float32)

        # Creates sparse matrix from Blastn file, according to index of bucket table
        filename = open(self.blastfile, "r")
        for line in filename:
            line = line.split()
            if line[0] in feature_ids and line[1] in feature_ids:
                self.css_matrix[feature_ids[line[0]], feature_ids[line[1]]] = float(line[2])*0.01
                self.css_matrix[feature_ids[line[1]], feature_ids[line[0]]] = float(line[2])*0.01
        filename.close()
    
    def save_similarity_matrix(self):
        #os.remove(self.blastfile)
        return sparse.save_npz(os.path.join(self.outdir,self.filename + ".npz"), self.css_matrix.tocoo())

    def PCOA(self):
        # Transforms data
        dist_dissimilarity = euclidean_distances(1-self.table)
        self.pcoa_res = skbio.stats.ordination.pcoa(dist_dissimilarity)
        # plotting
        plt.scatter(self.pcoa_res.samples['PC1'], self.pcoa_res.samples['PC2'], s=5)
        Total = sum(np.square(self.pcoa_res.eigvals))
        PC1 = round((np.square(self.pcoa_res.eigvals[0])/Total)*100, 2) 
        PC2 = round((np.square(self.pcoa_res.eigvals[1])/Total)*100, 2) 
        plt.xlabel(f"PC1 ({PC1}%)")
        plt.ylabel(f"PC2 ({PC2}%)")
        plt.title(f"PCoA of before optimization")
        #plt.savefig(os.path.join(self.outdir,self.filename + ".png"), format='png')

    def optimization(self):
        # normalization of abundance (counts)
        self.abundance = self.counts.div(self.counts.sum(axis=0), axis=1)
        #eigval, eigvec = sparse.linalg.eigsh(self.css_matrix)
        
        
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
test = virome(infile, outdir)
#test.pairwise()

test.similarity_matrix()
#test.PCOA()
#test.optimization()

#test.save_similarity_matrix()
print(f"Elapsed time: {(time.time()-start_time)} seconds")