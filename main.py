# BLAST related
from Bio import SeqIO
from Bio.Blast.Applications import NcbiblastnCommandline, NcbiblastpCommandline
# plotting
import matplotlib.pyplot as plt
# scientific libraries
import scipy.sparse as sparse
# command parser
import argparse
# basic libraries
import numpy as np
import pandas as pd
import os, sys, time, random

#-------------------#
### Define Parser ###
#-------------------#

parser = argparse.ArgumentParser(description="Structural similarity identifier")
parser.add_argument("-i", type=str, dest="input_files", nargs='+', help="Provide at least one fasta file and count table")
parser.add_argument("-o", action="store", dest="outdir", type=str, help="Provide name of directory for outfiles")
parser.add_argument("-M", type=str, dest="mode", action="store", help="Specify the mode: protein, genomics or spectral")

args = parser.parse_args()
infile = args.input_files
outdir = args.outdir
mode = args.mode

# TO DO:    Optimization step with weights, abundance, css against eigenvectors

#---------------#
### Functions ###
#---------------#

class tools():
    def __init__(self, infile, outdir):
        fasta_format = ["fasta", "fna", "ffn", "faa", "frn", "fa"]
        for file in range(len(infile)):
            if os.path.split(infile[file])[1].split('.')[1] == "tsv":
                counts = pd.read_csv(infile[file], index_col=0, sep="\t")
            elif os.path.split(infile[file])[1].split('.')[1] == "csv":
                counts = pd.read_csv(infile[file], index_col=0, sep=",")
            elif os.path.split(infile[file])[1].split('.')[1] in fasta_format:
                self.file = infile[file]
                
                # Pre-filtering of Fasta file
                self.feature_ids = {str(id):it for it, id in enumerate(list(counts.index))}
                pre_filter = [pair for pair in SeqIO.parse(self.file, "fasta") if pair.id in self.feature_ids]
                self.tmp_file = os.path.join(self.outdir, "tmp.fa")
                SeqIO.write(pre_filter, self.tmp_file, "fasta")
            else:
                raise IOError("File format is not accepted!")

        # Important file paths
        self.filename = '.'.join(os.path.split(self.file)[1].split('.')[:-1])
        self.outdir = os.path.join(os.path.realpath(os.path.dirname(__file__)), outdir)
        if not os.path.exists(self.outdir):
            os.mkdir(self.outdir)
            print(f"Directory path made: {self.outdir}")

    def clean(self):
        os.remove(self.tmp_file)      
        
    def similarity_matrix(self):
        self.css_matrix = sparse.dok_matrix((len(self.feature_ids), len(self.feature_ids)), dtype=np.float32)
        # Creates sparse matrix from Blastn file, according to index of bucket table
        for line in self.output.splitlines():
            line = line.split()
            if line[0] in self.feature_ids and line[1] in self.feature_ids:
                self.css_matrix[self.feature_ids[line[0]], self.feature_ids[line[1]]] = float(line[2])*0.01
                self.css_matrix[self.feature_ids[line[1]], self.feature_ids[line[0]]] = float(line[2])*0.01
    
    def save_similarity_matrix(self):
        return sparse.save_npz(os.path.join(self.outdir,self.filename + ".npz"), self.css_matrix.tocoo())

    def PCOA(self, sparse_matrix):
        # Compute principial coordinated from sparse matrix
        n = sparse_matrix.shape[0]
        centered_matrix = np.eye(n) - np.ones((n, n))/n
        X = -0.5 * centered_matrix @ sparse_matrix @ centered_matrix
        eigvals, eigvecs = sparse.linalg.eigs(X, k=n-1)
        coordinates = eigvecs @ np.diag(np.sqrt(eigvals))

        # plotting principial coordinated
        plt.scatter(coordinates[0], coordinates[1], s=5)
        Total = sum(np.square(np.real(eigvals)))
        PC1 = round((np.square(np.real(eigvals[0]))/Total)*100, 2) 
        PC2 = round((np.square(np.real(eigvals[1]))/Total)*100, 2) 
        plt.xlabel(f"PC1 ({PC1}%)")
        plt.ylabel(f"PC2 ({PC2}%)")
        plt.title(f"PCoA of before optimization") 
        return plt.show() 
        #plt.savefig(os.path.join(self.outdir,self.filename + ".png"), format='png')

    def error_eig(self, matrix):
        eigval, eigvecs = sparse.linalg.eigs(matrix)
        error = np.arccos(np.dot(eigval[0],eigval[1])/(np.linalg.norm(eigval[0])*np.linalg.norm(eigval[1])))
        self.alpha = 2/(sum(eigval[:1]))
        #k_con = eigval[1] / eigval[0]
        self.loss = sum(error*((1-(self.alpha*eigval[0]))**k_con)*eigvecs[0] + error*((1-(self.alpha*eigval[1]))**k_con)*eigvecs[1])

    def optimization(self):
        # normalization of abundance (counts)
        self.abundance = sparse.csr_matrix(self.counts.div(self.counts.sum(axis=0), axis=1))
        #eigval, eigvec = sparse.linalg.eigsh(self.css_matrix)
        
        
class genomics(tools):
    def __init__(self, infile, outdir):
        super().__init__(infile, outdir)

    def pairwise(self):
        cline = NcbiblastnCommandline(query = self.tmp_file, subject = self.tmp_file, outfmt=6, out='-', max_hsps=1)
        self.output = cline()[0].strip()

class transcriptomics(tools):
    def __init__(self, infile, outdir):
        super().__init__(infile, outdir)

    def pairwise(self):
        cline = NcbiblastpCommandline(query = self.tmp_file, subject = self.tmp_file, outfmt=6, out='-', max_hsps=1)
        self.output = cline()[0].strip()

class proteomics(tools):
    def __init__(self, infile, outdir):
        for file in range(len(infile)):
            if os.path.split(infile[file])[1].split('.')[1] == "tsv":
                self.counts = pd.read_csv(infile[file], index_col=0, sep="\t")
            elif os.path.split(infile[file])[1].split('.')[1] == "csv":
                self.counts = pd.read_csv(infile[file], index_col=0, sep=",")
            else:
                with open(infile[file]) as infile:
                    if infile.readline().find("CLUSTERID1") > -1:
                        self.file = infile[infile]
                    else:
                        raise IOError("File format is not accepted!")

        self.filename = '.'.join(os.path.split(self.file)[1].split('.')[:-1])
        self.outdir = os.path.join(os.path.realpath(os.path.dirname(__file__)), outdir)
        self.blastfile = os.path.join(self.outdir,self.file)
        if not os.path.exists(self.outdir):
            os.mkdir(self.outdir)
            print(f"Directory path made: {self.outdir}")     
 
#----------#
### MAIN ###
#----------#

try:
    start_time = time.time()
    if mode == "dna" or mode == "rna":
        DNA = genomics(infile, outdir)
        DNA.pairwise()
        DNA.similarity_matrix()
        #DNA.PCOA()
        #DNA.optimization()
        #DNA.PCOA()

    if mode == "protein":
        protein = transcriptomics(infile, outdir)
        protein.similarity_matrix()
        protein.PCOA()
        protein.optimization()
        protein.PCOA()
    
    if mode == "spectral":
        spec = proteomics(infile, outdir)
        spec.similarity_matrix()
        spec.PCOA()
        spec.optimization()
        spec.PCOA()

    print(f"Elapsed time: {(time.time()-start_time)} seconds")
except ValueError:
    print("Please specify the mode, which indicates the type of data input!")
    sys.exit(1)