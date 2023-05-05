from Bio import SeqIO
from Bio.Blast.Applications import NcbiblastnCommandline, NcbiblastpCommandline
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.sparse as sparse
import argparse
import numpy as np
import pandas as pd
import os, sys, time, itertools, gc
import skbio
from sklearn.decomposition import PCA
import mkl
from numba import njit
import multiprocessing as mp

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

#---------------#
### Functions ###
#---------------#

class tools():
#---------------------------------------------------------------------------------------------------------------------#
# File handling
#---------------------------------------------------------------------------------------------------------------------#
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

#---------------------------------------------------------------------------------------------------------------------#
# Matrix construction and Parallel CSCS computation
#---------------------------------------------------------------------------------------------------------------------#
    @njit
    def similarity_matrix(self):
        self.css_matrix = sparse.dok_matrix((len(self.feature_ids), len(self.feature_ids)), dtype=np.float32)
        # Creates sparse matrix from Blastn stdout, according to index of bucket table
        pscore, norm = 2, 0.01
        for line in self.output:
            if line.find("CLUSTERID1") > -1:
                pscore, norm = 4, 1

            line = line.split()
            if line[0] in self.feature_ids and line[1] in self.feature_ids:
                self.css_matrix[self.feature_ids[line[0]], self.feature_ids[line[1]]] = float(line[pscore])*norm
                self.css_matrix[self.feature_ids[line[1]], self.feature_ids[line[0]]] = float(line[pscore])*norm
    
    @njit
    def cscs(self, A, B, css):
        cssab = np.multiply(css, np.multiply(A, B.T))
        cssaa = np.multiply(css, np.multiply(A, A.T))
        cssbb = np.multiply(css, np.multiply(B, B.T))
        scaler = max(np.sum(cssaa), np.sum(cssbb))
        if scaler == 0:
            result = 0
        else:
            result = np.sum(cssab) / scaler
        return result

    @njit
    def jaccard_distance(A, B):
        nominator = np.setdiff1d(A, B)
        denominator = np.union1d(A, B)
        return len(nominator)/len(denominator)
    
    def __worker(self, input, output, css):
        for func, A, B, index_a, index_b in iter(input.get, None):
            result = func(A, B, css)
            output.put([index_a, index_b, result])
    
    def __Parallelize(self, func, samples, css):
        NUMBER_OF_PROCESSES = mp.cpu_count()

        cscs_u = np.zeros([samples.shape[1], samples.shape[1]])
        TASKS = [(func, samples[:,i], samples[:,j], i, j) for i,j in itertools.combinations(range(0, samples.shape[1]), 2)]

        # Create queues
        task_queue = mp.Queue()
        done_queue = mp.Queue()    

        # Submit tasks
        for task in TASKS:
            task_queue.put(task)

        # Start worker processes
        for i in range(NUMBER_OF_PROCESSES):
            mp.Process(target=self.__worker, args=(task_queue, done_queue, css)).start()

        # Get and print results
        for i in range(len(TASKS)):
            res = done_queue.get()
            cscs_u[res[0],res[1]] = res[2]
            cscs_u[res[1],res[0]] = cscs_u[res[0],res[1]]

        # Tell child processes to stop
        for i in range(NUMBER_OF_PROCESSES):
            task_queue.put(None)

        cscs_u[np.diag_indices(cscs_u.shape[0])] = 1 

        return cscs_u.astype(np.float64)

    def CSCS_metric(self, ):
        self.samples = sparse.csr_matrix(self.counts.div(self.counts.sum(axis=0), axis=1), dtype='float64')
        self.cscs_u = self.__Parallelize(self.cscs, self.samples.toarray(), self.css_matrix.toarray())

        # deallocate memory prior to optimization
        self.counts = None
        self.css_matrix = None
        self.samples = None
        gc.collect()

        # initialize optimization
        df, best_W, iter, Weight_stack = self.optimization(self, self.cscs_u, num_iters=1000, epss = np.finfo(np.float64).eps)

    def save_matrix_tsv(matrix, headers, filename):
        with open(filename + ".tsv", 'w') as outfile:
            outfile.write("\t".join(headers) + "\n")
            np.savetxt(outfile, matrix, delimiter="\t")


#---------------------------------------------------------------------------------------------------------------------#
# Eigendecomposition optimization
#---------------------------------------------------------------------------------------------------------------------#
    @njit
    def __grad_function(self, X, W):
        M = X * W
        _, eigval, eigvec = np.linalg.svd(M)

        # gradient & variance explained
        grad = X * np.dot(eigvec[:,0], np.transpose(eigvec[:,0]))
        e_sum = np.sum(eigval)
        if e_sum == 0:
            var_explained = 0
        else:
            var_explained = np.sum(eigval[:2]) / e_sum

        return grad, var_explained
    
    def __initialize_theta(X):
        sample_mean = np.mean(X)
        sample_var = np.var(X, ddof=1)
        alpha = sample_mean * (sample_mean * (1 - sample_mean) / sample_var - 1)
        if alpha < 0:
            alpha *= -1
        beta = (1 - sample_mean) * (sample_mean * (1 - sample_mean) / sample_var - 1)
        if beta < 0:
            beta *= -1

        # random weights important to increase F-stat and var_explained
        w = np.random.beta(alpha, beta, size=X.shape[0])
        W = np.triu(w, 1) + np.triu(w, 1).T 
        W.astype(np.float64)
        return W
    
    @njit
    def __add_column(self, m1, m2):
        return np.column_stack((m1, m2))

    def optimization(self, X, alpha=0.1, num_iters=100, epss = np.finfo(np.float64).eps):
        X[np.isnan(X)] = 0
        W = self.__initialize_theta(X)
        best_W, iter = np.ones((X.shape[0], X.shape[0]), dtype=np.float64), 0
        # Computes original variance
        _, s, _ = np.linalg.svd(X)
        e_sum = np.sum(s)
        best_var = np.sum(s[:2]) / e_sum
        original_var = best_var
        prev_var = best_var
        # collects weights
        Weight_stack = W[:,0]
        for i in range(num_iters):
            get_grad, current_var = self.__grad_function(X, W)
            abs_diff = np.absolute(current_var - prev_var)
            # Early stopping
            if abs_diff < epss:
                break

            if current_var > best_var:
                best_var = current_var
                best_W = W
                iter = i+1
            
            W += (alpha * get_grad)        
            W = np.clip(W, 0.0, 1.0)
            prev_var = current_var
            Weight_stack = self.__add_column(Weight_stack, W[:,0])

        return best_W, best_var, original_var, iter, Weight_stack

#---------------------------------------------------------------------------------------------------------------------#
# Visualization: PCoA, heatmaps, gradients
#---------------------------------------------------------------------------------------------------------------------#

    def pcoa_permanova(self, data, titles, filename, plabel, ncols=2):
        # Setup for figure and font size
        plt.figure(figsize=(15, 15))
        plt.subplots_adjust(hspace=0.2)
        plt.rcParams.update({'font.size': 12})

        # Defines same colors for members
        permanova_color = sns.color_palette('hls', len(titles))
        F_stats = pd.DataFrame(columns=["F-test", "P-value"])

        for n, id in enumerate(data):
            ax = plt.subplot(ncols, len(data) // ncols + (len(data) % ncols > 0), n + 1)

            # PCA decomposition
            pca = PCA(n_components=2)
            pca.fit_transform(id)
            var = pca.explained_variance_ratio_
            pcs = pca.components_
        
            # Permanova
            id[np.isnan(id)] = 0.0
            dist = id / id[0,0]
            dist = 1 - dist

            np.fill_diagonal(dist, 0.0)
            dist = skbio.DistanceMatrix(dist)
            result = skbio.stats.distance.permanova(dist, plabel, permutations=9999)
            F_stats.loc[n] = [result["test statistic"], result["p-value"]]

            # plots components and variances
            for i in range(id.shape[1]):
                ax.scatter(pcs[0][i], pcs[1][i], s=10)
                ax.annotate(f"{str(plabel[i])}", (pcs[0][i], pcs[1][i]))

            # Adds labels and R-squared
            ax.set_xlabel(f"PC1: {round(var[0]*100,2)}%")
            ax.set_ylabel(f"PC2: {round(var[1]*100,2)}%")
            ax.set_title(f"{titles[n]}")

        # plots barplot of permanova
        ax = plt.subplot(ncols, len(data) // ncols + (len(data) % ncols > 0), n + 2)
        ax.bar(titles, F_stats["F-test"], color=permanova_color, label=["$p={:.4f}$".format(pv) for pv in F_stats["P-value"]])
        ax.set_title("PERMANOVA")
        ax.set_xlabel("distance metrics")
        ax.set_ylabel("Pseudo-F test statistic")
        ax.set_xticklabels(titles, rotation = 45)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(f"../{filename}_multi_PCAs.png", format='png')
        plt.close()


    def heatmap_weights(self, data, titles, filename, vline = None, ncols=2):
        plt.figure(figsize=(20, 15))
        plt.subplots_adjust(hspace=0.2)
        plt.rcParams.update({'font.size': 12})

        for n, id in enumerate(data):
            ax = plt.subplot(ncols, len(data) // ncols + (len(data) % ncols > 0), n + 1)
            sns.heatmap(id, ax=ax)
            ax.set_title(f"{titles[n]}")
            ax.set_xlabel("Iterations")
            ax.set_ylabel("samples")
            if vline is not None:
                ax.axvline(x=vline[n], linestyle=':', color='grey')

        ax.legend()
        plt.tight_layout()
        plt.savefig(f"../{filename}_multi_heatmaps.png", format='png')
        plt.close()
        
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

# Parallel C interface optimization
os.environ["USE_INTEL_MKL"] = "1"
mkl.set_num_threads(4)

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
        spec.save_similarity_matrix()
        #spec.PCOA()
        #spec.optimization()
        #spec.PCOA()

    print(f"Elapsed time: {(time.time()-start_time)} seconds")
except ValueError:
    print("Please specify the mode, which indicates the type of data input!")
    sys.exit(1)