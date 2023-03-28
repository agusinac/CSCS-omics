from Bio import SeqIO
from Bio.Blast.Applications import NcbiblastnCommandline, NcbiblastpCommandline
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.sparse as sparse
import argparse
import numpy as np
import pandas as pd
import os, sys, time, itertools, gc
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

    def save_similarity_matrix(self):
        return sparse.save_npz(os.path.join(self.outdir,self.filename + ".npz"), self.css_matrix.tocoo())
    
    @njit
    def cscs(self, A, B, css):
        cssab = A * B.T * css
        cssaa = A * A.T * css
        cssbb = B * B.T * css
        scaler = max(np.sum(cssaa), np.sum(cssbb))
        if scaler == 0:
            result = 0
        else:
            result = np.sum(cssab) / scaler
        return result
    
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


#---------------------------------------------------------------------------------------------------------------------#
# Eigendecomposition optimization
#---------------------------------------------------------------------------------------------------------------------#
    @njit
    def __variance_explained(self, gradient):
        eigval = np.linalg.eigvals(gradient)
        e_sum = np.sum(eigval)
        var_explained = np.sum(eigval[:2]) / e_sum
        alpha = ((e_sum/eigval[0]) - 1)/((e_sum/eigval[0]) + 1)
        return var_explained, alpha, eigval

    @njit
    def __grad_function(self, X, W):
        M = X * W
        _, eigvec_w = np.linalg.eig(M)
        grad = eigvec_w * X * eigvec_w.T
        return grad
    
    def __initialize_theta(X):
        sample_mean = np.mean(X)
        sample_var = np.var(X, ddof=1)
        alpha = sample_mean * (sample_mean * (1 - sample_mean) / sample_var - 1)
        if alpha < 0:
            alpha *= -1
        beta = (1 - sample_mean) * (sample_mean * (1 - sample_mean) / sample_var - 1)
        if beta < 0:
            beta *= -1

        w = np.random.beta(alpha, beta, size=X.shape[0])
        W = np.triu(w, 1) + np.triu(w, 1).T
        W[np.diag_indices(W.shape[0])] = 1
        W.astype(np.float64)
        return W
    
    @njit
    def __add_column(self, m1, m2):
        return np.column_stack((m1, m2))

    def optimization(self, X, num_iters=100, epss = np.finfo(np.float64).eps):
        W = self.__initialize_theta(X)
        df = pd.DataFrame(columns=["iter", "variance_explained", "abs_diff", "eigval1", "eigval2"])

        best_var, best_W, iter = 0, 0, 0
        prev_var, _, _ = self.__variance_explained(X)
        Weight_stack = W[:,0]
        
        for i in range(num_iters):
            get_grad = self.__grad_function(X, W)
            
            current_var, alpha, eigval = self.__variance_explained(get_grad)
            abs_diff = np.sum(np.absolute(current_var - prev_var))

            # epss is based on the machine precision of np.float64 64
            df.loc[i] = [i, np.real(current_var), np.real(abs_diff), np.real(eigval[0]), np.real(eigval[1])]

            if abs_diff < epss:
                break

            if current_var > best_var:
                best_var = current_var
                best_W = W
                iter = i

            W = (W + alpha * get_grad)
            W = np.clip(W, 0, 1)
            prev_var = current_var
            Weight_stack = self.__add_column(Weight_stack, W[:,0])
        
        return df, best_W, iter, Weight_stack

#---------------------------------------------------------------------------------------------------------------------#
# Visualization: PCoA, heatmaps, gradients
#---------------------------------------------------------------------------------------------------------------------#

    def PCOA(self, dense_matrix):
        # Converts sparse matrix into symmetric dissimilarity
        mean = dense_matrix.mean(axis=0) 
        center = dense_matrix - mean 
        _, stds, pcs = np.linalg.svd(center/np.sqrt(dense_matrix.shape[0])) 
        # plotting
        font_size, var = 15, stds**2
        plt.rcParams.update({"font.size": 12})
        pca_color = sns.color_palette(None, dense_matrix.shape[1])
        fig, ax = plt.subplots(1)
        fig.set_size_inches(15, 10)
        for i in range(dense_matrix.shape[1]):
            ax.scatter(pcs[0][i], pcs[1][i], color=pca_color[i], s=10, label=f"{i+1}")
            ax.annotate(f"{str(i+1)}", (pcs[0][i], pcs[1][i]))
        ax.set_xlabel(f"PC1: {round(var[0]/np.sum(var)*100,2)}%")
        ax.set_ylabel(f"PC2: {round(var[1]/np.sum(var)*100,2)}%")
        ax.set_title(f"Unweighted CSCS")
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.7))
        fig.savefig(os.path.join(self.outdir,self.filename + ".png"), format='png')
        plt.clf()
    
    def gradient_plot_3D(self, M, title):
        grad_x, grad_y = np.gradient(M)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        surf = ax.plot_surface(*np.meshgrid(np.arange(M.shape[0]), np.arange(M.shape[0])), M, cmap='viridis', linewidth=0)
        # Controls the arrows
        ax.quiver(*np.meshgrid(np.arange(M.shape[0]), np.arange(M.shape[0])), np.zeros_like(M), \
            grad_x, grad_y, np.zeros_like(M),\
                # Parameters for arrows
                length=0.1, normalize=True, color='r')
        # Set the labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f"Gradient of {title}")
        #fig.colorbar(ax=ax)
        fig.savefig(f"../{title}_3D_GD.png", format='png')
        plt.clf()

    def heatmap_W(self, M, title):
        p1 = sns.heatmap(M)
        p1.set(xlabel=f"{title}", ylabel="")
        p1.set(title="Weights per iteration")
        plt.savefig(f"../heatmap_{title}.png", format="png")
        plt.clf()

        
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