from Bio import SeqIO
from Bio.Blast.Applications import NcbiblastnCommandline, NcbiblastpCommandline
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.sparse as sparse
import argparse
import numpy as np
import pandas as pd
import os, sys, time, itertools, gc, csv
import skbio
from sklearn.decomposition import PCA
import mkl
from numba import njit
import multiprocessing as mp

#-------------------#
### Define Parser ###
#-------------------#

parser = argparse.ArgumentParser(description="Structural similarity identifier")
parser.add_argument("-i", type=str, dest="input_files", nargs='+', help="Provide at least one fasta file and count table. \
    If you have specified 'mode custom' then you will input here your custom matrix file in tsv or csv format")
parser.add_argument("-o", action="store", dest="outdir", type=str, help="Provide name of directory for outfiles")
parser.add_argument("-M", type=str, dest="mode", action="store", help="Specify the mode: 'protein', 'metagenomics', 'spectral' or 'custom'")
parser.add_argument("-plot", type=str, dest="plot", action="store", default=True, help="Specify if plots are required by 'plot False'")
parser.add_argument("-norm", type=str, dest="norm", action="store", default=False, help="Specify if normalization is required by '-norm True'")

args = parser.parse_args()
infile = args.input_files
outdir = args.outdir
mode = args.mode
plot = args.plot
norm = args.norm

#---------------#
### Functions ###
#---------------#

# TO DO:    1. group samples based on column from metadata


class tools():
#---------------------------------------------------------------------------------------------------------------------#
# File handling
#---------------------------------------------------------------------------------------------------------------------#
    def __init__(self, infile, outdir):
        fasta_format = ["fasta", "fna", "ffn", "faa", "frn", "fa"]
        for file in range(len(infile)):
            if os.path.split(infile[file])[1].split('.')[-1] == "tsv":
                self.counts = pd.read_csv(infile[file], index_col=0, sep="\t")
            elif os.path.split(infile[file])[1].split('.')[-1] == "csv":
                self.counts = pd.read_csv(infile[file], index_col=0, sep=",")
            elif os.path.split(infile[file])[1].split('.')[-1] in fasta_format:
                self.file = infile[file]
            else:
                raise IOError("File format is not accepted!")
        
        # Important file paths
        self.filename = '.'.join(os.path.split(self.file)[1].split('.')[:-1])
        self.outdir = os.path.join(os.path.realpath(os.path.dirname(__file__)), outdir)
        if not os.path.exists(self.outdir):
            os.mkdir(self.outdir)
            print(f"Directory path made: {self.outdir}")

        # Pre-filtering of Fasta file
        self.feature_ids = {str(id):it for it, id in enumerate(list(self.counts.index))}
        pre_filter = [pair for pair in SeqIO.parse(self.file, "fasta") if pair.id in self.feature_ids]
        self.tmp_file = os.path.join(self.outdir, "tmp.fa")
        SeqIO.write(pre_filter, self.tmp_file, "fasta")
        self.metric = None

#---------------------------------------------------------------------------------------------------------------------#
# Matrix construction and Parallel CSCS computation
#---------------------------------------------------------------------------------------------------------------------#

    def similarity_matrix(self):
        self.css_matrix = sparse.dok_matrix((len(self.feature_ids), len(self.feature_ids)), dtype=np.float32)
        # Creates sparse matrix from Blastn stdout, according to index of bucket table
        pscore, norm = 2, 0.01
        for line in self.output:
            if line.find("CLUSTERID1") > -1:
                pscore, norm = 4, 1

            line = line.split("\t")
            if line[0] in self.feature_ids and line[1] in self.feature_ids:
                self.css_matrix[self.feature_ids[line[0]], self.feature_ids[line[1]]] = float(line[pscore])*norm
                self.css_matrix[self.feature_ids[line[1]], self.feature_ids[line[0]]] = float(line[pscore])*norm
        self.output, self.tmp_file = None, None
        gc.collect()
    
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

    def distance_metric(self, Normilization=False):
        if self.metric == None:
            if Normilization == True:
                self.samples = sparse.csr_matrix(self.counts.div(self.counts.sum(axis=0), axis=1), dtype='float64')
            else:
                self.samples = self.counts.values

            self.pairwise()
            self.similarity_matrix()
            self.metric = self.__Parallelize(self.cscs, self.samples, self.css_matrix.toarray())

            # deallocate memory prior to optimization
            self.counts = None
            self.css_matrix = None
            self.samples = None
            gc.collect()

        # initialize optimization
        self.optimization()
        self.metric_w = self.best_W * self.metric
        self.save_matrix_tsv(self.metric_w, self.feature_ids)

        #if plot == True:
        #    self.pcoa_permanova(*self.metric_w, ["weighted distance metric"], plabel = self.groups)
        #    self.heatmap_weights(*Weight_stack, ["weighted distance metric"], *iter)

    def save_matrix_tsv(self, matrix, headers):
        file_destination = os.path.join(self.outdir, "CSCS_distance.tsv")
        with open(file_destination, 'w') as outfile:
            outfile.write("\t".join(headers) + "\n")
            np.savetxt(outfile, matrix, delimiter="\t")

    def matrix_sparsity(self, matrix):
        nonzero_n = np.count_nonzero(matrix == 0)
        return nonzero_n / matrix.size

#---------------------------------------------------------------------------------------------------------------------#
# Eigendecomposition optimization
#---------------------------------------------------------------------------------------------------------------------#

    def __grad_function(self):
        M = np.multiply(self.metric, self.W)
        _, eigval, eigvec = np.linalg.svd(M)

        # gradient & variance explained
        grad = np.multiply(self.metric, np.dot(eigvec[:,0], np.transpose(eigvec[:,0])))
        e_sum = np.sum(eigval)
        if e_sum == 0:
            var_explained = 0
        else:
            var_explained = np.sum(eigval[:2]) / e_sum

        return grad, var_explained
    
    def __initialize_theta(self):
        sample_mean = np.mean(self.metric)
        sample_var = np.var(self.metric, ddof=1)
        alpha = sample_mean * (sample_mean * (1 - sample_mean) / sample_var - 1)
        if alpha < 0:
            alpha *= -1
        beta = (1 - sample_mean) * (sample_mean * (1 - sample_mean) / sample_var - 1)
        if beta < 0:
            beta *= -1

        # random weights important to increase F-stat and var_explained
        w = np.random.beta(alpha, beta, size=self.metric.shape[0])
        self.W = np.triu(w, 1) + np.triu(w, 1).T 
        self.W.astype(np.float64)
           
    def __add_column(self, m1, m2):
        return np.column_stack((m1, m2))

    def optimization(self, alpha=0.1, num_iters=100, epss = np.finfo(np.float64).eps):
        self.metric[np.isnan(self.metric)] = 0
        # Convert dissimilarity into similarity matrix
        if np.allclose(np.diag(self.metric), 0):
            self.metric = 1 - self.metric
            np.fill_diagonal(self.metric, 1.0)
        
        self.__initialize_theta()
        self.best_W, self.iter = np.ones((self.metric.shape[0], self.metric.shape[0]), dtype=np.float64), 0
        # Computes original variance
        _, s, _ = np.linalg.svd(self.metric)
        e_sum = np.sum(s)
        best_var = np.sum(s[:2]) / e_sum
        prev_var = best_var
        # collects weights
        self.Weight_stack = self.W[:,0]
        for i in range(num_iters):
            get_grad, current_var = self.__grad_function()
            abs_diff = np.absolute(current_var - prev_var)
            # Early stopping
            if abs_diff < epss:
                break

            if current_var > best_var:
                best_var = current_var
                self.best_W = self.W
                self.iter = i+1
            
            self.W += (alpha * get_grad)        
            self.W = np.clip(self.W, 0.0, 1.0)
            prev_var = current_var
            self.Weight_stack = self.__add_column(self.Weight_stack, self.W[:,0])

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
        
class metagenomics(tools):
    def __init__(self, infile, outdir):
        super().__init__(infile, outdir)

    def pairwise(self):
        cline = NcbiblastnCommandline(query = self.tmp_file, subject = self.tmp_file, outfmt=6, out='-', max_hsps=1)
        self.output = cline()[0].strip("\n")

class transcriptomics(tools):
    def __init__(self, infile, outdir):
        super().__init__(infile, outdir)

    def pairwise(self):
        cline = NcbiblastpCommandline(query = self.tmp_file, subject = self.tmp_file, outfmt=6, out='-', max_hsps=1)
        self.output = cline()[0].strip("\n")

class proteomics(tools):
    def __init__(self, infile, outdir):
        for file in range(len(infile)):
            if os.path.split(infile[file])[1].split('.')[-1] == "tsv":
                self.counts = pd.read_csv(infile[file], index_col=0, sep="\t")
            elif os.path.split(infile[file])[1].split('.')[-1] == "csv":
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
        self.metric = None
        if not os.path.exists(self.outdir):
            os.mkdir(self.outdir)
            print(f"Directory path made: {self.outdir}")

class custom_matrix(tools):
    def __init__(self, infile, outdir):
        for file in range(len(infile)):
            if os.path.split(infile[file])[1].split('.')[-1] == "tsv":
                self.__read_matrix(infile[file], "\t")
            elif os.path.split(infile[file])[1].split('.')[-1] == "csv":
                self.__read_matrix(infile[file], ",")
        self.counts = None

        self.outdir = os.path.join(os.path.realpath(os.path.dirname(__file__)), outdir)
        if not os.path.exists(self.outdir):
            os.mkdir(self.outdir)
            print(f"Directory path made: {self.outdir}")

    def __read_matrix(self, file, delimitor):
        # FIX this
        with open(file, "r") as infile:
            reader = csv.reader(infile, delimiter=delimitor)
            headers = next(reader)[1:]
            data = [row[1:] for row in reader]
        self.metric = np.array(data, dtype=np.float64)
        self.feature_ids = {header: idx for idx, header in enumerate(headers)}     
 
#----------#
### MAIN ###
#----------#

os.environ["USE_INTEL_MKL"] = "1"
mkl.set_num_threads(4)

try:
    start_time = time.time()
    if mode == "custom":
        custom = custom_matrix(infile, outdir)
        custom.distance_metric(Normilization=False)

    if mode == "metagenomics":
        DNA = metagenomics(infile, outdir)
        DNA.distance_metric(Normilization=norm)

    if mode == "protein":
        protein = transcriptomics(infile, outdir)
        protein.distance_metric(Normilization=norm)
    
    if mode == "spectral":
        spec = proteomics(infile, outdir)
        spec.distance_metric(Normilization=norm)

    print(f"Elapsed time: {round((time.time()-start_time), 2)} seconds")
except ValueError as error:
    print("Please specify the mode, which indicates the type of data input!", error)
    sys.exit(1)