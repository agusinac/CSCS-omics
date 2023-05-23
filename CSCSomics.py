from Bio import SeqIO
from Bio.Blast.Applications import NcbiblastnCommandline, NcbiblastpCommandline
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.sparse as sparse
import argparse, warnings
import numpy as np
import pandas as pd
import os, sys, time, itertools, gc
import skbio
from sklearn.decomposition import PCA
import mkl
import multiprocessing as mp
import matplotlib.patches as mpatches

# FixedFormat warning
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

#-------------------#
### Define Parser ###
#-------------------#

parser = argparse.ArgumentParser(description="Structural similarity identifier")
parser.add_argument("-i", type=str, dest="input_files", nargs='+', help="Provide at least one fasta file and count table. \
    If you have specified 'mode custom' then you will input here your custom matrix file in tsv or csv format")
parser.add_argument("-o", action="store", dest="outdir", type=str, help="Provide name of directory for outfiles")
parser.add_argument("-M", type=str, dest="mode", action="store", help="Specify the mode: 'protein', 'metagenomics', 'spectral' or 'custom'")
parser.add_argument("-plot", type=bool, dest="plot", action="store", default=False, help="Specify if plots are required by 'plot True'")
parser.add_argument("-metadata", type=str, dest="metadata", nargs='+', action="store", help="If you specify '-plot True' and want to add a permanova test. \
    Please use the command as follows: '-metadata [FILE PATH] [SAMPLE ID] [GROUPING COLUMN]'")
parser.add_argument("-norm", type=bool, dest="norm", action="store", default=False, help="Specify if normalization is required by '-norm True'")

args = parser.parse_args()
infile = args.input_files
outdir = args.outdir
metadata = args.metadata
mode = args.mode
plot = args.plot
norm = args.norm

#---------------------------#
### Generic Pool function ###
#---------------------------#

def cscs(A, B, css):
    cssab = np.multiply(css, np.multiply(A, B.T))
    cssaa = np.multiply(css, np.multiply(A, A.T))
    cssbb = np.multiply(css, np.multiply(B, B.T))
    scaler = max(np.sum(cssaa), np.sum(cssbb))
    if scaler == 0:
        result = 0
    else:
        result = np.sum(cssab) / scaler
    return result

def worker(task):
    func, A, B, index_a, index_b, css = task
    result = func(A, B, css)
    return [index_a, index_b, result]

def Parallelize(func, samples, css):
    NUMBER_OF_PROCESSES = mp.cpu_count()

    cscs_u = np.zeros([samples.shape[1], samples.shape[1]])
    TASKS = [(func, samples[:,i], samples[:,j], i, j, css) for i,j in itertools.combinations(range(0, samples.shape[1]), 2)]

    with mp.Pool(processes=NUMBER_OF_PROCESSES) as pool:
        result = pool.map(worker, TASKS)

    # Get and print results
    for res in result:
        cscs_u[res[0],res[1]] = res[2]
        cscs_u[res[1],res[0]] = res[2]

    cscs_u[np.diag_indices(cscs_u.shape[0])] = 1.0 

    return cscs_u

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
        self.sample_ids = {str(id):it for it, id in enumerate(list(self.counts.columns))}
        pre_filter = [pair for pair in SeqIO.parse(self.file, "fasta") if pair.id in self.feature_ids]
        self.tmp_file = os.path.join(self.outdir, "tmp.fa")
        SeqIO.write(pre_filter, self.tmp_file, "fasta")

        # Empty array
        self.metric = np.array([])

#---------------------------------------------------------------------------------------------------------------------#
# Matrix construction and Parallel CSCS computation
#---------------------------------------------------------------------------------------------------------------------#

    def similarity_matrix(self):
        self.css_matrix = sparse.dok_matrix((len(self.feature_ids), len(self.feature_ids)), dtype=np.float64)
        # Creates sparse matrix from Blastn stdout, according to index of otu table
        pscore, norm = 2, 0.01
        for line in self.output:
            if line.find("CLUSTERID1") > -1:
                pscore, norm = 4, 1

            line = line.split("\t")
            if line[0] in self.feature_ids and line[1] in self.feature_ids:
                self.css_matrix[self.feature_ids[line[0]], self.feature_ids[line[1]]] = float(line[pscore])*norm
                self.css_matrix[self.feature_ids[line[1]], self.feature_ids[line[0]]] = float(line[pscore])*norm
        os.remove(self.tmp_file)
        self.output, self.tmp_file = None, None
        gc.collect()

    def distance_metric(self, meta_file, Normilization, plot):
        if self.metric.size == 0:
            if Normilization == True:
                self.samples = sparse.csr_matrix(self.counts.div(self.counts.sum(axis=0), axis=1), dtype=np.float64)
            else:
                self.samples = self.counts.values

            # Generic CSCS Pipeline
            self.pairwise()
            self.similarity_matrix()
            self.metric = Parallelize(cscs, self.samples, self.css_matrix.toarray())
            # deallocate memory prior to optimization
            self.counts = None
            self.css_matrix = None
            self.samples = None
            gc.collect()

        # initialize optimization
        self.optimization()
        self.metric_w = self.best_W * self.metric
        self.save_matrix_tsv(self.metric_w, self.sample_ids)
        # Plotting if specified
        if plot == True and len(meta_file) == 3:
            groups = pd.read_csv(meta_file[0], usecols=[meta_file[1], meta_file[2]])
            labels = {int(self.sample_ids[id]) : group for id, group in zip(groups[meta_file[1]], groups[meta_file[2]]) if id in self.sample_ids}
            self.sorted_labels = [labels[key] for key in sorted(labels.keys())]
            self.pcoa_permanova([self.metric, self.metric_w], ["unweighted","weighted"], filename="PCoA_Permanova_stats")
        #self.heatmap_weights(self.Weight_stack, "weighted distance metric", filename="Weights_per_iteration", vline=iter)

    def save_matrix_tsv(self, matrix, headers):
        file_destination = os.path.join(self.outdir, "CSCS_distance.tsv")
        with open(file_destination, 'w') as outfile:
            outfile.write("\t".join(headers) + "\n")
            np.savetxt(outfile, matrix, delimiter="\t")

    def matrix_sparsity(self, matrix):
        nonzero_n = np.count_nonzero(matrix == 0)
        return nonzero_n / matrix.size

#---------------------------------------------------------------------------------------------------------------------#
# Gradient descent for distance explained optimization
#---------------------------------------------------------------------------------------------------------------------#

    def grad_function(self):
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
    
    def initialize_theta(self):
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
           
    def add_column(self, m1, m2):
        return np.column_stack((m1, m2))

    def optimization(self, alpha=0.1, num_iters=100, epss = np.finfo(np.float64).eps):
        self.metric[np.isnan(self.metric)] = 0
        # Convert dissimilarity into similarity matrix
        if np.allclose(np.diag(self.metric), 0):
            self.metric = 1 - self.metric
            np.fill_diagonal(self.metric, 1.0)

        self.initialize_theta()
        self.best_W, self.iter = np.ones((self.metric.shape[0], self.metric.shape[0]), dtype=np.float64), 0
        # Computes original variance
        _, s, _ = np.linalg.svd(self.metric)
        e_sum = np.sum(s)
        best_var = np.sum(s[:2]) / e_sum
        prev_var = best_var
        # collects weights
        self.Weight_stack = self.W[:,0]
        for i in range(num_iters):
            get_grad, current_var = self.grad_function()
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
            self.Weight_stack = self.add_column(self.Weight_stack, self.W[:,0])

#---------------------------------------------------------------------------------------------------------------------#
# Visualization: PCoA, heatmaps, gradients
#---------------------------------------------------------------------------------------------------------------------#
    def assign_random_colors(self):
        unique_variables = list(set(self.sorted_labels))
        num_colors_needed = len(unique_variables)
        color_palette = sns.color_palette("hls", num_colors_needed)
        self.color_mapping = {variable: color for variable, color in zip(unique_variables, color_palette)}
    
        self.replaced_list = []
        for item in self.sorted_labels:
            if item in self.color_mapping:
                self.replaced_list.append(self.color_mapping[item])
            else:
                self.replaced_list.append(item)
        self.replaced_list

    def pcoa_permanova(self, data, titles, filename, ncols=3):
        # Setup for figure and font size
        plt.figure(figsize=(15, 15))
        plt.subplots_adjust(hspace=0.2)
        plt.rcParams.update({'font.size': 12})

        # Defines same colors for members
        permanova_color = sns.color_palette('hls', len(set(self.sorted_labels)))
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
            result = skbio.stats.distance.permanova(dist, self.sorted_labels, permutations=9999)
            F_stats.loc[n] = [result["test statistic"], result["p-value"]]

            # plots components and variances
            self.assign_random_colors()
            for i in range(id.shape[1]):
                ax.scatter(pcs[0][i], pcs[1][i], s=10, color=self.replaced_list[i])

            # Adds labels and R-squared
            ax.set_xlabel(f"PC1: {round(var[0]*100,2)}%")
            ax.set_ylabel(f"PC2: {round(var[1]*100,2)}%")
            ax.set_title(f"{titles[n]}")

            # Creates dummy legend colors
            group_labels = list(self.color_mapping.keys())
            group_colors = [self.color_mapping[label] for label in group_labels]
            legend_elements = [mpatches.Patch(color=color) for color in group_colors]
            ax.legend(legend_elements, group_labels, facecolor='white', edgecolor='black')
        
        # plots barplot of permanova
        ax = plt.subplot(ncols, len(data) // ncols + (len(data) % ncols > 0), n + 2)
        ax.bar(titles, F_stats["F-test"], color=permanova_color, label=["$p={:.4f}$".format(pv) for pv in F_stats["P-value"]])
        ax.set_title("PERMANOVA")
        ax.set_xlabel("distance metric")
        ax.set_ylabel("Pseudo-F test statistic")
        ax.legend(facecolor='white', edgecolor='black')
        plt.savefig(os.path.join(self.outdir, f"{filename}.png"), format='png')
        plt.close()

    def heatmap_weights(self, data, titles, filename, vline, ncols=3):
        plt.figure(figsize=(20, 15))
        plt.subplots_adjust(hspace=0.2)
        plt.rcParams.update({'font.size': 12})
        
        ax = plt.subplot(ncols, len(data) // ncols + (len(data) % ncols > 0), 1)
        sns.heatmap(data, ax=ax)
        ax.set_title(f"{titles}")
        ax.set_xlabel("Iterations")
        ax.set_ylabel("samples")
        line = int(vline)
        ax.axvline(x=line, linestyle=':', color='grey')

        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.outdir, f"{filename}.png"), format='png')
        plt.close()
        
class metagenomics(tools):
    def __init__(self, infile, outdir):
        super().__init__(infile, outdir)

    def pairwise(self):
        cline = NcbiblastnCommandline(query = self.tmp_file, subject = self.tmp_file, outfmt=6, out='-', max_hsps=1)
        self.output = cline()[0].strip().split("\n")

class transcriptomics(tools):
    def __init__(self, infile, outdir):
        super().__init__(infile, outdir)

    def pairwise(self):
        cline = NcbiblastpCommandline(query = self.tmp_file, subject = self.tmp_file, outfmt=6, out='-', max_hsps=1)
        self.output = cline()[0].strip().split("\n")

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
        self.outdir = os.path.join(os.path.realpath(os.path.dirname(file)), outdir)
        self.blastfile = os.path.join(self.outdir,self.file)
        self.metric = None
        if not os.path.exists(self.outdir):
            os.mkdir(self.outdir)
            print(f"Directory path made: {self.outdir}")

class custom_matrix(tools):
    def __init__(self, infile, outdir):
        for file in range(len(infile)):
            if os.path.split(infile[file])[1].split('.')[-1] == "tsv":
                self.read_matrix(infile[file], "\t")
            elif os.path.split(infile[file])[1].split('.')[-1] == "csv":
                self.read_matrix(infile[file], ",")
        self.counts = None

        self.outdir = os.path.join(os.path.realpath(os.path.dirname(__file__)), outdir)
        if not os.path.exists(self.outdir):
            os.mkdir(self.outdir)
            print(f"Directory path made: {self.outdir}")

    def read_matrix(self, file, delimitor):
        with open(file, "r") as infile:
            headers = next(infile).strip().split(delimitor)
        self.sample_ids = {header: idx for idx, header in enumerate(headers)}
        self.metric = np.genfromtxt(file, skip_header=1, usecols=range(1, len(headers)+1), dtype=np.float64, delimiter=delimitor)          
 
#----------#
### MAIN ###
#----------#

os.environ["USE_INTEL_MKL"] = "1"
mkl.set_num_threads(4)

try:
    start_time = time.time()
    if mode == "custom":
        custom = custom_matrix(infile, outdir)
        custom.distance_metric(Normilization=False, plot=plot, meta_file=metadata)

    if mode == "metagenomics":
        DNA = metagenomics(infile, outdir)
        DNA.distance_metric(Normilization=norm, plot=plot, meta_file=metadata)

    if mode == "protein":
        protein = transcriptomics(infile, outdir)
        protein.distance_metric(Normilization=norm, plot=plot, meta_file=metadata)

    if mode == "spectral":
        spec = proteomics(infile, outdir)
        spec.distance_metric(Normilization=norm, plot=plot, meta_file=metadata)

    print(f"Elapsed time: {round((time.time()-start_time), 2)} seconds")
except ValueError as error:
    print("Please specify the mode, which indicates the type of data input!", error)
    sys.exit(1)

import psutil

# Get the current process ID
pid = os.getpid()

# Create a process object for the current process
process = psutil.Process(pid)

# Get the memory usage
memory_info = process.memory_info()
memory_usage = memory_info.rss  # in bytes

# Print the memory usage
print(f"Memory usage: {memory_usage} bytes")