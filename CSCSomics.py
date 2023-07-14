#!/usr/bin/env python3
import numpy as np
import scipy
from Bio import SeqIO
from Bio.Blast.Applications import NcbiblastnCommandline, NcbiblastpCommandline
import matplotlib.pyplot as plt
import seaborn as sns
import argparse, warnings
import pandas as pd
import os, sys, time, itertools, gc, psutil
import skbio
from sklearn.decomposition import PCA
import mkl
import multiprocessing as mp
import matplotlib.patches as mpatches

# Enables MKL environment in scipy and numpy
os.environ["USE_INTEL_MKL"] = "1"
mkl.set_num_threads(4)

# Ignores warnings
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib") # Ignores FixedFormat
np.seterr(divide='ignore') # Ignores RunTimeWarning (division by zero)

#-------------------#
### Define Parser ###
#-------------------#

parser = argparse.ArgumentParser(description="Generalized CSCS for omics")
parser.add_argument("-i", "--input", type=str, dest="input_files", nargs='+', help="Provide at least one feature file and abundance table. \
    If you have specified '-m custom' then you will input here your custom matrix file in tsv or csv format")
parser.add_argument("-o", "--output", action="store", dest="outdir", type=str, help="Provide name of directory for output")
parser.add_argument("-m", "--mode", type=str, dest="mode", action="store", help="Specify the mode: '-m proteomics', '-m metagenomics', '-m metabolomics' or '-m custom'")
parser.add_argument("-md", "--metadata", type=str, dest="metadata", nargs='+', action="store", help="If you specify '-p True' and want to add a permanova test. \
    Please use the command as follows: '-md [METADATA TABLE] [COLUMN ID] [GROUPING COLUMN]'")
parser.add_argument("-n", "--normalise", type=str, dest="norm", action="store", default="True", help="Specify if normalization is required by '-n True'")
parser.add_argument("-w", "--weighted", type=str, dest="weight", action="store", default="True", help="CSCSomics automatically uses the abundances to weight the features, use '-w False' to disable")
parser.add_argument("-s", "--seed", type=int, dest="seed", action="store", help="Adjust the weights to a specific seed if desired")
parser.add_argument("-it", "--iterations", type=int, dest="num_iters", action="store", help="Adjusts the number of iterations for the optimization algorithm to run")

args = parser.parse_args()
infile = args.input_files
outdir = args.outdir
metadata = args.metadata
mode = args.mode
norm = args.norm
weight = args.weight
seed = args.seed
num_iters = args.num_iters

if seed != None:
    np.random.seed(seed)

#----------------------------------#
### Generic Parallelism function ###
#----------------------------------#

def cscs(A, B, css):
    """
    Performs element-wise matrix multiplication to generate the CSCS similarity metric 
    between two samples, a and b, and similarity matrix of features.

    Parameters:
        - A (numpy.ndarray): 1D array with integers
        - B (numpy.ndarray): 1D array with integers
        - CSS (numpy.ndarray): similarity matrix with 1's on diagonal

    Returns:
        CSCS result
    """
    cssab = css.multiply(A.multiply(B.T))
    cssaa = css.multiply(A.multiply(A.T))
    cssbb = css.multiply(B.multiply(B.T))
    scaler = max(cssaa.sum(), cssbb.sum())
    if scaler == 0:
        result = 0
    else:
        result = cssab.sum() / scaler
    return result

def worker(task):
    """
    Calls function to distribute the task and collects the result

    Parameters:
        - task [int/numpy.ndarray]: List containing integers, numpy arrays
        
    Returns:
        - [index_a, index_b, result]: indices of a and b for the symmetric matrix, result contains the CSCS similarity.
    """
    func, A, B, index_a, index_b, css = task
    result = func(A, B, css)
    return [index_a, index_b, result]

def Parallelize(func, samples, css):
    """
    Distributes tasks to workers to assemble the CSCS matrix from sample abundances and feature similarities.

    Parameters:
        - func (str): Function to be called
        - samples (numpy.ndarray): Matrix with columns as samples and rows by feature order of CSS
        - CSS (numpy.ndarray): similarity matrix with 1's on diagonal

    Returns:
        - CSCS matrix
    """
    NUMBER_OF_PROCESSES = mp.cpu_count()

    cscs_u = np.zeros([samples.shape[1], samples.shape[1]])
    TASKS = [(func, samples[:,i], samples[:,j], i, j, css) for i,j in itertools.combinations(range(0, samples.shape[1]), 2)]

    with mp.Pool(processes=NUMBER_OF_PROCESSES) as pool:
        result = pool.map(worker, TASKS)

    # Get and print results
    for res in result:
        cscs_u[res[0],res[1]] = res[2]
        cscs_u[res[1],res[0]] = res[2]

    np.fill_diagonal(cscs_u, 1)

    return cscs_u

class tools():
    """
    Superclass content:
        - basic file handling functions for features, abundance and metadata file
        - Construction of CSS and coordinator functions for CSCS construction
        - Eigenvalue optimization algorithm
        - PERMANOVA and PCoA functions
    
    Returns:
        - CSCS metric as tsv file
        - PERMANOVA and PCoA graphs (optional)

    """
    def __init__(self, infile, outdir):
        """
        Initializes class via files and output directory

        Parameters:
            - infile [str]: List of filename strings 
            - outdir (str): Path to (non)-existing directory

        """
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

        # Stores indices of features and samples in a dictionary
        self.feature_ids = {str(id):it for it, id in enumerate(list(self.counts.index))}
        self.sample_ids = {str(id):it for it, id in enumerate(list(self.counts.columns))}
        
        # Pre-filtering of Fasta file
        pre_filter = [pair for pair in SeqIO.parse(self.file, "fasta") if pair.id in self.feature_ids]
        self.tmp_file = os.path.join(self.outdir, "tmp.fa")
        SeqIO.write(pre_filter, self.tmp_file, "fasta")

        # Empty array
        self.metric = np.array([])

#---------------------------------------------------------------------------------------------------------------------#
# Matrix construction and Parallel CSCS computation
#---------------------------------------------------------------------------------------------------------------------#

    def similarity_matrix(self):
        """
        Collects BLASTn percent identity scores into a dictionary of keys for sparse matrices.

        Parameters:
            - self.output (str): stdout of BLASTn

        Returns:
            - css_matrix (scipy.sparse.csr_matrix): Symmetric matrix with 1's on diagonal

        """
        # Creates sparse matrix from Blastn stdout, according to index of otu table
        self.css_matrix = scipy.sparse.dok_matrix((len(self.feature_ids), len(self.feature_ids)), dtype=np.float64)
        pscore, norm = 2, 0.01
        for line in self.output:
            line = line.split("\t")
            if line[0] in self.feature_ids and line[1] in self.feature_ids:
                self.css_matrix[self.feature_ids[line[0]], self.feature_ids[line[1]]] = float(line[pscore])*norm
                self.css_matrix[self.feature_ids[line[1]], self.feature_ids[line[0]]] = float(line[pscore])*norm

        # converts to csr_matrix
        self.css_matrix = scipy.sparse.csr_matrix(self.css_matrix)

        # cleans unused variables
        os.remove(self.tmp_file)
        self.output = None
        self.tmp_file = None
        gc.collect()

    def distance_metric(self, meta_file, Normalization, weight, num_iters):
        """
        Coordinates the construction of the CSCS matrix and generation of graphs based on user arguments

        Parameters:
            - meta_file [str]: List of strings
            - Normalization (str): Default set at "True"
            - num_iters (int): Adjusts number of iterations for optimization

        """
        if self.metric.size == 0:
            if weight == "True":
                if Normalization == "True":
                    self.samples = scipy.sparse.csr_matrix(self.counts.div(self.counts.sum(axis=0), axis=1), dtype=np.float64)
                else:
                    self.samples = scipy.sparse.csr_matrix(self.counts.values)
            else:
                self.samples = scipy.sparse.csr_matrix(np.where(self.counts.values > 0, 1, self.counts.values))
            # Generic CSCS Pipeline           
            self.similarity_matrix()
            self.metric = Parallelize(cscs, self.samples, self.css_matrix)
            
            # deallocate memory prior to optimization
            self.counts = None
            self.css_matrix = None
            self.samples = None
            gc.collect()

        # Generic optimization
        if num_iters is None:
            self.optimization()
        else:
            self.optimization(num_iters=num_iters)
        self.scale_weighted_matrix()
        self.save_matrix_tsv(self.metric_w, self.sample_ids)

        # Plotting if specified
        if len(meta_file) == 3:
            groups = pd.read_csv(meta_file[0], usecols=[meta_file[1], meta_file[2]])
            labels = {int(self.sample_ids[id]) : group for id, group in zip(groups[meta_file[1]], groups[meta_file[2]]) if id in self.sample_ids}
            self.sorted_labels = [labels[key] for key in sorted(labels.keys())]
            self.pcoa_permanova([self.metric, self.metric_w], ["unweighted","weighted"], filename="PCoA_Permanova_stats")
                
    def save_matrix_tsv(self, matrix, headers):
        """
        Saves a matrix with headers as a tsv file

        Parameters:
            - matrix (numpy.ndarray): 2D array
            - headers [str]: List of strings
            - filename (str): string name

        Returns:
            - Saves matrix as tsv file
        """
        file_destination = os.path.join(self.outdir, "CSCS_distance.tsv")
        with open(file_destination, 'w') as outfile:
            outfile.write("\t".join(headers) + "\n")
            np.savetxt(outfile, matrix, delimiter="\t")

#---------------------------------------------------------------------------------------------------------------------#
# Gradient descent for distance explained optimization
#---------------------------------------------------------------------------------------------------------------------#

    def grad_function(self):
        """
        Computes the gradient based on the formula:

        dlambda = U * (dX) * U.T

        Eigenvectors (U) are computed from the weighted X

        Parameters:
            - self.metric (numpy.ndarray): symmetric 2D array, integers in range of 0 and 1
            - self.W (numpy.ndarray): symmetric 2D array, integers in range of 0 and 1

        Returns:
            - grad (numpy.ndarray): 2D array
            - dist_explained (int): Percentage explained on first two eigenvalues

        """
        # gradient
        M = np.multiply(self.metric, self.W)
        u, s, _ = np.linalg.svd(M)
        grad = np.multiply(self.metric, np.multiply(u[:,:1], u[:,:1].T))

        # distance explained
        e_sum = np.sum(s)
        if e_sum == 0:
            dist_explained = 0
        else:
            dist_explained = np.sum(s[:2]) / e_sum

        return grad, dist_explained

    def scale_weighted_matrix(self):
        """
        Normalizes the weighted matrix by the diagonal. 
        Cleans the matrix from infinites and reconstructs a symmetric matrix

        Parameters:
            - self.metric_w (numpy.ndarray): 2D array

        Returns:
            - self.metric_w (numpy.ndarray): symmetric 2D array
        """
        self.metric_w = self.metric_w / self.metric_w[np.diag_indices(self.metric_w.shape[0])]
        self.metric_w = np.triu(self.metric_w, 1) + np.triu(self.metric_w, 1).T
        np.fill_diagonal(self.metric_w, 1)
        self.metric_w[np.isnan(self.metric_w)] = 0
        self.metric_w[self.metric_w == -np.inf] = 0
        self.metric_w[self.metric_w == np.inf] = 1
    
    def initialize_theta(self):
        """
        Samples weights from Beta distribution and returns as a matrix

        Parameters:
            - self.metric (numpy.ndarray): symmetric 2D array, integers in range of 0 and 1

        Returns:
            - self.W (numpy.ndarray): symmetric 2D array, integers in range of 0 and 1
        """
        # Computes alpha and beta of input matrix
        sample_mean = np.mean(self.metric)
        sample_var = np.var(self.metric, ddof=1)
        alpha = sample_mean * (sample_mean * (1 - sample_mean) / sample_var - 1)
        if alpha < 0:
            alpha *= -1
        beta = (1 - sample_mean) * (sample_mean * (1 - sample_mean) / sample_var - 1)
        if beta < 0:
            beta *= -1

        # Samples weights from Beta distribution
        w = np.random.beta(alpha, beta, size=self.metric.shape[0])
        W = np.triu(w, 1) + np.triu(w, 1).T 
        W[np.isnan(W)] = 0
        self.W = np.clip(W, 0, 1)
        self.W.astype(np.float64)

    def optimization(self, alpha=0.1, num_iters=1000, epss = np.finfo(np.float64).eps):
        """
        Performs gradient descent to find highest similarity explained based on first two eigenvalues
        Repeats 5 times and selects highest similarity explained.

        Parameters:
            - self.metric (numpy.ndarray): 2D array
            - alpha (int): step size of gradient, default is 0.1
            - num_iters (int): number of iterations, default is 1000
            - epss (float): Detection limit of numpy float 64 bit

        Returns:
            - self.metric_w (numpy.ndarray): 2D array of weighted CSCS
        
        """
        fold_results = dict()

        for j in range(5):
            self.metric[np.isnan(self.metric)] = 0
            # Converts dissimilarity into similarity matrix
            if np.allclose(np.diag(self.metric), 0):
                self.metric = np.array([1]) - self.metric
                np.fill_diagonal(self.metric, 1.0)

            self.initialize_theta()
            best_W = np.ones((self.metric.shape[0], self.metric.shape[0]), dtype=np.float64)
            
            # Computes original distance explained
            s = np.linalg.svd(self.metric, compute_uv=False)
            e_sum = np.sum(s)
            best_dist = np.sum(s[:2]) / e_sum
            original_dist = best_dist
            prev_dist = best_dist

            for i in range(num_iters):
                get_grad, current_dist = self.grad_function()
                # Early stopping
                if np.absolute(current_dist - prev_dist) < epss:
                    break

                if current_dist > best_dist:
                    best_dist = current_dist
                    best_W = self.W
                
                self.W += (alpha * get_grad)        
                self.W = np.clip(self.W, 0, 1)
                prev_dist = current_dist

            fold_results[j] = [best_W, best_dist, original_dist]

        best_fold = 0
        highest_var = fold_results[best_fold][1]
        for key, value in fold_results.items():
            if value[1] >= highest_var:
                highest_var = value[1]
                best_fold = key

        best_key = fold_results[best_fold]
        self.metric_w = np.multiply(self.metric, best_key[0])         

#---------------------------------------------------------------------------------------------------------------------#
# Visualization: PCoA, heatmaps, gradients
#---------------------------------------------------------------------------------------------------------------------#
    def assign_random_colors(self):
        """
        Generates colors based on the order of the list indices to a list of string names

        Parameters:
            - self.sorted_labels [int]: List of indices

        Returns:
            - self.replaced_list [str]: List of strings
            - self.color_mapping [color]: List of colors from the seaborn.color_palette
        """
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
        """
        Generates a figure with multiple PCoA's subplots from matrices
        Final plot is the PERMANOVA statistics.

        Parameters:
            - data [numpy.ndarray]: Input list of 2D arrays.
            - titles [str]: List of titles for each subplot
            - filename (str): Name for the returned png image file
            - self.sorted_labels [int]: List of indices
            - ncols (int): number of columns for the subplots, default is 3 

        Returns:
            - Saves figure as png image

        """
        # Setup for figure and font size
        plt.figure(figsize=(15, 15))
        plt.subplots_adjust(hspace=0.2)
        plt.rcParams.update({'font.size': 15})

        # Defines same colors for members
        permanova_color = sns.color_palette('hls', len(set(self.sorted_labels)))
        F_stats = pd.DataFrame(columns=["F-test", "P-value"])

        for n, id in enumerate(data):
            ax = plt.subplot(ncols, len(data) // ncols + (len(data) % ncols > 0), n + 1)
            id[np.isnan(id)] = 0.0
            if np.allclose(np.diag(id), 1):
                id = np.array([1]) - id
                np.fill_diagonal(id, 0)

            # PCA decomposition
            pca = PCA(n_components=2)
            pca.fit_transform(id)
            var = pca.explained_variance_ratio_
            pcs = pca.components_
        
            # Permanova
            dist = skbio.DistanceMatrix(id)
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
        ax.set_xlabel("CSCS metric")
        ax.set_ylabel("Pseudo-F test statistic")
        ax.legend(facecolor='white', edgecolor='black')
        plt.tight_layout()
        plt.savefig(os.path.join(self.outdir, f"{filename}.png"), format='png')
        plt.close()
        
class metagenomics(tools):
    """
    Subclass metagenomics inherits methods from Superclass.
    """
    def __init__(self, infile, outdir):
        """
        Class initialization is inherited from Superclass
        """
        super().__init__(infile, outdir)
        self.pairwise()

    def pairwise(self):
        """
        Calls BLASTn from command line

        Parameters:
            - self.tmp_file (str): Path to pre-filtered FASTA file

        Returns:
            - self.output (str): stdout from BLASTn
        """
        cline = NcbiblastnCommandline(query = self.tmp_file, subject = self.tmp_file, outfmt=6, out='-', max_hsps=1)
        self.output = cline()[0].strip().split("\n")

class proteomics(tools):
    """
    Subclass proteomics inherits methods from Superclass.
    """
    def __init__(self, infile, outdir):
        """
        Class initialization is inherited from Superclass
        """
        super().__init__(infile, outdir)
        self.pairwise()

    def pairwise(self):
        """
        Calls BLASTp from command line

        Parameters:
            - self.tmp_file (str): Path to pre-filtered FASTA file

        Returns:
            - self.output (str): stdout from BLASTp
        """
        cline = NcbiblastpCommandline(query = self.tmp_file, subject = self.tmp_file, outfmt=6, out='-', max_hsps=1)
        self.output = cline()[0].strip().split("\n")

class metabolomics(tools):
    """
    Subclass metabolomics inherits methods from Superclass.
    """
    def __init__(self, infile, outdir):
        """
        Initializes class via files and output directory

        Parameters:
            - infile [str]: List of filename strings 
            - outdir (str): Path to (non)-existing directory

        """
        for file in range(len(infile)):
            with open(infile[file]) as FILE:
                if FILE.readline().find("CLUSTERID1") > -1:
                    self.file = infile[file]
            if self.file != None:
                if os.path.split(infile[file])[1].split('.')[-1] == "tsv":
                    self.counts = pd.read_csv(infile[file], index_col=0, sep="\t")
                elif os.path.split(infile[file])[1].split('.')[-1] == "csv":
                    self.counts = pd.read_csv(infile[file], index_col=0, sep=",") 

        # Important file paths
        self.filename = '.'.join(os.path.split(self.file)[1].split('.')[:-1])
        self.outdir = os.path.join(os.path.realpath(os.path.dirname(__file__)), outdir)

        # Stores indices of features and samples in a dictionary
        self.feature_ids = {str(id):it for it, id in enumerate(list(self.counts.index))}
        self.sample_ids = {str(id):it for it, id in enumerate(list(self.counts.columns))}
        
        # Checks if directory exists
        if not os.path.exists(self.outdir):
            os.mkdir(self.outdir)
            print(f"Directory path made: {self.outdir}")
        
        # Empty array
        self.metric = np.array([])
    
    def similarity_matrix(self):
        """
        Collects cosine scores from MS2 spectra into a dictionary of keys for sparse matrices.

        Parameters:
            - self.file (str): File containing cosine scores

        Returns:
            - css_matrix (scipy.sparse.csr_matrix): Symmetric matrix with 1's on diagonal

        """
        self.css_matrix = scipy.sparse.dok_matrix((len(self.feature_ids), len(self.feature_ids)), dtype=np.float64)
        with open(self.file) as infile:
            pscore, norm = 4, 1
            for line in infile:
                line = line.split("\t")
                if line[0] in self.feature_ids and line[1] in self.feature_ids:
                    self.css_matrix[self.feature_ids[line[0]], self.feature_ids[line[1]]] = float(line[pscore])*norm
                    self.css_matrix[self.feature_ids[line[1]], self.feature_ids[line[0]]] = float(line[pscore])*norm

        # converts to csr_matrix
        self.css_matrix = scipy.sparse.csr_matrix(self.css_matrix)
        # cleans unused variables
        self.file = None
        gc.collect()

class custom_matrix(tools):
    """
    Subclass custom_matrix inherits methods from Superclass.
    """
    def __init__(self, infile, outdir):
        """
        Initializes class via files and output directory

        Parameters:
            - infile [str]: List of filename strings 
            - outdir (str): Path to (non)-existing directory

        """
        for file in range(len(infile)):
            if os.path.split(infile[file])[1].split('.')[-1] == "tsv":
                self.read_matrix(infile[file], "\t")
            elif os.path.split(infile[file])[1].split('.')[-1] == "csv":
                self.read_matrix(infile[file], ",")
        self.counts = None

        # Checks if directory exists
        self.outdir = os.path.join(os.path.realpath(os.path.dirname(__file__)), outdir)
        if not os.path.exists(self.outdir):
            os.mkdir(self.outdir)
            print(f"Directory path made: {self.outdir}")

    def read_matrix(self, file, delimitor):
        """
        Reads distance matrix and stores it as a 2D array

        Parameters:
            - file (str): Filename of distance matrix 
            - delimitor (str): Specified by user, e.g. "," or "/t"
            
        """
        with open(file, "r") as infile:
            headers = next(infile).strip().split(delimitor)
        self.sample_ids = {header: idx for idx, header in enumerate(headers)}
        self.metric = np.genfromtxt(file, skip_header=1, usecols=range(1, len(headers)+1), dtype=np.float64, delimiter=delimitor)          
 
#----------#
### MAIN ###
#----------#

process = psutil.Process(os.getpid())

try:
    start_time = time.time()
    if mode == "custom":
        custom = custom_matrix(infile, outdir)
        custom.distance_metric(Normalization=False, meta_file=metadata, weight=False, num_iters=num_iters)

    if mode == "metagenomics":
        DNA = metagenomics(infile, outdir)
        DNA.distance_metric(Normalization=norm, meta_file=metadata, weight=weight, num_iters=num_iters)

    if mode == "proteomics":
        protein = proteomics(infile, outdir)
        protein.distance_metric(Normalization=norm, meta_file=metadata, weight=weight, num_iters=num_iters)

    if mode == "metabolomics":
        spec = metabolomics(infile, outdir)
        spec.distance_metric(Normalization=norm, meta_file=metadata, weight=weight, num_iters=num_iters)

    # memory usage in megabytes
    memory_info = process.memory_info()
    memory_usage = memory_info.rss / 10**6

    print(f"Elapsed time: {round((time.time()-start_time))} seconds")
    print(f"Memory usage: {round(memory_usage)} Mb")
    
except ValueError as error:
    print("Please specify the mode, which indicates the type of data input!", error)
    sys.exit(1)
