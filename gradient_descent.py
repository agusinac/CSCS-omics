import numpy as np
import scipy
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import itertools, os, mkl, pickle, gc
import skbio
import seaborn as sns
import multiprocessing as mp
import igraph as ig
from numba import njit
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

#---------------------------------------------------------------------------------------------------------------------#
# Functions under development
#---------------------------------------------------------------------------------------------------------------------#

def contours(M, title):
    x, y = np.gradient(M)
    fig, ax = plt.subplots()
    ax.contour(M, cmap="viridis")
    ax.quiver(x, y)
    ax.set_title(f"Contour plot of {title}")
    #fig.colorbar(ax=ax)
    fig.savefig(f"../{title}_contours.png", format='png')
    plt.clf()

def gradient_plot_2D(M, title):
    x, y = np.gradient(M)
    fig, ax = plt.subplots()
    im = ax.imshow(M, cmap="viridis")
    ax.quiver(x, y)
    ax.set_title(f"Gradient of {title}")
    #fig.colorbar(ax=ax)
    fig.savefig(f"../{title}_2D_GD.png", format='png')
    plt.clf()

def gradient_plot_3D(M, title):
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

def multi_heatmaps(data, titles, filename, vline = None, ncols=2):
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
            ax.axvline(x=vline[n], linestyle=':', color='black')
        if n == 4:
            ax.axvline(x=vline[n], linestyle=':', color='white')

    ax.legend()
    plt.tight_layout()
    plt.savefig(f"../{filename}_multi_heatmaps.png", format='png')
    plt.close()

def multi_stats(data, titles, filename, plabel, ncols=3):
    # Setup for figure and font size
    plt.figure(figsize=(15, 15))
    plt.subplots_adjust(hspace=0.2)
    plt.rcParams.update({'font.size': 12})

    # Defines same colors for members
    #members = igraph_label(data[0], label=betas)
    pca_color = sns.color_palette('hls', len(set(plabel)))
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
            #label_idx = int(plabel[i])
            ax.scatter(pcs[0][i], pcs[1][i], s=10)#, label=plabel[i], color=pca_color[label_idx])
            ax.annotate(f"{str(plabel[i])}", (pcs[0][i], pcs[1][i]))

        # Adds labels and R-squared
        ax.set_xlabel(f"PC1: {round(var[0]*100,2)}%")
        ax.set_ylabel(f"PC2: {round(var[1]*100,2)}%")
        
        # computes R-squared
        ax.set_title(f"{titles[n]}")# R-squared = {round(R2, 3)}")

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

def igraph_label(matrix):
    ## graph of matrix
    #edges_samples = [(i, j) for i in range(matrix.shape[0]) for j in range(matrix.shape[1])]
    g = ig.Graph.Adjacency(matrix, mode="undirected")
    # communities grouped by dominant eigenvectors
    communities = g.community_multilevel()
    
    # plotting igraph
    pal = ig.drawing.colors.ClusterColoringPalette(len(communities.membership))
    g.vs["color"] = pal.get_many(communities.membership)
    ig.plot(g, target="../communities_cscs.png")
    plt.close()
    return communities.membership

def data_dump(data, title):
    file = open(f"../{title}", "wb")
    pickle.dump(data, file)
    file.close()

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

def worker(input, output, css):
    for func, A, B, index_a, index_b in iter(input.get, None):
        result = func(A, B, css)
        output.put([index_a, index_b, result])

def Parallelize(func, samples, css):
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
        mp.Process(target=worker, args=(task_queue, done_queue, css)).start()

    # Get and print results
    for i in range(len(TASKS)):
        res = done_queue.get()
        cscs_u[res[0],res[1]] = res[2]
        cscs_u[res[1],res[0]] = cscs_u[res[0],res[1]]

    # Tell child processes to stop
    for i in range(NUMBER_OF_PROCESSES):
        task_queue.put(None)

    cscs_u[np.diag_indices(cscs_u.shape[0])] = 1.0 

    return cscs_u

def do_bootstraps(data: np.array, n_bootstraps: int=100):
    # input ->      data, bootstraps
    # outputs ->    dictionary bootsample & matrix 
    Dict = {}
    n_unique_val = 0
    sample_size = data.shape[0]
    idx = [i for i in range(sample_size)]
    
    for b in range(n_bootstraps):
        # Bootstrap with replacement
        sample_idx = np.random.choice(idx, replace=True, size=sample_size)
        boot_sample = data[:,sample_idx]

        # Number of unique values
        n_unique_val += len(set(sample_idx))

        # store results
        Dict["boot_"+str(b)] = {"boot" : np.absolute(boot_sample)}
    
    return Dict

def jaccard_distance(A, B):
    #Find symmetric difference of two sets
    nominator = np.setdiff1d(A, B)

    #Find union of two sets
    denominator = np.union1d(A, B)

    #Take the ratio of sizes
    distance = len(nominator)/len(denominator)
    
    return distance

def save_matrix_tsv(matrix, headers, filename):
    matrix_with_headers = np.vstack((headers, matrix))
    with open(filename + ".tsv", 'w') as outfile:
        outfile.write("\t".join(headers) + "\n")
        np.savetxt(outfile, matrix_with_headers, delimiter="\t", fmt="%s")

#---------------------------------------------------------------------------------------------------------------------#
# Simulated data
#---------------------------------------------------------------------------------------------------------------------#

# TO DO:         - Run 1 timepoint with Unifrac, also displaying the PCA plots

# Main.py should output a table of sample vs sample distances for post-analysis

np.random.seed(100)

def generate_data(signatures, n_samples=100, n_features=2):
    X = np.random.uniform(low=0.0, high=1.0, size=(n_samples, n_features))
    labels = np.ones((X.shape[0]), dtype=int)

    n_signatures = len(signatures)
    samples_per_signature = n_samples // (2 * n_signatures)
    data = np.zeros((n_samples, len(signatures[0])), dtype=int)

    for i in range(n_samples):
        signature_idx = i % n_signatures
        signature = signatures[signature_idx]
        inverse_signature = signature[::-1]
        if i % (2 * samples_per_signature) < samples_per_signature:
            data[i, :] = signature
            labels[i] = signature_idx
        else:
            data[i, :] = inverse_signature
            labels[i] = signature_idx + n_signatures
    
    linear_eq = data * X

    return linear_eq.T, cosine_similarity(X.T), labels

#---------------------------------------------------------------------------------------------------------------------#
# Case study data Sponges
#---------------------------------------------------------------------------------------------------------------------#

#import dendropy
#
#tree = dendropy.Tree.get(path="/home/pokepup/DTU_Subjects/MSc_thesis/data/case_study/raw_data/tree_relabelled.tre", schema='newick')
#pdm = tree.phylogenetic_distance_matrix()
#pdm.to_csv("/home/pokepup/DTU_Subjects/MSc_thesis/data/case_study/raw_data/tree_distances.csv")

# Parallel C interface optimization
os.environ["USE_INTEL_MKL"] = "1"
mkl.set_num_threads(4)

#---------------------------------------------------------------------------------------------------------------------#
# Optimization algorithm
#---------------------------------------------------------------------------------------------------------------------#

def initialize_theta(X):
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
def grad_function(X, W):
    M = X * W
    _, eigval, eigvec = np.linalg.svd(M)

    # gradient & variance explained
    grad = X * np.dot(eigvec[:,0], np.transpose(eigvec[:,0]))
    e_sum = np.sum(eigval)
    if e_sum == 0:
        var_explained = 0
    else:
        var_explained = np.sum(eigval[:2]) / e_sum

    return grad, var_explained, eigval

@njit
def add_column(m1, m2):
    return np.column_stack((m1, m2))

def optimization(X, alpha=0.1, num_iters=100, epss=np.finfo(np.float64).eps):
    X[np.isnan(X)] = 0
    W = initialize_theta(X)
    #df = pd.DataFrame(columns=["iter", "variance_explained", "eigval1", "eigval2"])
    #Weight_stack = pd.DataFrame(columns=["weights"])
    best_W, iter = np.ones((X.shape[0], X.shape[0]), dtype=np.float64), 0
    # Computes first variance
    # If optimization cannot succeed, returns original
    _, s, _ = np.linalg.svd(X)
    e_sum = np.sum(s)
    best_var = np.sum(s[:2]) / e_sum
    original_var = best_var
    prev_var = best_var
    #df.loc[0] = [0, np.real(best_var), np.real(s[0]), np.real(s[1])]

    Weight_stack = W[:,0]
    for i in range(num_iters):
        get_grad, current_var, eigval = grad_function(X, W)
        abs_diff = np.absolute(current_var - prev_var)
        # epss is based on the machine precision of np.float64 64
        #df.loc[i+1] = [i+1, np.real(current_var), np.real(eigval[0]), np.real(eigval[1])]
        
        # Early stopping
        #if abs_diff < epss:
        #    break

        if current_var > best_var:
            best_var = current_var
            best_W = W
            iter = i+1
        
        W += (alpha * get_grad)        
        W = np.clip(W, 0.0, 1.0)
        prev_var = current_var
        Weight_stack = add_column(Weight_stack, W[:,0])

    return best_W, best_var, original_var, iter, Weight_stack

#---------------------------------------------------------------------------------------------------------------------#
# Visualizing simulated data
#---------------------------------------------------------------------------------------------------------------------#

def GD_parameters(data, title, it_W, a=0.01):
    fig, (ax1, ax2, ax3) = plt.subplots(3)
    fig.set_size_inches(10, 8)
    ax1.plot(data["iter"], data["variance_explained"], label=f"a= {a}")
    ax1.axvline(x=it_W, ls='--', c="red", label=f"a = {a}")
    ax1.set_xlabel("iterations")
    ax1.set_title("Variance explained")
    ax2.plot(data["iter"], data["eigval1"], label=f"a = {a}")
    ax2.set_xlabel(f"iterations")
    ax2.set_title("Eigenvalue 1")
    ax3.plot(data["iter"], data["eigval2"], label=f"a = {a}")
    ax3.set_xlabel("iterations")
    ax3.set_title("Eigenvalue 2")
    fig.tight_layout(pad=2.0)
    fig.savefig(f"../{title}_statistics.png", format='png')
    plt.clf()

#GD_parameters(data=df_cscs, title="sparse30_10000F_cscs" , it_W=it_W_cscs, a=a)
#multi_stats(data=data_u, titles=titles, plabel=groups, filename="sparse30_10000F_unweighted")
#multi_stats(data=data_w, titles=titles, plabel=groups, filename="sparse30_10000F_weighted")
#multi_heatmaps(weights_series, titles, filename="sparse30_10000F_metrics")

#---------------------------------------------------------------------------------------------------------------------#
# Assessing sparse density effect on Permanova & Variance explained on Simulated data
#---------------------------------------------------------------------------------------------------------------------#

"""
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 
warnings.simplefilter("ignore", category=FutureWarning) 

num_iters = 10
sparse_densities = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
features = 100
sample_size = [10, 20, 40, 60, 80, 100]

for s in range(0, num_iters):
    print(f"Starting duplicate {s+1} out of {num_iters}")
    df = pd.DataFrame(columns=["duplicates", "sparse_level", "sample_size", "n_features", "metric_ID", "var_explained", "F_stat", "p_val"])
    for swab, sparse_d in itertools.product(sample_size, sparse_densities):
        # simulated data
        test = scipy.sparse.random(1, features, density=sparse_d, random_state=np.random.default_rng(), data_rvs=scipy.stats.poisson(50, loc=10).rvs)
        label_compact = test.A.tolist()
        samples, css, groups = generate_data(signatures=label_compact, n_features=len(label_compact[0]), n_samples=swab)

        # distance metrics
        # Bray curtis
        BC = np.zeros([samples.shape[1], samples.shape[1]], dtype=np.float64)
        for i,j in itertools.combinations(range(0, samples.shape[1]), 2):
            BC[i,j] = scipy.spatial.distance.braycurtis(samples[:,i], samples[:,j])
            BC[j,i] = BC[i,j]
        BC = 1 - BC

        # Jaccard distance
        JD = np.zeros([samples.shape[1], samples.shape[1]], dtype=np.float64)
        for i,j in itertools.combinations(range(0, samples.shape[1]), 2):
            JD[i,j] = jaccard_distance(samples[:,i], samples[:,j])
            JD[j,i] = JD[i,j]
        JD[np.diag_indices(JD.shape[0])] = 1.0 

        # Jensen-Shannon divergence
        JSD = np.zeros([samples.shape[1], samples.shape[1]], dtype=np.float64)
        for i,j in itertools.combinations(range(0, samples.shape[1]), 2):
            JSD[i,j] = scipy.spatial.distance.jensenshannon(samples[:,i], samples[:,j])
            JSD[j,i] = JSD[i,j]
        JSD[np.isnan(JSD)] = 0
        JSD[np.diag_indices(JD.shape[0])] = 1.0 

        # Euclidean distance
        Euc = np.zeros([samples.shape[1], samples.shape[1]], dtype=np.float64)
        for i,j in itertools.combinations(range(0, samples.shape[1]), 2):
            Euc[i,j] = scipy.spatial.distance.euclidean(samples[:,i], samples[:,j])
            Euc[j,i] = Euc[i,j]
        Euc[np.diag_indices(Euc.shape[0])] = 1.0

        cscs_u = Parallelize(cscs, samples, css)
        cscs_u.astype(np.float64)

        W_cscs, var_cscs_w, var_cscs_u = optimization(cscs_u)
        W_BC, var_BC_w, var_BC_u = optimization(BC)
        W_JD, var_JD_w, var_JD_u = optimization(JD)
        W_JSD, var_JSD_w, var_JSD_u = optimization(JSD)
        W_Euc, var_Euc_w, var_Euc_u = optimization(Euc)
        
        cscs_w = cscs_u * W_cscs
        BC_w = BC * W_BC
        JD_w = JD * W_JD
        JSD_w = JSD * W_JSD
        Euc_w = Euc * W_Euc
        
        data_u = [cscs_u, BC, JD, JSD, Euc]
        data_w = [cscs_w, BC_w, JD_w, JSD_w, Euc_w]
        var_u = [var_cscs_u, var_BC_u, var_JD_u, var_JSD_u, var_Euc_u]
        var_w = [var_cscs_w, var_BC_w, var_JD_w, var_JSD_w, var_Euc_w]
        title_u = ["CSCS", "Bray-curtis", "Jaccard", "Jensen-Shannon", "Euclidean"]
        title_w = ["CSCS_w", "Bray-curtis_w", "Jaccard_w", "Jensen-Shannon_w", "Euclidean_w"]
        heatmap_title = f"{s+1}_{swab}_{sparse_d}"

        for n, id in enumerate(data_u):
            dist = 1 - id
            np.fill_diagonal(dist, 0.0)
            dist = skbio.DistanceMatrix(dist)
            result = skbio.stats.distance.permanova(dist, groups, permutations=9999)
            row = {"duplicates": s+1, "sparse_level": sparse_d, "sample_size": swab, "n_features": features, "metric_ID": title_u[n],\
                "var_explained": var_u[n], "F_stat": result["test statistic"], "p_val": result["p-value"]}
            df = df.append(row, ignore_index=True)
        
        for n, id in enumerate(data_w):
            id[np.isnan(id)] = 0.0
            dist = id / id[0,0]
            dist = 1 - dist

            np.fill_diagonal(dist, 0.0)
            dist = skbio.DistanceMatrix(dist)
            result = skbio.stats.distance.permanova(dist, groups, permutations=9999)
            row = {"duplicates": s+1, "sparse_level": sparse_d, "sample_size": swab, "n_features": features, "metric_ID": title_w[n],\
                "var_explained": var_w[n], "F_stat": result["test statistic"], "p_val": result["p-value"]}
            df = df.append(row, ignore_index=True)

            #if n == 0:
            #    multi_heatmaps(data=[Weight_stack], titles=title_w[n], filename=heatmap_title)
    if s == 0:
        df.to_csv("/home/pokepup/DTU_Subjects/MSc_thesis/scripts/python/Benchmark_simulated.csv", mode='a', header=True, index=False)
    df.to_csv("/home/pokepup/DTU_Subjects/MSc_thesis/scripts/python/Benchmark_simulated.csv", mode='a', header=False, index=False)
"""
#---------------------------------------------------------------------------------------------------------------------#
# Assessing sparse density effect on Permanova & Variance explained on Empirical data
#---------------------------------------------------------------------------------------------------------------------#

from Bio import SeqIO
from Bio.Blast.Applications import NcbiblastnCommandline

import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 
warnings.simplefilter("ignore", category=FutureWarning) 
warnings.filterwarnings("ignore", category=RuntimeWarning)

file_path = "/home/pokepup/DTU_Subjects/MSc_thesis/data/Mice_data/"
blast_file = "/home/pokepup/DTU_Subjects/MSc_thesis/data/Mice_data/720sample.16S.otu.repTag.filter.fasta"

# groups based on "Sample.Time"
metadata = pd.read_csv(file_path + "metadata.csv", sep=",", header=0, usecols=["Sample.ID","Sample.Time"])

group_A, group_B = [], []
groups = []
for i,j in zip(metadata["Sample.ID"], metadata["Sample.Time"]):
    if j == "Pre diet":
        groups.append(1)
        group_A.append(i)
    elif j == "Termination":
        groups.append(0)
        group_B.append(i)

# Separate OTU_table into two groups
OTU_table = pd.read_csv(file_path + "otu_table.csv", sep=",", header=0, index_col=0)
samples_ids = OTU_table.columns.tolist()
otu_ids = OTU_table.index.tolist()
samples = OTU_table.values
feature_ids = {str(id):it for it, id in enumerate(list(OTU_table.index))}

# Creates temporary blast file
pre_filter = [pair for pair in SeqIO.parse(blast_file, "fasta") if pair.id in feature_ids]
tmp_file = os.path.join("../tmp.fa")
SeqIO.write(pre_filter, tmp_file, "fasta")

# pairwise blast alignment
cline = NcbiblastnCommandline(query = tmp_file, subject = tmp_file, outfmt=6, out='-', max_hsps=1)
blast_output = cline()[0].strip().split("\n")

# samples css from blast
css_matrix = scipy.sparse.dok_matrix((len(feature_ids), len(feature_ids)), dtype=np.float64)
for line in blast_output:
    line = line.split("\t")
    if line[0] in feature_ids and line[1] in feature_ids:
        css_matrix[feature_ids[line[0]], feature_ids[line[1]]] = float(line[2])*0.01
        css_matrix[feature_ids[line[1]], feature_ids[line[0]]] = float(line[2])*0.01
os.remove(tmp_file)

# distance metrics
# Bray curtis
BC = np.zeros([samples.shape[1], samples.shape[1]], dtype=np.float64)
for i,j in itertools.combinations(range(0, samples.shape[1]), 2):
    BC[i,j] = scipy.spatial.distance.braycurtis(samples[:,i], samples[:,j])
    BC[j,i] = BC[i,j]
BC = 1 - BC
np.fill_diagonal(BC, 1.0)

# Jaccard distance
JD = np.zeros([samples.shape[1], samples.shape[1]], dtype=np.float64)
for i,j in itertools.combinations(range(0, samples.shape[1]), 2):
    JD[i,j] = jaccard_distance(samples[:,i], samples[:,j])
    JD[j,i] = JD[i,j]
JD[np.diag_indices(JD.shape[0])] = 1.0 

# Jensen-Shannon divergence
JSD = np.zeros([samples.shape[1], samples.shape[1]], dtype=np.float64)
for i,j in itertools.combinations(range(0, samples.shape[1]), 2):
    JSD[i,j] = scipy.spatial.distance.jensenshannon(samples[:,i], samples[:,j])
    JSD[j,i] = JSD[i,j]
JSD[np.isnan(JSD)] = 0
JSD[np.diag_indices(JD.shape[0])] = 1.0 

# Euclidean distance
Euc = np.zeros([samples.shape[1], samples.shape[1]], dtype=np.float64)
for i,j in itertools.combinations(range(0, samples.shape[1]), 2):
    Euc[i,j] = scipy.spatial.distance.euclidean(samples[:,i], samples[:,j])
    Euc[j,i] = Euc[i,j]
Euc[np.diag_indices(Euc.shape[0])] = 1.0

from skbio.diversity.beta import weighted_unifrac

# Convert the DataFrame to a skbio table
otu_table = skbio.TabularMSA(OTU_table.values.T, index=OTU_table.columns)

# Load your phylogenetic tree
tree = skbio.TreeNode.read('phylogenetic_tree.nwk')

# Calculate weighted UniFrac distance matrix
wu_dm = weighted_unifrac(otu_table, tree, normalized=True)
print(wu_dm)


"""
cscs_u = Parallelize(cscs, samples, css_matrix.toarray())
cscs_u.astype(np.float64)

W_cscs, var_cscs_w, var_cscs_u, cscs_it, cscs_weights = optimization(cscs_u)
W_BC, var_BC_w, var_BC_u, BC_it, BC_weights = optimization(BC)
W_JD, var_JD_w, var_JD_u, JD_it, JD_weights = optimization(JD)
W_JSD, var_JSD_w, var_JSD_u, JSD_it, JSD_weights = optimization(JSD)
W_Euc, var_Euc_w, var_Euc_u, Euc_it, Euc_weights = optimization(Euc)

cscs_w = cscs_u * W_cscs
BC_w = BC * W_BC
JD_w = JD * W_JD
JSD_w = JSD * W_JSD
Euc_w = Euc * W_Euc

data_u = [cscs_u, BC, JD, JSD, Euc]
data_w = [cscs_w, BC_w, JD_w, JSD_w, Euc_w]
var_u = [var_cscs_u, var_BC_u, var_JD_u, var_JSD_u, var_Euc_u]
var_w = [var_cscs_w, var_BC_w, var_JD_w, var_JSD_w, var_Euc_w]
title_u = ["CSCS", "Bray-curtis", "Jaccard", "Jensen-Shannon", "Euclidean"]
title_w = ["CSCS_w", "Bray-curtis_w", "Jaccard_w", "Jensen-Shannon_w", "Euclidean_w"]
weights = [cscs_weights, BC_weights, JD_weights, JSD_weights, Euc_weights]
iters = [cscs_it, BC_it, JD_it, JSD_it, Euc_it]

heatmap_title = f"mice_data"

multi_stats(data=data_u, titles=title_u, filename="../simulated_mice_unweighted", plabel=groups)

multi_stats(data=data_w, titles=title_w, filename="../simulated_mice_weighted", plabel=groups)
multi_heatmaps(data=weights, titles=title_w, filename=heatmap_title, vline=iters)
"""

#def construct_matrix(sparse_d, n_samples, samples_ids, group_A, group_B, n_features=100):
#    # Subsets groups
#    array_A = OTU_table.values[:, np.isin(samples_ids, group_A)]
#    array_B = OTU_table.values[:, np.isin(samples_ids, group_B)]
#
#    # create zero and positive indices per group
#    zero_indices_A = np.argwhere(array_A == 0)
#    pos_indices_A = np.argwhere(array_A > 0)
#    zero_indices_B = np.argwhere(array_B == 0)
#    pos_indices_B = np.argwhere(array_B > 0)
#
#    # sample parameters
#    n_samples = n_samples // 2
#    num_zero_cols = int(sparse_d * n_samples)
#    num_pos_cols = n_samples - num_zero_cols
#    row_list = []
#    otu_idx = []
#    
#    rows = np.unique(np.concatenate([zero_indices_A[:,0], zero_indices_B[:,0]]))
#    for row_idx in rows:
#        zero_indices_A_row = zero_indices_A[zero_indices_A[:, 0] == row_idx]
#        pos_indices_A_row = pos_indices_A[pos_indices_A[:, 0] == row_idx]
#        zero_indices_B_row = zero_indices_B[zero_indices_B[:, 0] == row_idx]
#        pos_indices_B_row = pos_indices_B[pos_indices_B[:, 0] == row_idx]
#        
#        if (((len(zero_indices_A_row) >= num_zero_cols and len(zero_indices_B_row) >= num_zero_cols)) and \
#            ((len(pos_indices_A_row) >= num_pos_cols and len(pos_indices_B_row) >= num_pos_cols))) and row_idx not in otu_idx:
#            otu_idx.append(row_idx)
#            row = np.hstack((
#                array_A[row_idx, zero_indices_A_row[:, 1][:num_zero_cols]],
#                array_A[row_idx, pos_indices_A_row[:, 1][:num_pos_cols]],
#                array_B[row_idx, zero_indices_B_row[:, 1][:num_zero_cols]],
#                array_B[row_idx, pos_indices_B_row[:, 1][:num_pos_cols]],
#            ))
#            row_list.append(row)
#        if len(row_list) == n_features:
#            break
#
#    return np.vstack(row_list), otu_idx

"""
# parameters to test
sparse_densities = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0]
n_samples = [10, 20, 40, 60, 80, 100]
num_iters = 10

for s in range(1, num_iters):
    print(f"Starting duplicate {s+1} out of {num_iters}")
    df = pd.DataFrame(columns=["duplicates", "sparse_level", "sample_size", "n_features", "metric_ID", "var_explained", "F_stat", "p_val"])
    for swab, sparse_d in itertools.product(n_samples, sparse_densities):
        print(f"Starting combination sample size: {swab} and sparsity: {sparse_d}")
        # construct matrix sample
        samples, otu_idx = construct_matrix(sparse_d, swab, samples_ids, group_A, group_B)
        feature_ids = {str(otu_ids[otu_idx[i]]): i for i in range(len(otu_idx))}
        groups = np.concatenate((np.ones((swab//2,)), np.zeros((swab//2,))), axis=0)

        # Creates temporary blast file
        pre_filter = [pair for pair in SeqIO.parse(blast_file, "fasta") if pair.id in feature_ids]
        tmp_file = os.path.join("../tmp.fa")
        SeqIO.write(pre_filter, tmp_file, "fasta")

        # pairwise blast alignment
        cline = NcbiblastnCommandline(query = tmp_file, subject = tmp_file, outfmt=6, out='-', max_hsps=1)
        blast_output = cline()[0].strip().split("\n")

        # samples css from blast
        css_matrix = scipy.sparse.dok_matrix((len(feature_ids), len(feature_ids)), dtype=np.float64)
        for line in blast_output:
            line = line.split("\t")
            if line[0] in feature_ids and line[1] in feature_ids:
                css_matrix[feature_ids[line[0]], feature_ids[line[1]]] = float(line[2])*0.01
                css_matrix[feature_ids[line[1]], feature_ids[line[0]]] = float(line[2])*0.01
        os.remove(tmp_file)

        # distance metrics
        # Bray curtis
        BC = np.zeros([samples.shape[1], samples.shape[1]], dtype=np.float64)
        for i,j in itertools.combinations(range(0, samples.shape[1]), 2):
            BC[i,j] = scipy.spatial.distance.braycurtis(samples[:,i], samples[:,j])
            BC[j,i] = BC[i,j]
        BC = 1 - BC
        np.fill_diagonal(BC, 1.0)
        BC[np.isnan(BC)] = 0

        # Jaccard distance
        JD = np.zeros([samples.shape[1], samples.shape[1]], dtype=np.float64)
        for i,j in itertools.combinations(range(0, samples.shape[1]), 2):
            JD[i,j] = jaccard_distance(samples[:,i], samples[:,j])
            JD[j,i] = JD[i,j]
        JD[np.diag_indices(JD.shape[0])] = 1.0 

        # Jensen-Shannon divergence
        JSD = np.zeros([samples.shape[1], samples.shape[1]], dtype=np.float64)
        for i,j in itertools.combinations(range(0, samples.shape[1]), 2):
            JSD[i,j] = scipy.spatial.distance.jensenshannon(samples[:,i], samples[:,j])
            JSD[j,i] = JSD[i,j]
        JSD[np.isnan(JSD)] = 0
        JSD[np.diag_indices(JD.shape[0])] = 1.0 

        # Euclidean distance
        Euc = np.zeros([samples.shape[1], samples.shape[1]], dtype=np.float64)
        for i,j in itertools.combinations(range(0, samples.shape[1]), 2):
            Euc[i,j] = scipy.spatial.distance.euclidean(samples[:,i], samples[:,j])
            Euc[j,i] = Euc[i,j]
        Euc[np.diag_indices(Euc.shape[0])] = 1.0

        cscs_u = Parallelize(cscs, samples, css_matrix.toarray())
        cscs_u.astype(np.float64)

        W_cscs, var_cscs_w, var_cscs_u = optimization(cscs_u)
        W_BC, var_BC_w, var_BC_u = optimization(BC)
        W_JD, var_JD_w, var_JD_u = optimization(JD)
        W_JSD, var_JSD_w, var_JSD_u = optimization(JSD)
        W_Euc, var_Euc_w, var_Euc_u = optimization(Euc)
        
        cscs_w = cscs_u * W_cscs
        BC_w = BC * W_BC
        JD_w = JD * W_JD
        JSD_w = JSD * W_JSD
        Euc_w = Euc * W_Euc
        
        data_u = [cscs_u, BC, JD, JSD, Euc]
        data_w = [cscs_w, BC_w, JD_w, JSD_w, Euc_w]
        var_u = [var_cscs_u, var_BC_u, var_JD_u, var_JSD_u, var_Euc_u]
        var_w = [var_cscs_w, var_BC_w, var_JD_w, var_JSD_w, var_Euc_w]
        title_u = ["CSCS", "Bray-curtis", "Jaccard", "Jensen-Shannon", "Euclidean"]
        title_w = ["CSCS_w", "Bray-curtis_w", "Jaccard_w", "Jensen-Shannon_w", "Euclidean_w"]
        heatmap_title = f"{s+1}_{swab}_{sparse_d}"

        for n, id in enumerate(data_u):
            dist = 1 - id
            np.fill_diagonal(dist, 0.0)
            dist = skbio.DistanceMatrix(dist)
            result = skbio.stats.distance.permanova(dist, groups, permutations=9999)
            row = {"duplicates": s+1, "sparse_level": sparse_d, "sample_size": swab, "n_features": len(feature_ids), "metric_ID": title_u[n],\
                "var_explained": var_u[n], "F_stat": result["test statistic"], "p_val": result["p-value"]}
            df = df.append(row, ignore_index=True)
        
        for n, id in enumerate(data_w):
            id[np.isnan(id)] = 0.0
            dist = id / id[0,0]
            dist = 1 - dist

            np.fill_diagonal(dist, 0.0)
            dist = skbio.DistanceMatrix(dist)
            result = skbio.stats.distance.permanova(dist, groups, permutations=9999)
            row = {"duplicates": s+1, "sparse_level": sparse_d, "sample_size": swab, "n_features": len(feature_ids), "metric_ID": title_w[n],\
                "var_explained": var_w[n], "F_stat": result["test statistic"], "p_val": result["p-value"]}
            df = df.append(row, ignore_index=True)

        # intitiate garbage collector
        samples, otu_idx, feature_ids, groups = None, None, None, None
        pre_filter, blast_output, css_matrix, result = None, None, None, None
        BC, JD, JSD, Euc, cscs_u = None, None, None, None, None
        cscs_w, BC_w, JD_w, JSD_w, Euc_w = None, None, None, None, None
        gc.collect()

    if s == 0:
        df.to_csv("/home/pokepup/DTU_Subjects/MSc_thesis/scripts/python/Benchmark_Emprical_test.csv", mode='a', header=True, index=False)
    df.to_csv("/home/pokepup/DTU_Subjects/MSc_thesis/scripts/python/Benchmark_Emprical_test.csv", mode='a', header=False, index=False)
"""
#---------------------------------------------------------------------------------------------------------------------#
# Case study: Sponges
#---------------------------------------------------------------------------------------------------------------------#

#import biom

#table = biom.load_table('/home/pokepup/DTU_Subjects/MSc_thesis/data/case_study/1_70/table.biom')
#
#df = pd.DataFrame(table.to_dataframe())
#df.to_csv('/home/pokepup/DTU_Subjects/MSc_thesis/data/case_study/1_70/table.tsv', sep="\t")
#blast_file = open("/home/pokepup/DTU_Subjects/MSc_thesis/scripts/python/case_study.blast", "r")

#biom_table = pd.read_csv("/home/pokepup/DTU_Subjects/MSc_thesis/data/case_study/1_70/table.tsv", sep="\t", header=0, index_col=0)
#sample_ids = biom_table.columns.tolist()
#samples = biom_table.values
#
#sample_size = len(sample_ids)
#
#css_matrix = scipy.sparse.dok_matrix((len(feature_ids), len(feature_ids)), dtype=np.float64)
#for line in blast_file:
#    line = line.split()
#    if line[0] in feature_ids and line[1] in feature_ids:
#        css_matrix[feature_ids[line[0]], feature_ids[line[1]]] = float(line[2])*0.01
#        css_matrix[feature_ids[line[1]], feature_ids[line[0]]] = float(line[2])*0.01
#
#cscs_u = Parallelize(cscs, samples, css_matrix.toarray())
#cscs_u.astype(np.float64)
#
#df = pd.DataFrame(data=cscs_u, index=sample_ids, columns=sample_ids)
#
## save the DataFrame to a CSV file with sample_ids on both axes
#df.to_csv("/home/pokepup/DTU_Subjects/MSc_thesis/data/case_study/1_150/CSCS_distances.tsv")


#path_case_data = "/home/pokepup/DTU_Subjects/MSc_thesis/data/case_study/1_70/"
#
#metadata_df = pd.read_csv("/home/pokepup/DTU_Subjects/MSc_thesis/data/case_study/1_70/metadata.tsv", sep="\t", usecols=["org_index", "health_status"])
#
#Unifrac_df = pd.read_csv(path_case_data + "GUniFrac_alpha_one_Distance.tsv", sep="\t", header=0, index_col=0)
#Braycurtis_df = pd.read_csv(path_case_data + "Bray_Distance.tsv", sep="\t", header=0, index_col=0)
#CSCS_df = pd.read_csv(path_case_data + "CSCS_distances.tsv", sep=",", header=0, index_col=0)
#Jaccard_df = pd.read_csv(path_case_data + "jaccard_dist.csv", sep=",", header=0, index_col=0)
#JSD_df = pd.read_csv(path_case_data + "JSD_dist.csv", sep=",", header=0, index_col=0)
#
#reference_IDs = metadata_df["org_index"].tolist()
#conditions = metadata_df["health_status"].tolist()
#
#groups = []
#
#for id in Braycurtis_df.columns:
#    if id in reference_IDs:
#        idx = reference_IDs.index(id)
#        if conditions[idx] == "Healthy":
#            groups.append(0)
#        else:
#            groups.append(1)
#
#cscs_u = CSCS_df.values
#Unifrac_u = 1 - Unifrac_df.values
#Bray_u = 1 - Braycurtis_df.values
#Jaccard_u = Jaccard_df.values
#JSD_u = JSD_df.values
#
#W_cscs, _, _ = optimization(cscs_u)
#W_Unifrac, _, _ = optimization(Unifrac_u)
#W_Bray, _, _ = optimization(Bray_u)
#W_Jaccard, _, _ = optimization(Jaccard_u)
#W_JSD, _, _ = optimization(JSD_u)
#
#cscs_w = cscs_u * W_cscs
#Unifrac_w = Unifrac_u * W_Unifrac
#Bray_w = Bray_u * W_Bray
#Jaccard_w = Jaccard_u * W_Jaccard
#JSD_w = JSD_u * W_JSD
#
#titles_u = ["CSCS", "Unifrac", "Bray-Curtis", "Jaccard", "Jensen-Shannon"]
#titles_w = ["CSCS_w", "Unifrac_w", "Bray-Curtis_w", "Jaccard_w", "Jensen-Shannon_w"]
#data_u = [cscs_u, Unifrac_u, Bray_u, Jaccard_u, JSD_u]
#data_w = [cscs_w, Unifrac_w, Bray_w, Jaccard_w, JSD_w]
#
#multi_stats(data=data_u, titles=titles_u, filename="../Case_study_unweighted", plabel=groups)
#multi_stats(data=data_w, titles=titles_w, filename="../Case_study_weighted", plabel=groups)
