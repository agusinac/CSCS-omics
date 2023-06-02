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
import matplotlib.patches as mpatches

#---------------------------------------------------------------------------------------------------------------------#
# Functions under development
#---------------------------------------------------------------------------------------------------------------------#

def multi_heatmaps(data, titles, filename, y_labels, vline = None, ncols=2):
    plt.figure(figsize=(25, 20))
    plt.subplots_adjust(left=0.5, hspace=0.2)
    plt.rcParams.update({'font.size': 12})
    plt.rcParams['ytick.major.pad']='8'

    for n, id in enumerate(data):
        ax = plt.subplot(ncols, len(data) // ncols + (len(data) % ncols > 0), n + 1)
        sns.heatmap(id, ax=ax)
        ax.set_title(f"{titles[n]}")
        ax.set_xlabel("Iterations")
        ax.set_ylabel("samples")
        ax.set_yticks(range(len(y_labels)))
        ax.set_yticklabels(y_labels)
        if vline is not None:
            ax.axvline(x=vline[n], linestyle=':', color='grey')

    ax.legend()
    plt.tight_layout()
    plt.savefig(f"../{filename}_multi_heatmaps.png", format='png')
    plt.close()

def assign_random_colors(sorted_labels):
    unique_variables = list(set(sorted_labels))
    num_colors_needed = len(unique_variables)
    color_palette = sns.color_palette("hls", num_colors_needed)
    color_mapping = {variable: color for variable, color in zip(unique_variables, color_palette)}

    replaced_list = []
    for item in sorted_labels:
        if item in color_mapping:
            replaced_list.append(color_mapping[item])
        else:
            replaced_list.append(item)
    return replaced_list, color_mapping

def multi_stats(data, titles, filename, sorted_labels, ncols=3):
    # Setup for figure and font size
    plt.figure(figsize=(15, 15))
    plt.subplots_adjust(hspace=0.2)
    plt.rcParams.update({'font.size': 12})

    # Defines same colors for members

    permanova_color = sns.color_palette('hls', len(data))
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
        result = skbio.stats.distance.permanova(dist, sorted_labels, permutations=9999)
        F_stats.loc[n] = [result["test statistic"], result["p-value"]]

        # plots components and variances
        replaced_list, color_mapping = assign_random_colors(sorted_labels)
        for i in range(id.shape[1]):
            ax.scatter(pcs[0][i], pcs[1][i], s=10, color=replaced_list[i])

        # Adds labels and R-squared
        ax.set_xlabel(f"PC1: {round(var[0]*100,2)}%")
        ax.set_ylabel(f"PC2: {round(var[1]*100,2)}%")
        ax.set_title(f"{titles[n]}")

        # Creates dummy legend colors
        group_labels = list(color_mapping.keys())
        group_colors = [color_mapping[label] for label in group_labels]
        legend_elements = [mpatches.Patch(color=color) for color in group_colors]
        ax.legend(legend_elements, group_labels, facecolor='white', edgecolor='black')
    
    # plots barplot of permanova
    ax = plt.subplot(ncols, len(data) // ncols + (len(data) % ncols > 0), n + 2)
    ax.bar(titles, F_stats["F-test"], color=permanova_color, label=["$p={:.4f}$".format(pv) for pv in F_stats["P-value"]])
    ax.set_title("PERMANOVA")
    ax.set_xlabel("distance metric")
    ax.set_ylabel("Pseudo-F test statistic")
    ax.set_xticklabels(titles, rotation = 45)
    ax.legend(facecolor='white', edgecolor='black')
    plt.tight_layout()
    plt.savefig(f"../{filename}.png", format='png')
    plt.close()

def data_dump(data, title):
    file = open(f"../{title}", "wb")
    pickle.dump(data, file)
    file.close()

def cscs_w(A, B, css):
    cssab = np.multiply(css, np.multiply(A, B.T)) 
    cssab_w = cssab * optimization(X=cssab)
    cssaa = np.multiply(css, np.multiply(A, A.T))
    cssbb = np.multiply(css, np.multiply(B, B.T))
    scaler = max(np.sum(cssaa), np.sum(cssbb))
    if scaler == 0:
        result = 0
    else:
        result = np.sum(cssab_w) / scaler
    return result

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

def jaccard_distance(A, B):
    #Find symmetric difference of two sets
    nominator = np.setdiff1d(A, B)

    #Find union of two sets
    denominator = np.union1d(A, B)

    #Take the ratio of sizes
    distance = len(nominator)/len(denominator)
    
    return distance

def save_matrix_tsv(matrix, headers, filename):
    with open(filename + ".tsv", 'w') as outfile:
        outfile.write("\t".join(headers) + "\n")
        np.savetxt(outfile, matrix, delimiter="\t")

#---------------------------------------------------------------------------------------------------------------------#
# Simulated data
#---------------------------------------------------------------------------------------------------------------------#

# TO DO:         - Run 1 timepoint with Unifrac, also displaying the PCA plots

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
    samples = linear_eq.T
    norm_samples = np.asarray(np.divide(samples,samples.sum(axis=0)), dtype=np.float64)

    return norm_samples, cosine_similarity(X.T), labels

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

@njit
def initialize_theta(X):
    alpha, beta, _, _ = scipy.stats.beta.fit(X, floc=0, fscale=1)
    w = np.random.beta(alpha, beta, size=X.shape[0])
    W = np.triu(w, 1) + np.triu(w, 1).T 
    W.astype(np.float64)
    return W

@njit
def grad_function(X, W):
    M = X * W
    _, eigval, eigvec = np.linalg.svd(M)

    # gradient & variance explained
    grad = X * np.multiply(eigvec[:,0], eigvec[:,0].T)
    grad = np.triu(grad, 1) + np.triu(grad, 1).T 
    np.fill_diagonal(grad, 1)
    e_sum = np.sum(eigval)
    if e_sum == 0:
        var_explained = 0
    else:
        var_explained = np.sum(eigval[:2]) / e_sum

    return grad, var_explained, eigval

def add_column(col1, col2):
    return np.column_stack((col1, col2))

def theta_diff(matrix):
    return np.sum(np.diff(matrix, axis=1), axis=1)
    
def optimization(X, alpha=0.1, num_iters=100, epss=np.finfo(np.float64).eps):
    X[np.isnan(X)] = 0
    W = initialize_theta(X)
    best_W, iter = np.ones((X.shape[0], X.shape[0]), dtype=np.float64), 0
    # Computes first variance
    # If optimization cannot succeed, returns original
    s = np.linalg.svd(X, compute_uv=False)
    
    e_sum = np.sum(s)
    best_var = np.sum(s[:2]) / e_sum
    #original_var = best_var
    prev_var = best_var
    
    #Weight_stack = theta_diff(W)

    for i in range(num_iters):
        get_grad, current_var, _ = grad_function(X, W)
        
        # Early stopping
        if np.absolute(current_var - prev_var) < epss:
            break

        if current_var > best_var:
            best_var = current_var
            best_W = W
            #iter = i+1
        
        W += (alpha * get_grad)        
        W = np.clip(W, 0, 1)
        prev_var = current_var

        #Weight_stack = add_column(Weight_stack, theta_diff(W))

    return best_W #, best_var, original_var, iter, Weight_stack

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

def benchmark_metrics(samples, css, groups, df, sparse_d, swab, s):
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
    #heatmap_title = f"{s+1}_{swab}_{sparse_d}"

    for n, id in enumerate(data_u):
        dist = 1 - id
        np.fill_diagonal(dist, 0.0)
        dist = skbio.DistanceMatrix(dist)
        result = skbio.stats.distance.permanova(dist, groups, permutations=9999)
        row = {"duplicates": s+1, "sparse_level": sparse_d, "sample_size": swab, "n_features": samples.shape[0], "metric_ID": title_u[n],\
            "var_explained": var_u[n], "F_stat": result["test statistic"], "p_val": result["p-value"]}
        df = df.append(row, ignore_index=True)
    
    for n, id in enumerate(data_w):
        id[np.isnan(id)] = 0.0
        dist = id / id[0,0]
        dist = 1 - dist

        np.fill_diagonal(dist, 0.0)
        dist = skbio.DistanceMatrix(dist)
        result = skbio.stats.distance.permanova(dist, groups, permutations=9999)
        row = {"duplicates": s+1, "sparse_level": sparse_d, "sample_size": swab, "n_features": samples.shape[0], "metric_ID": title_w[n],\
            "var_explained": var_w[n], "F_stat": result["test statistic"], "p_val": result["p-value"]}
        df = df.append(row, ignore_index=True)
    return df

def beta_switch(feature, sparse_d):
    rng = np.random.default_rng()
    rvs = lambda n: np.random.randint(1, 1000, size=n)
    S = scipy.sparse.random(1, feature, density=sparse_d, random_state=rng, data_rvs=rvs)
    return S.toarray()

"""
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 
warnings.simplefilter("ignore", category=FutureWarning) 

num_iters = 10
sparse_densities = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
sparse_densities = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
features_S20 = [914, 826, 759, 705, 674, 655, 633, 621, 621, 718]
features_S40 = [657, 540, 472, 419, 373, 324, 312, 297, 296, 425]
features_S60 = [426, 341, 277, 213, 169, 139, 119, 108, 125, 270]
features_S80 = [211, 167, 118, 91, 60, 42, 35, 43, 59, 165]
sample_size = [20, 40, 60, 80]


for s in range(0, num_iters):
    print(f"Starting duplicate {s+1} out of {num_iters}")
    df = pd.DataFrame(columns=["duplicates", "sparse_level", "sample_size", "n_features", "metric_ID", "var_explained", "F_stat", "p_val"])
    for swab, sparse_d in itertools.product(sample_size, sparse_densities):
        # simulated data
        if swab == 20:
            if sparse_d == 0.1:
                it = 0
            print(f"Starting combination sample size: {swab}, sparsity: {sparse_d}, features: {features_S20[it]}")
            label_compact = beta_switch(features_S20[it], sparse_d)
            samples, css, groups = generate_data(signatures=label_compact.tolist(), n_features=features_S20[it], n_samples=swab)
            df = benchmark_metrics(samples, css, groups, df, sparse_d, swab, s)
            it += 1
        if swab == 40:
            if sparse_d == 0.1:
                it = 0
            print(f"Starting combination sample size: {swab}, sparsity: {sparse_d}, features: {features_S40[it]}")
            label_compact = beta_switch(features_S40[it], sparse_d)
            samples, css, groups = generate_data(signatures=label_compact.tolist(), n_features=features_S40[it], n_samples=swab)
            df = benchmark_metrics(samples, css, groups, df, sparse_d, swab, s)
            it += 1
        if swab == 60:
            if sparse_d == 0.1:
                it = 0
            print(f"Starting combination sample size: {swab}, sparsity: {sparse_d}, features: {features_S60[it]}")
            label_compact = beta_switch(features_S60[it], sparse_d)
            samples, css, groups = generate_data(signatures=label_compact.tolist(), n_features=features_S60[it], n_samples=swab)
            df = benchmark_metrics(samples, css, groups, df, sparse_d, swab, s)
            it += 1
        if swab == 80:
            if sparse_d == 0.1:
                it = 0
            print(f"Starting combination sample size: {swab}, sparsity: {sparse_d}, features: {features_S80[it]}")
            label_compact = beta_switch(features_S80[it], sparse_d)
            samples, css, groups = generate_data(signatures=label_compact.tolist(), n_features=features_S80[it], n_samples=swab)
            df = benchmark_metrics(samples, css, groups, df, sparse_d, swab, s)
            it += 1
        #if n == 0:
            #    multi_heatmaps(data=[Weight_stack], titles=title_w[n], filename=heatmap_title)
    if s == 0:
        df.to_csv("../Benchmark_simulated_10rep.csv", mode='a', header=True, index=False)
    df.to_csv("../Benchmark_simulated_10rep.csv", mode='a', header=False, index=False)
"""
#---------------------------------------------------------------------------------------------------------------------#
# Assessing sparse density effect on Permanova & Variance explained on Empirical data
#---------------------------------------------------------------------------------------------------------------------#
"""
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
samples_ids = {str(id):it for it, id in enumerate(list(OTU_table.columns))}

labels = {int(samples_ids[id]) : group for id, group in zip(metadata["Sample.ID"], metadata["Sample.Time"]) if id in samples_ids}
sorted_labels = [labels[key] for key in sorted(labels.keys())]

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

#from skbio.diversity.beta import weighted_unifrac
#
## Convert the DataFrame to a skbio table
#otu_table = skbio.TabularMSA(OTU_table.values.T, index=OTU_table.columns)
#
## Load your phylogenetic tree
#tree = skbio.TreeNode.read(file_path + "tree/mice_gtr.nwk")
#
## Calculate weighted UniFrac distance matrix
#wu_dm = weighted_unifrac(otu_table, tree, normalized=True)
#print(wu_dm)

cscs_ab_w = Parallelize(cscs_w, samples, css_matrix.toarray())
cscs_ab_w.astype(np.float64)

cscs_u = Parallelize(cscs, samples, css_matrix.toarray())
cscs_u.astype(np.float64)

W_cscs = optimization(cscs_u)
W_BC = optimization(BC)
W_JD = optimization(JD)
W_JSD = optimization(JSD)
W_Euc = optimization(Euc)

cscs_w = cscs_u * W_cscs
BC_w = BC * W_BC
JD_w = JD * W_JD
JSD_w = JSD * W_JSD
Euc_w = Euc * W_Euc

data_u = [cscs_u, BC, JD, JSD, Euc]
data_w = [cscs_w, cscs_ab_w, BC_w, JD_w, JSD_w, Euc_w]
title_u = ["CSCS", "Bray-curtis", "Jaccard", "Jensen-Shannon", "Euclidean"]
title_w = ["CSCS_w", "CSCS_samples_w", "Bray-curtis_w", "Jaccard_w", "Jensen-Shannon_w", "Euclidean_w"]
#weights = [cscs_weights, BC_weights, JD_weights, JSD_weights, Euc_weights]
#iters = [cscs_it, BC_it, JD_it, JSD_it, Euc_it]

heatmap_title = f"empirical_mice_data"

multi_stats(data=data_u, titles=title_u, filename="../empirical_mice_unweighted", sorted_labels=sorted_labels)

multi_stats(data=data_w, titles=title_w, filename="../empirical_mice_weighted", sorted_labels=sorted_labels)
#multi_heatmaps(data=weights, titles=title_w, filename=heatmap_title, vline=iters, y_labels=sorted_labels)
"""
def construct_matrix(sparse_d, n_samples, samples_ids, group_A, group_B, OTU_table):
    # Subsets groups
    array_A = OTU_table.values[:, np.isin(samples_ids, group_A)]
    array_B = OTU_table.values[:, np.isin(samples_ids, group_B)]

    # create zero and positive indices per group
    zero_indices_A = np.argwhere(array_A == 0)
    pos_indices_A = np.argwhere(array_A > 0)
    zero_indices_B = np.argwhere(array_B == 0)
    pos_indices_B = np.argwhere(array_B > 0)

    # sample parameters
    n_samples = n_samples // 2
    num_zero_cols = int(sparse_d * n_samples)
    num_pos_cols = n_samples - num_zero_cols
    row_list = []
    otu_idx = []
    
    for row_idx in range(len(OTU_table.index)):
        zero_indices_A_row = zero_indices_A[zero_indices_A[:, 0] == row_idx]
        pos_indices_A_row = pos_indices_A[pos_indices_A[:, 0] == row_idx]
        zero_indices_B_row = zero_indices_B[zero_indices_B[:, 0] == row_idx]
        pos_indices_B_row = pos_indices_B[pos_indices_B[:, 0] == row_idx]
        
        if (((len(zero_indices_A_row) >= num_zero_cols and len(zero_indices_B_row) >= num_zero_cols)) and \
            ((len(pos_indices_A_row) >= num_pos_cols and len(pos_indices_B_row) >= num_pos_cols))) and row_idx not in otu_idx:
            otu_idx.append(row_idx)

            # creates row A and B, shuffles randomly and concatenates as final row
            row_A = np.hstack((array_A[row_idx, zero_indices_A_row[:, 1][:num_zero_cols]],\
                 array_A[row_idx, pos_indices_A_row[:, 1][:num_pos_cols]]))
            np.random.shuffle(row_A)

            row_B = np.hstack((array_B[row_idx, zero_indices_B_row[:, 1][:num_zero_cols]],\
                array_B[row_idx, pos_indices_B_row[:, 1][:num_pos_cols]]))
            np.random.shuffle(row_B)

            row = np.hstack((row_A, row_B))
            row_list.append(row)

    samples = np.vstack(row_list)
    norm_samples = np.asarray(np.divide(samples,samples.sum(axis=0)), dtype=np.float64)
    return norm_samples, otu_idx

"""
# parameters to test
sparse_densities = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0]
n_samples = [20, 40, 60, 80]
num_iters = 10

for s in range(0, num_iters):
    print(f"Starting duplicate {s+1} out of {num_iters}")
    df = pd.DataFrame(columns=["duplicates", "sparse_level", "sample_size", "n_features", "metric_ID", "var_explained", "F_stat", "p_val"])
    for swab, sparse_d in itertools.product(n_samples, sparse_densities):
        print(f"Starting combination sample size: {swab} and sparsity: {sparse_d}")
        # construct matrix sample
        samples, otu_idx = construct_matrix(sparse_d, swab, samples_ids, group_A, group_B, OTU_table)
        feature_ids = {str(otu_ids[otu_idx[i]]): i for i in range(len(otu_idx))}
        groups = np.concatenate((np.ones((swab//2,)), np.zeros((swab//2,))), axis=0)
        print(f"dimensions: {samples.shape}")
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
        df.to_csv("../Benchmark_Emprical_10rep.csv", mode='a', header=True, index=False)
    df.to_csv("../Benchmark_Emprical_10rep.csv", mode='a', header=False, index=False)
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

#biom_table = pd.read_csv("/home/pokepup/DTU_Subjects/MSc_thesis/data/case_study/1_150/table.tsv", sep="\t", header=0, index_col=0)
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
# Jaccard distance
#JD = np.zeros([samples.shape[1], samples.shape[1]], dtype=np.float64)
#for i,j in itertools.combinations(range(0, samples.shape[1]), 2):
#    JD[i,j] = jaccard_distance(samples[:,i], samples[:,j])
#    JD[j,i] = JD[i,j]
#JD[np.diag_indices(JD.shape[0])] = 1.0 
#
## Jensen-Shannon divergence
#JSD = np.zeros([samples.shape[1], samples.shape[1]], dtype=np.float64)
#for i,j in itertools.combinations(range(0, samples.shape[1]), 2):
#    JSD[i,j] = scipy.spatial.distance.jensenshannon(samples[:,i], samples[:,j])
#    JSD[j,i] = JSD[i,j]
#JSD[np.isnan(JSD)] = 0
#JSD[np.diag_indices(JD.shape[0])] = 1.0 
#
#dir_path = "/home/pokepup/DTU_Subjects/MSc_thesis/data/case_study/1_150/"
#save_matrix_tsv(JD, sample_ids, dir_path + "Jaccard_dist")
#save_matrix_tsv(JSD, sample_ids, dir_path + "JSD_dist")


#path_case_data = "/home/pokepup/DTU_Subjects/MSc_thesis/data/case_study/1_150/"
#
#metadata_df = pd.read_csv(path_case_data + "metadata.tsv", sep="\t", usecols=["org_index", "health_status"])
#
#Unifrac_df = pd.read_csv(path_case_data + "GUniFrac_alpha_one_Distance.tsv", sep="\t", header=0, index_col=0)
#Braycurtis_df = pd.read_csv(path_case_data + "Bray_Distance.tsv", sep="\t", header=0, index_col=0)
#CSCS_df = pd.read_csv(path_case_data + "CSCS_distances.tsv", sep=",", header=0, index_col=0)
#Jaccard_df = pd.read_csv(path_case_data + "Jaccard_dist.tsv", sep="\t", header=0)
#JSD_df = pd.read_csv(path_case_data + "JSD_dist.tsv", sep="\t", header=0)
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
#W_cscs, _, _, cscs_it, cscs_weight = optimization(cscs_u)
#W_Unifrac, _, _, Unifrac_it, Unifrac_weight = optimization(Unifrac_u)
#W_Bray, _, _, Bray_it, Bray_weight = optimization(Bray_u)
#W_Jaccard, _, _, JD_it, JD_weight = optimization(Jaccard_u)
#W_JSD, _, _, JSD_it, JSD_weight = optimization(JSD_u)
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
#weights = [cscs_weight, Unifrac_weight, Bray_weight, JD_weight, JSD_weight]
#iters = [cscs_it, Unifrac_it, Bray_it, JD_it, JSD_it]
#
#multi_stats(data=data_u, titles=titles_u, filename="../Case_study_unweighted", plabel=groups)
#multi_stats(data=data_w, titles=titles_w, filename="../Case_study_weighted", plabel=groups)
#multi_heatmaps(weights, titles_w, "../weights_case_study", iters)