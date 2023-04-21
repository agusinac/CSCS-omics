import scipy.sparse as sparse
import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import itertools, os, mkl, pickle, sys
import skbio
import seaborn as sns
import multiprocessing as mp
import igraph as ig
from numba import njit
import gc
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances

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

def multi_heatmaps(data, titles, filename, ncols=2):
    plt.figure(figsize=(20, 15))
    plt.subplots_adjust(hspace=0.2)
    plt.rcParams.update({'font.size': 12})

    for n, id in enumerate(data):
        ax = plt.subplot(ncols, len(data) // ncols + (len(data) % ncols > 0), n + 1)
        sns.heatmap(id, ax=ax)
        ax.set_title(f"{titles[n]}")
        ax.set_xlabel("Iterations")
        ax.set_ylabel("samples")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"../{filename}_multi_heatmaps.png", format='png')
    plt.clf()

def multi_stats(data, titles, filename, plabel, ncols=2):
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
        if n == 0:
            dist = id / id[0,0]
            dist = 1 - dist
        else:
            dist = 1 - id

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
    plt.clf()

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
    plt.clf()
    return communities.membership

def data_dump(data, title):
    file = open(f"../{title}", "wb")
    pickle.dump(data, file)
    file.close()

def cscs(A, B, css):
    cssab = A * B.T * css
    cssaa = A * A.T * css
    cssbb = B * B.T * css
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

#---------------------------------------------------------------------------------------------------------------------#
# Simulated data
#---------------------------------------------------------------------------------------------------------------------#

# TO DO:
# case study sponges (optional)                                                                         IN PROGRESS
# plot permanova vs explained variance against sparse density levels and features/samples ratio         DONE
# Create a benchmark loop to test different sparse density variables for CSCSw                          IN PROGRESS
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

    return linear_eq.T, cosine_similarity(X.T) ,labels


#---------------------------------------------------------------------------------------------------------------------#
# Case study data Sponges
#---------------------------------------------------------------------------------------------------------------------#

#import dendropy
#
#tree = dendropy.Tree.get(path="/home/pokepup/DTU_Subjects/MSc_thesis/data/case_study/raw_data/tree_relabelled.tre", schema='newick')
#pdm = tree.phylogenetic_distance_matrix()
#pdm.to_csv("/home/pokepup/DTU_Subjects/MSc_thesis/data/case_study/raw_data/tree_distances.csv")


#---------------------------------------------------------------------------------------------------------------------#
# Comparison to other distance metrics
#---------------------------------------------------------------------------------------------------------------------#

def jaccard_distance(A, B):
    #Find symmetric difference of two sets
    nominator = np.setdiff1d(A, B)

    #Find union of two sets
    denominator = np.union1d(A, B)

    #Take the ratio of sizes
    distance = len(nominator)/len(denominator)
    
    return distance

#---------------------------------------------------------------------------------------------------------------------#
# CSCS Parallelization
#---------------------------------------------------------------------------------------------------------------------#

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

    w = np.random.beta(alpha, beta, size=X.shape[0])
    W = np.triu(w, 1) + np.triu(w, 1).T
    #W = np.full((X.shape[0], X.shape[0]), 1/10, dtype=np.float64)
    W.astype(np.float64)
    return W

@njit
def grad_function(X, W):
    M = X * W
    _, eigval, eigvec = np.linalg.svd(M)

    # gradient & variance explained
    grad = X * np.dot(eigvec, eigvec.T)
    e_sum = np.sum(eigval)
    var_explained = np.sum(eigval[:2]) / e_sum

    return grad, var_explained, eigval

@njit
def add_column(m1, m2):
    return np.column_stack((m1, m2))

def optimization(X, alpha=0.1, num_iters=100, epss=np.finfo(np.float64).eps):
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

    #Weight_stack = W[:,0]
    for i in range(num_iters):
        get_grad, current_var, eigval = grad_function(X, W)

        abs_diff = np.absolute(current_var - prev_var)
        # epss is based on the machine precision of np.float64 64
        #df.loc[i+1] = [i+1, np.real(current_var), np.real(eigval[0]), np.real(eigval[1])]
        
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
        #Weight_stack = add_column(Weight_stack, W[:,0])

    return best_W, best_var, original_var

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
# Assessing sparse density effect on Permanova & Variance explained
#---------------------------------------------------------------------------------------------------------------------#

import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 

num_iters = 100
sparse_densities = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
features = 1000
sample_size = [25, 50, 100, 150]


for s in range(num_iters):
    print(f"Starting duplicate {s+1} out of {num_iters}")
    df = pd.DataFrame(columns=["duplicates", "sparse_level", "n_features", "metric_ID", "var_explained", "F_stat", "p_val"])
    for swab, sparse_d in itertools.product(sample_size, sparse_densities):
        # simulated data
        test = scipy.sparse.random(4, features, density=sparse_d, random_state=np.random.default_rng(), data_rvs=scipy.stats.poisson(50, loc=10).rvs)
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
        
        for n, id in enumerate(data_u):
            dist = 1 - id
            np.fill_diagonal(dist, 0.0)
            dist = skbio.DistanceMatrix(dist)
            result = skbio.stats.distance.permanova(dist, groups, permutations=9999)
            row = pd.DataFrame([s+1,sparse_d,features,title_u[n],var_u[n],result["test statistic"],result["p-value"]])
            df = pd.concat([df, row], ignore_index=True)
        
        for n, id in enumerate(data_w):
            id[np.isnan(id)] = 0.0
            if n == 0:
                dist = id / id[0,0]
                dist = 1 - dist
            else:
                dist = 1 - id

            np.fill_diagonal(dist, 0.0)
            dist = skbio.DistanceMatrix(dist)
            result = skbio.stats.distance.permanova(dist, groups, permutations=9999)
            row = pd.DataFrame([s+1,sparse_d,features,title_w[n],var_w[n],result["test statistic"],result["p-value"]])
            df = pd.concat([df, row], ignore_index=True)
    if s == 0:
        df.to_csv("/home/pokepup/DTU_Subjects/MSc_thesis/scripts/python/Benchmark_stats.csv", mode='a', header=True, index=False)
    df.to_csv("/home/pokepup/DTU_Subjects/MSc_thesis/scripts/python/Benchmark_stats.csv", mode='a', header=False, index=False)

def benchmark_plot(filename, title):
    df = pd.read_csv(filename, sep="\t")

    # Group by n_samples and sort by sparse_level
    groups = df.groupby('n_samples')
    sorted_groups = [group[1].sort_values(by='sparse_level') for group in groups]
    # Set up the subplots

    fig = plt.figure(figsize=(20, 10))
    gs = gridspec.GridSpec(1, len(sorted_groups), figure=fig, hspace=0.3)
    plt.subplots_adjust(top=0.95, bottom=0.05)
    # Loop over the groups and plot each one
    for i, group in enumerate(sorted_groups):
        ax = fig.add_subplot(gs[i])
        ax.set_ylabel('var_explained', color='tab:blue')
        ax.plot(group['sparse_level'], group['var_explained'], color='tab:blue', label="variance explained")
        ax.tick_params(axis='y', labelcolor='tab:blue')
        ax2 = ax.twinx()
        ax2.set_ylabel('F-statistic', color='tab:orange')
        ax2.plot(group['sparse_level'], group['F-statistic'], color='tab:orange', label="F-statistic")
        ax2.tick_params(axis='y', labelcolor='tab:orange')
        # Add asterisks for p-values below or equal to 0.0001
        for index, row in group.iterrows():
            if row["p-value"] <= 0.0001:
                ax2.scatter(row["sparse_level"], row["F-statistic"], marker='*', s=50, color='black')

        ax.set_title('n_samples = {}'.format(group.iloc[0]['n_samples']), loc='center')
        ax.set_xlabel('sparse density')
        ax.set_xticks(range(10, 100, 10))

    plt.savefig(f"../{title}_Benchmark_sparse_sample.png", format='png')
    plt.clf()

#---------------------------------------------------------------------------------------------------------------------#
# Case study: Sponges
#---------------------------------------------------------------------------------------------------------------------#

from Bio import Phylo
"""
with open("/home/pokepup/DTU_Subjects/MSc_thesis/data/case_study/raw_data/rep-seqs.qza/8928d9c3-08e2-499f-9fa7-87958707403d/data/dna-sequences.fasta", "r") as infile:
    ids = [line[:-1].strip(">") for line in infile if line.startswith(">")]

tree = Phylo.read("/home/pokepup/DTU_Subjects/MSc_thesis/data/case_study/raw_data/tree_relabelled.tre", "newick")

id_to_clade = {clade.name: clade for clade in tree.get_terminals() if clade.name in ids}

matrix = scipy.sparse.dok_matrix((len(id_to_clade), len(id_to_clade)), dtype=float)
for i, x in enumerate(id_to_clade.values()):
    for j, y in enumerate(id_to_clade.values()):
        distance = tree.distance(x, y)
        matrix[i, j] = distance
        matrix[j, i] = matrix[i, j] 

df = pd.DataFrame(matrix.toarray(), columns=id_to_clade.keys(), index=id_to_clade.keys())
df.to_csv("distance_matrix.csv")
"""