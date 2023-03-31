import scipy.sparse as sparse
import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools, os, mkl, pickle, sys
import skbio
import seaborn as sns
import multiprocessing as mp
import igraph as ig
from numba import njit
import gc

#---------------------------------------------------------------------------------------------------------------------#
# Functions under development
#---------------------------------------------------------------------------------------------------------------------#

def PCOA(sparse_matrix):
    # Compute principial coordinated from sparse matrix
    #n = sparse_matrix.shape[0]
    #centered_matrix = np.eye(n) - np.ones((n, n))/n
    #X = -(1/n) * centered_matrix @ sparse_matrix @ centered_matrix
    #eigvals, eigvecs = sparse.linalg.eigs(X, k=n)
    #coordinates = eigvecs @ np.diag(np.sqrt(eigvals))
    symmetric = sparse.csr_matrix.dot(sparse_matrix, sparse_matrix.T) / 2
    symmetric.setdiag(1)
    dissimilarity = skbio.stats.distance.DissimilarityMatrix(1-symmetric.toarray())
    coordinates = skbio.stats.ordination.pcoa(dissimilarity)

    # plotting principial coordinated
    #plt.scatter(coordinates[0], coordinates[1], s=5)
    #Total = sum(np.square(np.real(eigvals)))
    #PC1 = round((np.square(np.real(eigvals[0]))/Total)*100, 2) 
    #PC2 = round((np.square(np.real(eigvals[1]))/Total)*100, 2) 
    #plt.xlabel(f"PC1 ({PC1}%)")
    #plt.ylabel(f"PC2 ({PC2}%)")
    #plt.title(f"PCoA of before optimization")
    plt.scatter(coordinates.samples['PC1'], coordinates.samples['PC2'], s=5)
    Total = sum(np.square(np.real(coordinates.eigvals)))
    PC1 = round((np.square(np.real(coordinates.eigvals[0]))/Total)*100, 2) 
    PC2 = round((np.square(np.real(coordinates.eigvals[1]))/Total)*100, 2) 
    plt.xlabel(f"PC1 ({PC1}%)")
    plt.ylabel(f"PC2 ({PC2}%)")
    plt.title(f"PCoA of before optimization") 
    #plt.savefig("test_1.png", format='png')
    return plt.show()

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

def heatmap_W(M, title):
    p1 = sns.heatmap(M)
    p1.set(xlabel=f"{title}", ylabel="samples")
    p1.set(title="Weights per iteration")
    plt.savefig(f"../heatmap_{title}.png", format="png")
    # important to prevent overlap between seaborn and matplotlib
    plt.clf()

def igraph_label(matrix, label):
    ## graph of matrix
    edges_samples = [(i, j) for i in range(matrix.shape[0]) for j in range(matrix.shape[1])]
    g = ig.Graph.Adjacency(matrix, edges=edges_samples).as_undirected()
    # communities grouped by dominant eigenvectors
    communities = g.community_multilevel()
    
    # plotting igraph
    pal = ig.drawing.colors.ClusterColoringPalette(len(communities))
    g.vs["color"] = pal.get_many(communities.membership)
    ig.plot(g, target="../communities_cscs.png", vertex_label = label)
    plt.clf()

def pca(X):
    mean = X.mean(axis=0) 
    center = X - mean 
    _, stds, pcs = np.linalg.svd(center/np.sqrt(X.shape[0])) 
    return stds**2, pcs

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

    cscs_w = np.zeros([samples.shape[1], samples.shape[1]])
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
        cscs_w[res[0],res[1]] = res[2]
        cscs_w[res[1],res[0]] = cscs_w[res[0],res[1]]

    # Tell child processes to stop
    for i in range(NUMBER_OF_PROCESSES):
        task_queue.put(None)

    cscs_w[np.diag_indices(cscs_w.shape[0])] = 1 

    return cscs_w

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
# Set up multiple samples with X-attributes
# set Beta to be 0, 1 or scalar
# compare weights, set at scaled factor
# compare different alphas
# validate labeled communities before and after optimization
# perform statistics with R-squared and PermANOVA

np.random.seed(100)

def get_uniform(n_samples=10, n_features=2, Beta_switch=[1,1]):
    # defines X attributes
    x1 = np.random.uniform(low=0.0, high=1.0, size=(n_samples,))
    x2 = np.random.uniform(low=0.0, high=1.0, size=(n_samples,))

    # np.newaxis increases number of dimensions to allow broadcasting
    X = np.concatenate((x1[:, np.newaxis], x2[:, np.newaxis]), axis=1)

    if n_features > 2:
        X = np.random.choice([x1, x2], size=(n_samples, n_features-2))
        X = np.concatenate((x1[:, np.newaxis], x2[:, np.newaxis], X), axis=1)
    
    # Switches off X-attributes
    Beta = np.ones((n_samples, n_features), dtype=int)
    for i, col in enumerate(Beta_switch):
        if col == 0:
            Beta[:, i] = np.zeros((1, 1), dtype=int)

    # computes linear model for n samples
    linear_eq = np.sum(Beta * X, axis=1)
    return linear_eq


S1 = get_uniform(Beta_switch=[1,0])
S2 = get_uniform(Beta_switch=[0,1])
S3 = get_uniform(Beta_switch=[1,0])
S4 = get_uniform(Beta_switch=[0,1])

samples = np.concatenate((S1[:, np.newaxis], S2[:, np.newaxis], \
    S3[:, np.newaxis], S4[:, np.newaxis]), axis=1)
css = np.cov(samples)

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

# Bray curtis
BC = np.zeros([samples.shape[1], samples.shape[1]], dtype=np.float64)
for i,j in itertools.combinations(range(0, samples.shape[1]), 2):
    BC[i,j] = scipy.spatial.distance.braycurtis(samples[:,i], samples[:,j])
    BC[j,i] = BC[i,j]
BC = 1-BC
BC[np.diag_indices(BC.shape[0])] = 1

## Unifrac
#otu_ids = ['OTU{}'.format(i) for i in range(samples.shape[0])]
#samples_dist = skbio.DistanceMatrix.from_iterable([samples, samples])
#print(samples_dist)
#tree_root = skbio.tree.nj(samples_dist)
#
#unifrac_distance = skbio.diversity.beta.unweighted_unifrac(samples_dist[:,0], samples_dist[:,1], otu_ids=samples_dist.ids, tree=tree_root)
#dis
#print(unifrac_distance)

# Jaccard distance
JD = np.zeros([samples.shape[1], samples.shape[1]], dtype=np.float64)
for i,j in itertools.combinations(range(0, samples.shape[1]), 2):
    JD[i,j] = jaccard_distance(samples[:,i], samples[:,j])
    JD[j,i] = JD[i,j]
JD[np.diag_indices(JD.shape[0])] = 1 

# Jensen-Shannon divergence
JSD = np.zeros([samples.shape[1], samples.shape[1]], dtype=np.float64)
for i,j in itertools.combinations(range(0, samples.shape[1]), 2):
    JSD[i,j] = scipy.spatial.distance.jensenshannon(samples[:,i], samples[:,j])
    JSD[j,i] = JSD[i,j]
JSD[np.diag_indices(JD.shape[0])] = 1 

# Euclidean distance
Euc = np.zeros([samples.shape[1], samples.shape[1]], dtype=np.float64)
for i,j in itertools.combinations(range(0, samples.shape[1]), 2):
    Euc[i,j] = scipy.spatial.distance.euclidean(samples[:,i], samples[:,j])
    Euc[j,i] = Euc[i,j]
Euc[np.diag_indices(Euc.shape[0])] = 1 

#---------------------------------------------------------------------------------------------------------------------#
# CSCS Parallelization
#---------------------------------------------------------------------------------------------------------------------#

# Parallel C interface optimization
os.environ["USE_INTEL_MKL"] = "1"
mkl.set_num_threads(4)

cscs_u = Parallelize(cscs, samples, css)
cscs_u.astype(np.float64)

#M = cscs_u
#for impl in [np.linalg.eig, np.linalg.eigh, scipy.linalg.eig, scipy.linalg.eigh]:
#    w, v = impl(M)
#    print(np.sort(w))
#    reconstructed = np.dot(v * w, v.conj().T)
#    print("Allclose:", np.allclose(reconstructed, M), '\n')
    

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

    #w = np.random.beta(alpha, beta, size=X.shape[0])
    #W = np.triu(w, 1) + np.triu(w, 1).T
    W = np.full((X.shape[0], X.shape[0]), 1/X.shape[0], dtype=np.float64)
    #W.astype(np.float64)
    return W

@njit
def grad_function(X, W):
    M = X * W
    _, eigval, eigvec = np.linalg.svd(M)

    # gradient & variance explained
    grad = eigvec * X * eigvec.T 
    e_sum = np.sum(eigval)
    var_explained = np.sum(eigval[:2]) / e_sum

    return grad, var_explained, eigval

@njit
def add_column(m1, m2):
    return np.column_stack((m1, m2))

def Bare_bone(X, alpha=0.1, num_iters=100, epss = np.finfo(np.float64).eps):
    W = initialize_theta(X)
    df = pd.DataFrame(columns=["iter", "variance_explained", "abs_diff", "eigval1", "eigval2"])

    best_W, iter = np.ones((X.shape[0], X.shape[0]), dtype=np.float64), 0

    # Computes first variance
    # If optimization cannot succeed, returns original
    _, s, _ = np.linalg.svd(X)
    e_sum = np.sum(s)
    best_var = np.sum(s[:2]) / e_sum
    prev_var = best_var
    df.loc[0] = [0, np.real(best_var), 0, np.real(s[0]), np.real(s[1])]

    Weight_stack = W[:,0]
    for i in range(num_iters):
        get_grad, current_var, eigval = grad_function(X, W)

        abs_diff = np.sum(np.absolute(current_var - prev_var))
        # epss is based on the machine precision of np.float64 64
        df.loc[i+1] = [i+1, np.real(current_var), np.real(abs_diff), np.real(eigval[0]), np.real(eigval[1])]
        print(f"variance explained: {current_var}\t eigval 1: {eigval[0]}\t eigval 2: {eigval[1]}\t sum eigvals: {np.sum(eigval)}")
        
        if abs_diff < epss:
            break

        if current_var > best_var:
            best_var = current_var
            best_W = W
            iter = i
        
        W += (alpha * get_grad)
        W = np.clip(W, 0, 1)
        prev_var = current_var
        Weight_stack = add_column(Weight_stack, W[:,0])
    
    return df, best_W, iter, Weight_stack




"""
a = 0.01
df_emse3, W01, i_W01, weights_fixed_alpha = Bare_bone(cscs_u, alpha=a)

#---------------------------------------------------------------------------------------------------------------------#
# Visualizing simulated data
#---------------------------------------------------------------------------------------------------------------------#

fig, (ax0, ax1, ax2, ax3) = plt.subplots(4)
fig.set_size_inches(15, 10)
ax0.plot(df_emse3["iter"], df_emse3["variance_explained"], label=f"a= {a}")
ax0.axvline(x=i_W01, ls='--', c="red", label=f"a = {a}")
ax0.set_xlabel(f"iterations")
ax0.set_title("Variance explained")
ax1.plot(df_emse3["iter"], df_emse3["abs_diff"], label=f"a = {a}")
ax1.set_xlabel(f"iterations")
ax1.set_title("Absolute difference")
ax2.plot(df_emse3["iter"], df_emse3["eigval1"], label=f"a = {a}")
ax2.set_xlabel(f"iterations")
ax2.set_title("Eigenvalue 1")
ax3.plot(df_emse3["iter"], df_emse3["eigval2"], label=f"a = {a}")
ax3.set_xlabel(f"iterations")
ax3.set_title("Eigenvalue 2")
ax1.legend()
fig.tight_layout(pad=2.0)
fig.savefig("../cscsw_simulated_par.png", format='png')
plt.clf()

var_u, pcs_u = pca(cscs_u)
var_W01, pcs_W01 = pca(cscs_u*W01)

### subplot 2 ###
# font size
font_size = 15
plt.rcParams.update({"font.size": 12})
pca_color = sns.color_palette(None, cscs_u.shape[1])
fig0, (ax1, ax2) = plt.subplots(2)
fig0.set_size_inches(15, 10)
# unweigthed cscs
for i in range(cscs_u.shape[1]):
    ax1.scatter(pcs_u[0][i], pcs_u[1][i], color=pca_color[i], s=10, label=f"{i+1}")
    ax1.annotate(f"{str(i+1)}", (pcs_u[0][i], pcs_u[1][i]))
ax1.set_xlabel(f"PC1: {round(var_u[0]/np.sum(var_u)*100,2)}%")
ax1.set_ylabel(f"PC2: {round(var_u[1]/np.sum(var_u)*100,2)}%")
ax1.set_title(f"Unweighted CSCS")
ax1.legend(loc='center left', bbox_to_anchor=(1, 0.7))

# Weighted cscs fixed alpha
for i in range(cscs_u.shape[1]):
    ax2.scatter(pcs_W01[0][i], pcs_W01[1][i], color=pca_color[i], s=10, label=f"{i+1}")
    ax2.annotate(f"{str(i+1)}", (pcs_W01[0][i], pcs_W01[1][i]))
ax2.set_xlabel(f"PC1: {round(var_W01[0]/np.sum(var_W01)*100,2)}%")
ax2.set_ylabel(f"PC2: {round(var_W01[1]/np.sum(var_W01)*100,2)}%")
ax2.set_title(f"Weighted CSCS with alpha = {a}")
fig0.tight_layout(pad=2.0)
fig0.savefig("../cscsw_PCA.png", format='png')
plt.clf()

#heatmap_W(weights_fixed_alpha, "fixed_alpha")

#M01 = cscs_u * W01
#
#contours(M01, title="cscs_alpha01")
#gradient_plot_3D(M01, title="cscs_alpha01")
"""