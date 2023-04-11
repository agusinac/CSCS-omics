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
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA

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
# Set up multiple samples with X-attributes                                         DONE
# set Beta to be 0, 1 or scalar                                                     DONE
# compare weights, set at scaled factor                                             DONE
# compare different alphas
# validate labeled communities before and after optimization
# perform statistics with R-squared and PermANOVA
# Implement the gradient to check in both directions: Addition or substraction

np.random.seed(100)

def get_uniform(Beta_switch, n_samples=5, n_features=10):
    # defines X attributes
    X = np.random.uniform(low=0.0, high=1.0, size=(n_features, n_samples))  
    Beta = Beta_switch
    ## Switches off X-attributes
    for i in range(n_features-1):
        Beta = np.vstack((Beta, Beta_switch))
    # computes linear model for n samples
    linear_eq = Beta * X
    return linear_eq

label = [0, 1, 1.5, 2, 4, 10]
samples = get_uniform(Beta_switch=label, n_samples=len(label))
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
JSD[np.isnan(JSD)] = 0
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
    W = np.full((X.shape[0], X.shape[0]), 1/10, dtype=np.float64)
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
    df = pd.DataFrame(columns=["iter", "variance_explained", "eigval1", "eigval2"])

    best_W, iter = np.ones((X.shape[0], X.shape[0]), dtype=np.float64), 0

    # Computes first variance
    # If optimization cannot succeed, returns original
    _, s, _ = np.linalg.svd(X)
    e_sum = np.sum(s)
    best_var = np.sum(s[:2]) / e_sum
    prev_var = best_var
    df.loc[0] = [0, np.real(best_var), np.real(s[0]), np.real(s[1])]

    Weight_stack = W[:,0]
    for i in range(num_iters):
        get_grad, current_var, eigval = grad_function(X, W)

        abs_diff = np.sum(np.absolute(current_var - prev_var))
        # epss is based on the machine precision of np.float64 64
        df.loc[i+1] = [i+1, np.real(current_var), np.real(eigval[0]), np.real(eigval[1])]
        #print(f"variance explained: {current_var}\t eigval 1: {eigval[0]}\t eigval 2: {eigval[1]}\t sum eigvals: {np.sum(eigval)}")
        
        if abs_diff < epss:
            break

        if current_var > best_var:
            best_var = current_var
            best_W = W
            iter = i+1
        
        W += (alpha * get_grad)
        W = np.clip(W, 0, 1)
        prev_var = current_var
        Weight_stack = add_column(Weight_stack, W[:,0])
    
    return df, best_W, iter, Weight_stack


#igraph_label(cscs_u, label=label)

#---------------------------------------------------------------------------------------------------------------------#
# Gradient descent of distance metrics
#---------------------------------------------------------------------------------------------------------------------#

a = 0.01
df_cscs, W_cscs, it_W_cscs, Weights_cscs = Bare_bone(cscs_u, alpha=a)
df_BC, W_BC, it_W_BC, Weights_BC = Bare_bone(BC, alpha=a)
df_JD, W_JD, it_W_JD, Weights_JD = Bare_bone(JD, alpha=a)
df_JSD, W_JSD, it_W_JSD, Weights_JSD = Bare_bone(JSD, alpha=a)
df_Euc, W_Euc, it_W_Euc, Weights_Euc = Bare_bone(Euc, alpha=a)

data_u = [cscs_u, BC, JD, JSD, Euc]
data_w = [cscs_u*W_cscs, BC*W_BC, JD*W_JD, JSD*W_JSD, Euc*W_Euc]

titles = ["CSCS", "Bray-curtis", "Jaccard distance", "Jensen-Shannon Divergence", "Euclidean distance"]


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

#GD_parameters(data=df_cscs, title="cscs" , it_W=it_W_cscs, a=a)

def multi_PCoA(data, titles, filename):
    plt.figure(figsize=(15, 12))
    plt.subplots_adjust(hspace=0.2)
    plt.rcParams.update({'font.size': 12})
    for n, id in enumerate(data):
        ax = plt.subplot(1, len(data), n + 1)
        # PCA decomposition
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(id)
        X_pca_inverse = pca.inverse_transform(X_pca)
        var = pca.explained_variance_ratio_
        pcs = pca.components_

        pca_color = sns.color_palette(None, id.shape[1])
        for i in range(id.shape[1]):
            ax.scatter(pcs[0][i], pcs[1][i], color=pca_color[i], s=10, label=f"{i+1}")
            ax.annotate(f"{str(i+1)}", (pcs[0][i], pcs[1][i]))
        ax.text(0,0,f"R-squared = {round(r2_score(id, X_pca_inverse),3)}", fontsize=12)
        ax.set_xlabel(f"PC1: {round(var[0]*100,2)}%")
        ax.set_ylabel(f"PC2: {round(var[1]*100,2)}%")
        ax.set_title(f"{titles[n]}")
        ax.get_legend()
    plt.tight_layout()
    plt.savefig(f"../{filename}_multi_PCAs.png", format='png')
    plt.clf()

multi_PCoA(data=data_u, titles=titles, filename="unweighted")
#multi_PCoA(data=data_w, titles=titles, filename="weighted")

#heatmap_W(weights_fixed_alpha, "fixed_alpha")