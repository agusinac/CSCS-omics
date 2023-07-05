import numpy as np
import scipy
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import itertools, os, pickle, gc
import skbio
import seaborn as sns
import multiprocessing as mp
#import igraph as ig
#from numba import njit
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.patches as mpatches

# Enables OPENBLAS environment in scipy and numpy
os.environ["OPENBLAS_NUM_THREADS"] = "1"

# Sets seed
np.random.seed(100)


def clean_matrix(matrix):
    """
    Cleans the matrix from (-)infinite and NaN values.

    Parameters:
        - matrix (numpy.ndarray): 2D array

    Returns:
        matrix (numpy.ndarray): 2D array
    """
    matrix[np.isnan(matrix)] = 0
    matrix[matrix == -np.inf] = 0
    matrix[matrix == np.inf] = 1
    return matrix

def multi_heatmaps(data, titles, filename, vline = None, ncols=3):
    """
    Generates a figure with multiple heatmap subplots of the weights per iteration.

    Parameters:
        - data [numpy.ndarray]: Input list of 2D arrays.
        - titles [str]: List of titles for each subplot
        - filename (str): Name for the returned png image file
        - y_labels [str]: List of strings to adjust the y-axis
        - vline (int): Integer number to create a vertical dotted line
        - ncols (int): number of columns for the subplots, default is 3

    Returns:
        Saves figure with multi heatmap subplots as png image
    """
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
        #ax.set_yticks(range(len(y_labels)))
        #ax.set_yticklabels(y_labels)
        if vline is not None:
            ax.axvline(x=vline[n], linestyle=':', color='grey')

    ax.legend()
    plt.tight_layout()
    plt.savefig(f"../{filename}_multi_heatmaps.png", format='png')
    plt.close()

def assign_random_colors(sorted_labels):
    """
    Generates colors based on the order of the list indices to a list of string names

    Parameters:
        - sorted_labels [int]: List of indices

    Returns:
        replaced_list [str]: List of strings
        color_mapping [color]: List of colors from the seaborn.color_palette
    """
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

def multi_stats(data, titles, filename, sorted_labels, ncols=4):
    """
    Generates a figure with multiple PCoA's subplots from matrices
    Final plot is the PERMANOVA statistics.

    Parameters:
        - data [numpy.ndarray]: Input list of 2D arrays.
        - titles [str]: List of titles for each subplot
        - filename (str): Name for the returned png image file
        - sorted_labels [int]: List of indices
        - ncols (int): number of columns for the subplots, default is 4 

    Returns:
        Saves figure as png image
    """
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
        id = clean_matrix(id)
        pca = PCA(n_components=2)
        pca.fit_transform(id)
        var = pca.explained_variance_ratio_
        pcs = pca.components_
    
        # Permanova
        dist = id / id[0,0]
        dist = 1 - dist
        dist = clean_matrix(dist)
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

def GD_parameters(data, title, it_W, a=0.01):
    """
    Creates a line graph of the variance explained and first two eigenvalues per iteration

    Parameters:
        - data (numpy.ndarray): Input 2D array
        - title (str): Filename 
        - it_W (int): Iteration position where the optimal maximum is found
        - a (int): step-size of alpha, default = 0.01

    Returns:
        Saves graph as png image
    """
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
    plt.close()

def data_dump(data, title):
    """
    Generates a bytes compressed file of chosen matrix

    Parameters:
        - data (numpy.ndarray): Input 2D array
        - title (str): Filename 

    Returns:
        Saves as compressed pickle data
    """
    file = open(f"../{title}", "wb")
    pickle.dump(data, file)
    file.close()

def CSCS_W(A, B, css):
    """
    Model 3: Weights the sample A and sample B 1D arrays separately.
    Optimizes sample A weights against sample B.

    Parameters:
        - A (numpy.ndarray): 1D array with integers
        - B (numpy.ndarray): 1D array with integers
        - CSS (numpy.ndarray): similarity matrix with 1's on diagonal

    Returns:
        CSCS result
    """
    cssab = np.multiply(css, np.multiply(A, B.T))
    cssaa = clean_matrix(cssab)

    # weight intialization
    sample_mean = np.mean(A)
    sample_var = np.var(A, ddof=1)
    alpha = sample_mean * (sample_mean * (1 - sample_mean) / sample_var - 1)
    if alpha < 0:
        alpha *= -1
    beta = (1 - sample_mean) * (sample_mean * (1 - sample_mean) / sample_var - 1)
    if beta < 0:
        beta *= -1

    w = np.random.beta(alpha, beta, size=A.shape[0])
    #W = np.triu(w, 1) + np.triu(w, 1).T
    #W = clean_matrix(W)
    w[np.isnan(w)] = 0
    W = np.clip(w, 0, 1)
    W.astype(np.float64)

    # Place holders
    B_w = np.multiply(B, W.T)
    cssbb = np.multiply(css, np.multiply(B_w, B_w.T))
    cssbb = clean_matrix(cssbb)
    best_W, iter = np.ones((A.shape[0]), dtype=np.float64), 0
    # Computes first variance
    # If optimization cannot succeed, returns original
    _, s, _ = np.linalg.svd(cssab)
    epss = np.finfo(np.float64).eps
    e_sum = np.sum(s)
    best_var = np.sum(s[:2]) / e_sum
    original_var = best_var
    prev_var = best_var

    for i in range(100):
        # svd of pair
        M = np.multiply(css, np.multiply(np.multiply(A, W.T), B_w.T))
        M = clean_matrix(M)
        u, A_s, v = np.linalg.svd(M)
        _, B_s, _ = np.linalg.svd(cssbb)

        grad = np.multiply(np.subtract(A, np.multiply(B_s, B_w)), np.multiply(u[:,0], u[:,0].T))
       
        e_sum = np.sum(A_s)
        if e_sum == 0:
            current_var = 0
        else:
            current_var = np.sum(A_s[:2]) / e_sum

        if np.absolute(current_var - prev_var) < epss:
            break

        if current_var > best_var:
            best_var = current_var
            best_W = W
            #iter = i+1
        
        W += (0.1 * grad)   
        W = clean_matrix(W)     
        W = np.clip(W, 0, 1)
        prev_var = current_var
    
    A_w = np.multiply(A, best_W)
    cssab = np.multiply(css, np.multiply(A_w, B_w.T))
    cssaa = np.multiply(css, np.multiply(A_w, A_w.T))
    scaler = max(np.sum(cssaa), np.sum(cssbb))
    if scaler == 0:
        result = 0
    else:
        result = np.sum(cssab) / scaler
    return result

def CSCS(A, B, css):
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
    cssab = np.multiply(css, np.multiply(A, B.T))
    cssaa = np.multiply(css, np.multiply(A, A.T))
    cssbb = np.multiply(css, np.multiply(B, B.T))
    scaler = max(np.sum(cssaa), np.sum(cssbb))
    if scaler == 0:
        result = 0
    else:
        result = np.sum(cssab) / scaler
    return result

def worker(input, output):
    """
    Calls function to distribute the task and collects the result

    Parameters:
        - input (multiprocessing.Queue): Queue of tasks to be processed
        - output (multiprocessing.Queue): Queue of tasks containing the results
        
    Returns:
        None
    """
    for func, A, B, index_a, index_b, css in iter(input.get, None):
        result = func(A, B, css)
        output.put([index_a, index_b, result])

def Parallelize(func, samples, css):
    """
    Distributes tasks to workers to assemble the CSCS matrix from sample abundances and feature similarities.

    Parameters:
        - func (str): Function to be called
        - samples (numpy.ndarray): Matrix with columns as samples and rows by feature order of CSS
        - CSS (numpy.ndarray): similarity matrix with 1's on diagonal

    Returns:
        CSCS matrix
    """
    NUMBER_OF_PROCESSES = mp.cpu_count()

    cscs_u = np.zeros([samples.shape[1], samples.shape[1]])
    TASKS = [(func, samples[:,i], samples[:,j], i, j, css) for i,j in itertools.combinations(range(0, samples.shape[1]), 2)]

    # Create queues
    task_queue = mp.Queue()
    done_queue = mp.Queue()    

    # Submit tasks
    for task in TASKS:
        task_queue.put(task)

    # Start worker processes
    for i in range(NUMBER_OF_PROCESSES):
        mp.Process(target=worker, args=(task_queue, done_queue)).start()

    # Get and print results
    for i in range(len(TASKS)):
        res = done_queue.get()
        cscs_u[res[0],res[1]] = res[2]
        cscs_u[res[1],res[0]] = res[2]

    # Tell child processes to stop
    for i in range(NUMBER_OF_PROCESSES):
        task_queue.put(None)

    np.fill_diagonal(cscs_u, 1)

    return cscs_u

def jaccard_distance(A, B):
    """
    Computes the Jaccard distance between two samples

    Parameters:
        - A (numpy.ndarray): 1D array with integers
        - B (numpy.ndarray): 1D array with integers

    Returns:
        jaccard distance
    """
    #Find symmetric difference of two sets
    nominator = np.setdiff1d(A, B)

    #Find union of two sets
    denominator = np.union1d(A, B)

    #Take the ratio of sizes
    distance = len(nominator)/len(denominator)
    
    return distance

def save_matrix_tsv(matrix, headers, filename):
    """
    Saves a matrix with headers as a tsv file

    Parameters:
        - matrix (numpy.ndarray): 2D array
        - headers [str]: List of strings
        - filename (str): string name

    Returns:
        Saves matrix as tsv file
    """
    with open(filename + ".tsv", 'w') as outfile:
        outfile.write("\t".join(headers) + "\n")
        np.savetxt(outfile, matrix, delimiter="\t")

def generate_data(signatures, n_samples=100, n_features=2):
    """
    Generates simulated data based on a linear model

    Formula:
        - f = beta * x
        Where:  f = sample
                beta = an integer
                x = a feature from uniform distribution

    Parameters:
        - signatures (numpy.ndarray): vector containing positive and zero integers
        - n_samples (int): Specifying the size of the columns, default is 100
        - n_features (int): Specifying the size of the rows, default is 2

    Returns:
        - norm_samples (numpy.ndarray): Samples with relative abundance
        - cosine_similarity(X.T) (numpy.ndarray): 2D array with 1's on diagonal, representing the CSS matrix
        - labels [int]: List of indices

    """
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
# Optimization algorithm
#---------------------------------------------------------------------------------------------------------------------#

def initialize_theta(X):
    """
    Samples weights from Beta distribution and returns as a matrix

    Parameters:
        - X (numpy.ndarray): symmetric 2D array, integers in range of 0 and 1

    Returns:
        - W (numpy.ndarray): symmetric 2D array, integers in range of 0 and 1
    """
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
    W = clean_matrix(W)
    W = np.clip(W, 0, 1)
    W.astype(np.float64)
    return W

def grad_function(X, W):
    """
    Computes the gradient based on the formula:

    dlambda = U * (dX) * U.T

    Eigenvectors (U) are computed from the weighted X

    Parameters:
        - X (numpy.ndarray): symmetric 2D array, integers in range of 0 and 1
        - W (numpy.ndarray): symmetric 2D array, integers in range of 0 and 1

    Returns:
        - grad (numpy.ndarray): 2D array
        - var_explained (int): Percentage explained on first two eigenvalues

    """
    M = X * W
    u, s, v = np.linalg.svd(M)

    # gradient & variance explained
    grad = X * np.multiply(u[:,:1], u[:,:1].T)
    e_sum = np.sum(s)
    if e_sum == 0:
        var_explained = 0
    else:
        var_explained = np.sum(s[:2]) / e_sum

    return grad, var_explained

def add_column(col1, col2):
    """
    Combines two columns into one

    Parameters:
        - col1 (numpy.ndarray): 1D array
        - col2 (numpy.ndarray): 1D array

    Returns:
        Array with two columns
    """
    return np.column_stack((col1, col2))

def theta_diff(matrix):
    """
    Computes the difference of elements for each column from a matrix

    Parameters:
        - matrix (numpy.ndarray): 2D array

    Returns:
        1D array with elements containing the sum of difference of each column
    """
    return np.sum(np.diff(matrix, axis=1), axis=1)

def optimization(X, alpha=0.1, num_iters=100, epss=np.finfo(np.float64).eps):
    """
    Performs gradient descent to find highest distance explained based on first two eigenvalues

    Parameters:
        - X (numpy.ndarray): 2D array
        - sorted_labels [int]: List of indices
        - alpha (int): step size of gradient, default is 0.1
        - num_iters (int): number of iterations, default is 1000
        - epss (float): Detection limit of numpy float 64 bit

    Returns:
        - best_W (numpy.ndarray): 2D array containing the summed weight difference
        - best_var (int): optimal distance explained found
        - original_var (int): original distance explained
        - iter (int): Iteration of found optimal distance explained
        - Weight_stack (numpy.ndarray): 2D array of optimal weights
    
    """
    X = clean_matrix(X)
    W = initialize_theta(X)
    best_W, iter = np.ones((X.shape[0], X.shape[0]), dtype=np.float64), 0
    # Computes first variance
    # If optimization cannot succeed, returns original
    _, s, _ = np.linalg.svd(X)
    
    e_sum = np.sum(s)
    best_var = np.sum(s[:2]) / e_sum
    original_var = best_var
    prev_var = best_var
    
    Weight_stack = theta_diff(W)

    for i in range(num_iters):
        get_grad, current_var = grad_function(X, W)
        
        # Early stopping
        if np.absolute(current_var - prev_var) < epss:
            break

        if current_var > best_var:
            best_var = current_var
            best_W = W
            iter = i+1
        
        W += (alpha * get_grad)        
        W = np.clip(W, 0, 1)
        prev_var = current_var

        Weight_stack = add_column(Weight_stack, theta_diff(W))

    return best_W , best_var, original_var, iter, Weight_stack

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
        df.to_csv("/home/pokepup/DTU_Subjects/MSc_thesis/results/Benchmark/Model_3/Benchmark_stimulated.csv", mode='a', header=True, index=False)
    df.to_csv("/home/pokepup/DTU_Subjects/MSc_thesis/results/Benchmark/Model_3/Benchmark_stimulated.csv", mode='a', header=False, index=False)
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
blast_file = file_path + "720sample.16S.otu.repTag.filter.fasta"

# groups based on "Sample.Time"
metadata = pd.read_csv(file_path + "metadata.csv", sep=",", header=0, usecols=["Sample.ID","Sample.Time"])

group_A, group_B = [], []
for i,j in zip(metadata["Sample.ID"], metadata["Sample.Time"]):
    if j == "Pre diet":
        group_A.append(i)
    elif j == "Termination":
        group_B.append(i)

# Separate OTU_table into two groups
OTU_table = pd.read_csv(file_path + "otu_table.csv", sep=",", header=0, index_col=0)
samples_ids = OTU_table.columns.tolist()
otu_ids = OTU_table.index.tolist()
samples = OTU_table.values
feature_ids = {str(id):it for it, id in enumerate(list(OTU_table.index))}
#samples_ids = {str(id):it for it, id in enumerate(list(OTU_table.columns))}

def construct_matrix(sparse_d, n_samples, samples_ids, group_A, group_B, OTU_table):
    """
    Down-samples on Empirical mice data to construct new data sets based on sample size

    Parameters:
        - sparse_d (float): Sparsity degree
        - n_samples (int): size of the columns
        - sample_ids (dict): Containing index as key and sample name as value
        - group_A (numpy.ndarray): 2D array of samples belonging to Pre Diet group
        - group_B (numpy.ndarray): 2D array of samples belonging to Termination group
        - OTU_table (pandas.dataframe): Columns by sample name and rows by feature order. Values consist out of integers

    Returns:
        - norm_samples (numpy.ndarray): Relative abundance 2D array
        - otu_idx [int]: List of feature indices

    """  
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


# parameters to test
sparse_densities = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0]
n_samples = [20]
num_iters = 1

for s in range(0, num_iters):
    print(f"Starting duplicate {s+1} out of {num_iters}")
    df = pd.DataFrame(columns=["duplicates", "sparse_level", "sample_size", "n_features", "metric_ID", "var_explained", "F_stat", "p_val"])
    for swab, sparse_d in itertools.product(n_samples, sparse_densities):
        print(f"Starting combination sample size: {swab} and sparsity: {sparse_d}")
        # construct matrix sample
        samples, otu_idx = construct_matrix(sparse_d, swab, samples_ids, group_A, group_B, OTU_table)
        feature_ids = {str(otu_ids[otu_idx[i]]): i for i in range(len(otu_idx))}
        groups = np.concatenate((np.ones((swab//2,)), np.zeros((swab//2,))), axis=0)
        sorted_labels = []
        for i in groups:
            if i == 1:
                sorted_labels.append("Pre Diet")
            if i == 0:
                sorted_labels.append("Termination")

        # Creates temporary blast file
        pre_filter = [pair for pair in SeqIO.parse(blast_file, "fasta") if pair.id in feature_ids]
        tmp_file = os.path.join("tmp.fa")
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

        cscs_u = Parallelize(CSCS, samples, css_matrix.toarray())
        cscs_u.astype(np.float64)

        cscs_M3_w = Parallelize(CSCS_W, samples, css_matrix.toarray())
        cscs_M3_w.astype(np.float64)

        W_cscs, var_cscs_w, var_cscs_u, M1_it, M1_weights = optimization(cscs_u)
        WW_cscs, var_cscs_ww, var_cscs_uw, M3_it, M3_weights = optimization(cscs_M3_w)
        W_BC, var_BC_w, var_BC_u, BC_it, BC_weights = optimization(BC)
        W_JD, var_JD_w, var_JD_u, JD_it, JD_weights = optimization(JD)
        W_JSD, var_JSD_w, var_JSD_u, JSD_it, JSD_weights = optimization(JSD)
        W_Euc, var_Euc_w, var_Euc_u, Euc_it, Euc_weights = optimization(Euc)

        cscs_ww = cscs_M3_w * WW_cscs
        cscs_w = cscs_u * W_cscs
        BC_w = BC * W_BC
        JD_w = JD * W_JD
        JSD_w = JSD * W_JSD
        Euc_w = Euc * W_Euc

        data_u = [cscs_u, cscs_M3_w, BC, JD, JSD, Euc]
        data_w = [cscs_w, cscs_ww, BC_w, JD_w, JSD_w, Euc_w]
        var_u = [var_cscs_u, var_cscs_uw, var_BC_u, var_JD_u, var_JSD_u, var_Euc_u]
        var_w = [var_cscs_w, var_cscs_ww, var_BC_w, var_JD_w, var_JSD_w, var_Euc_w]
        title_u = ["CSCS_M1_u", "CSCS_M3_u","Bray-curtis", "Jaccard", "Jensen-Shannon", "Euclidean"]
        title_w = ["CSCS_M1_w", "CSCS_M3_w", "Bray-curtis_w", "Jaccard_w", "Jensen-Shannon_w", "Euclidean_w"]
        weights = [M1_weights, M3_weights, BC_weights, JD_weights, JSD_weights, Euc_weights]
        iters = [M1_it, M3_it, BC_it, JD_it, JSD_it, Euc_it]

        for n, id in enumerate(data_u):
            dist = 1 - id
            np.fill_diagonal(dist, 0.0)
            dist = clean_matrix(dist)
            dist = skbio.DistanceMatrix(dist)
            result = skbio.stats.distance.permanova(dist, groups, permutations=9999)
            row = {"duplicates": s+1, "sparse_level": sparse_d, "sample_size": swab, "n_features": len(feature_ids), "metric_ID": title_u[n],\
                "var_explained": var_u[n], "F_stat": result["test statistic"], "p_val": result["p-value"]}
            df = df.append(row, ignore_index=True)
        
        for n, id in enumerate(data_w):
            dist = id / id[0,0]
            dist = 1 - dist
            np.fill_diagonal(dist, 0.0)
            dist = clean_matrix(dist)
            dist = skbio.DistanceMatrix(dist)
            result = skbio.stats.distance.permanova(dist, groups, permutations=9999)
            row = {"duplicates": s+1, "sparse_level": sparse_d, "sample_size": swab, "n_features": len(feature_ids), "metric_ID": title_w[n],\
                "var_explained": var_w[n], "F_stat": result["test statistic"], "p_val": result["p-value"]}
            df = df.append(row, ignore_index=True)
        
        #if sparse_d == 0:
        #    multi_stats(data=data_u, titles=title_u, filename=f"M3_emp_mice_unweighted_{sparse_d}", sorted_labels=sorted_labels)
        #    multi_stats(data=data_w, titles=title_w, filename=f"M3_emp_mice_weighted_{sparse_d}", sorted_labels=sorted_labels)
        #    multi_heatmaps(data=weights, titles=title_w, filename=f"M3_emp_mice_{sparse_d}", vline=iters)

        # intitiate garbage collector
        samples, otu_idx, feature_ids, groups = None, None, None, None
        pre_filter, blast_output, css_matrix, result = None, None, None, None
        BC, JD, JSD, Euc, cscs_u = None, None, None, None, None
        cscs_w, BC_w, JD_w, JSD_w, Euc_w = None, None, None, None, None
        gc.collect()

    if s == 0:
        df.to_csv("Benchmark_emp_Model3_alpha01.csv", mode='a', header=True, index=False)
    df.to_csv("Benchmark_emp_Model3_alpha01.csv", mode='a', header=False, index=False)
