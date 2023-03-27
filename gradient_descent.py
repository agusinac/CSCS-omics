import scipy.sparse as sparse
import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools, os, mkl, pickle
import skbio
import seaborn as sns
import multiprocessing as mp
import igraph as ig

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
    p1.set(xlabel=f"{title}", ylabel="")
    p1.set(title="Weights per iteration")
    plt.savefig(f"../heatmap_{title}.png", format="png")
    # important to prevent overlap between seaborn and matplotlib
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
# simulated data
#---------------------------------------------------------------------------------------------------------------------#

#log-normal dist for positive numbers -> features
#
#binomial to give features interesting stuff for one group and another
#
#linear model ~ of features, 
#               -two linear factors to optimize on both PC1 and PC2, add two binomial dist. (read into it)
#DONE           -creating orthologonal values (read into it) 
#IN PROGRESS    -iGraph sampling algorithm
#
#DONE           log-normal -> intercept
#iGraph sampling 
#Select communities? Based on zero's and one's. 
#
#DONE           features = Uniform dist -> orthologonal values * Beta in linear model

def get_uniform(n_samples=10, n_features=2, Beta_switch=[1,1]):
    # defines X attributes
    x1 = np.random.uniform(low=0.5, high=1.0, size=(n_samples,))
    x2 = np.random.uniform(low=0.0, high=0.5, size=(n_samples,))

    # np.newaxis increases number of dimensions to allow broadcasting
    X = np.concatenate((x1[:, np.newaxis], x2[:, np.newaxis]), axis=1)

    if n_features > 2:
        X = np.random.choice([x1, x2], size=(n_samples, n_features-2))
        X = np.concatenate((x1[:, np.newaxis], x2[:, np.newaxis], X), axis=1)
    
    # defines intercept C
    C = np.random.lognormal(mean=np.mean(X), sigma=np.std(X), size=n_samples)
    
    # Switches off X-attributes
    Beta = np.ones((n_samples, n_features), dtype=int)
    for i, col in enumerate(Beta_switch):
        if col == 0:
            Beta[:, i] = np.zeros((1, 1), dtype=int)

    # computes linear model for n samples
    linear_eq = np.sum(Beta * X, axis=1) + C
    return linear_eq


#---------------------------------------------------------------------------------------------------------------------#
# Parallelization
#---------------------------------------------------------------------------------------------------------------------#

# Parallel C interface optimization
os.environ["USE_INTEL_MKL"] = "1"
mkl.set_num_threads(4)

#cscs_u = Parallelize(cscs, np.absolute(samples), css)
#cscs_u.astype(np.float32)

#M = cscs_u
#for impl in [np.linalg.eig, np.linalg.eigh, scipy.linalg.eig, scipy.linalg.eigh]:
#    w, v = impl(M)
#    print(np.sort(w))
#    reconstructed = np.dot(v * w, v.conj().T)
#    print("Allclose:", np.allclose(reconstructed, M), '\n')
    

#---------------------------------------------------------------------------------------------------------------------#
# Optimization algorithm
#---------------------------------------------------------------------------------------------------------------------#

def variance_explained(gradient):
    eigval = np.linalg.eigvals(gradient)
    var_explained = np.sum(eigval[:2]) / np.sum(eigval)
    return var_explained, eigval

def initialize_theta(X):
    sample_mean = np.mean(X)
    sample_var = np.var(X, ddof=1)
    alpha = sample_mean * (sample_mean * (1 - sample_mean) / sample_var - 1)
    if alpha < 0:
        alpha *= -1
    beta = (1 - sample_mean) * (sample_mean * (1 - sample_mean) / sample_var - 1)
    if beta < 0:
        beta *= -1
    #print(f"alpha: {alpha}\t beta: {beta}")
    # weights sample with constraint
    w = np.random.beta(alpha, beta, size=X.shape[0])
    W = np.triu(w, 1) + np.triu(w, 1).T
    W[np.diag_indices(W.shape[0])] = 1
    W.astype(np.float32)
    return W

def grad_function(X, W):
    M = X * W
    _, eigvec_w = np.linalg.eig(M)
    grad = eigvec_w * X * eigvec_w.T
    return grad

def add_column(m1, m2):
    return np.column_stack((m1, m2))

def Bare_bone(X, W, alpha, num_iters, epss = np.finfo(np.float32).eps):
    #W = initialize_theta(X)
    df1 = pd.DataFrame(columns=["iter", "variance_explained", "abs_diff", "eigval1", "eigval2"])

    best_var, best_W, iter = 0, 0, 0
    prev_var, _ = variance_explained(X)
    
    Weight_stack = W[:,0]

    for i in range(num_iters):
        get_grad = grad_function(X, W)
        
        current_var, eigval = variance_explained(get_grad)
        abs_diff = np.sum(np.absolute(current_var - prev_var))
        # epss is based on the machine precision of np.float32 64
        df1.loc[i] = [i, np.real(current_var), np.real(abs_diff), np.real(eigval[0]), np.real(eigval[1])]

        if abs_diff < epss:
            break

        if current_var > best_var:
            best_var = current_var
            best_W = W
            iter = i
        
        W = (W + alpha * get_grad)
        W = np.clip(W, 0, 1)
        prev_var = current_var
        Weight_stack = add_column(Weight_stack, W[:,0])
    
    return df1, best_W, iter, Weight_stack

def GD_alpha(X, W, num_iters, epss = np.finfo(np.float32).eps):
    #W = initialize_theta(X)
    df1 = pd.DataFrame(columns=["iter", "variance_explained", "abs_diff", "eigval1", "eigval2"])

    best_var, best_W, iter = 0, 0, 0
    prev_var, _ = variance_explained(X)
    Weight_stack = W[:,0]
    
    for i in range(num_iters):
        get_grad = grad_function(X, W)
        
        current_var, eigval = variance_explained(get_grad)
        abs_diff = np.sum(np.absolute(current_var - prev_var))
        alpha = 2 / np.sum(eigval[:2])
        print(alpha)
        # epss is based on the machine precision of np.float32 64
        df1.loc[i] = [i, np.real(current_var), np.real(abs_diff), np.real(eigval[0]), np.real(eigval[1])]

        if abs_diff < epss:
            break

        if current_var > best_var:
            best_var = current_var
            best_W = W
            iter = i

        W = (W + alpha * get_grad)
        W = np.clip(W, 0, 1)
        prev_var = current_var
        Weight_stack = add_column(Weight_stack, W[:,0])
    
    return df1, best_W, iter, Weight_stack

it = 100
a = 0.1
"""
W = initialize_theta(cscs_u)
df_emse3, W01, i_W01, weights_fixed_alpha = Bare_bone(cscs_u, W, alpha=a, num_iters=it)
df_emse, eW, i_eW, weights_unfixed_alpha = GD_alpha(cscs_u, W, it)


#---------------------------------------------------------------------------------------------------------------------#
# Visualizing simulated data
#---------------------------------------------------------------------------------------------------------------------#

fig, (ax0, ax1, ax2, ax3) = plt.subplots(4)
fig.set_size_inches(15, 10)
ax0.plot(df_emse3["iter"], df_emse3["variance_explained"], label="a=0.1")
ax0.plot(df_emse["iter"], df_emse["variance_explained"], label="a=2/e1+e2")
ax0.axvline(x=i_W01, ls='--', c="red", label=f"a = {a}")
ax0.axvline(x=i_eW, ls='--', c="blue", label="a=2/e1+e2")
ax0.set_xlabel(f"iterations")
ax0.set_title("Variance explained")
ax1.plot(df_emse3["iter"], df_emse3["abs_diff"], label=f"a = {a}")
ax1.plot(df_emse["iter"], df_emse["abs_diff"], label="a=2/e1+e2")
ax1.set_xlabel(f"iterations")
ax1.set_title("Absolute difference")
ax2.plot(df_emse3["iter"], df_emse3["eigval1"], label=f"a = {a}")
ax2.plot(df_emse["iter"], df_emse["eigval1"], label="a=2/e1+e2")
ax2.set_xlabel(f"iterations")
ax2.set_title("Eigenvalue 1")
ax3.plot(df_emse3["iter"], df_emse3["eigval2"], label=f"a = {a}")
ax3.plot(df_emse["iter"], df_emse["eigval2"], label="a=2/e1+e2")
ax3.set_xlabel(f"iterations")
ax3.set_title("Eigenvalue 2")
ax1.legend()
fig.tight_layout(pad=2.0)
fig.savefig("../cscsw_simulated_par.png", format='png')
plt.clf()

var_u, pcs_u = pca(cscs_u)
var_W01, pcs_W01 = pca(cscs_u*W01)
var_eW, pcs_eW = pca(cscs_u*eW)


### subplot 2 ###
# font size
font_size = 15
plt.rcParams.update({"font.size": 12})
pca_color = sns.color_palette(None, cscs_u.shape[1])
fig0, (ax1, ax2, ax3) = plt.subplots(3)
fig0.set_size_inches(15, 10)
# unweigthed cscs
for i in range(cscs_u.shape[1]):
    ax1.scatter(pcs_u[0][i], pcs_u[1][i], color=pca_color[i], s=10, label=f"{i+1}")
    ax1.annotate(f"{str(i+1)}", (pcs_u[0][i], pcs_u[1][i]))
ax1.set_xlabel(f"PC1: {round(var_u[0]/np.sum(var_u)*100,2)}%")
ax1.set_ylabel(f"PC2: {round(var_u[1]/np.sum(var_u)*100,2)}%")
ax1.set_title(f"Unweighted CSCS")
ax1.legend(loc='center left', bbox_to_anchor=(1, 0.7))

# Weighted cscs alpha = 0.1
for i in range(cscs_u.shape[1]):
    ax2.scatter(pcs_W01[0][i], pcs_W01[1][i], color=pca_color[i], s=10, label=f"{i+1}")
    ax2.annotate(f"{str(i+1)}", (pcs_W01[0][i], pcs_W01[1][i]))
ax2.set_xlabel(f"PC1: {round(var_W01[0]/np.sum(var_W01)*100,2)}%")
ax2.set_ylabel(f"PC2: {round(var_W01[1]/np.sum(var_W01)*100,2)}%")
ax2.set_title(f"Weighted CSCS with alpha = 0.1")

# Weighted cscs alpha = 2 / eigval 1 + 2
for i in range(cscs_u.shape[1]):
    ax3.scatter(pcs_eW[0][i], pcs_eW[1][i], color=pca_color[i], s=10, label=f"{i+1}")
    ax3.annotate(f"{str(i+1)}", (pcs_eW[0][i], pcs_eW[1][i]))
ax3.set_xlabel(f"PC1: {round(var_eW[0]/np.sum(var_eW)*100,2)}%")
ax3.set_ylabel(f"PC2: {round(var_eW[1]/np.sum(var_eW)*100,2)}%")
ax3.set_title(f"Weighted CSCS with alpha 2 / e1+e2")
#ax3.legend(loc='center left', bbox_to_anchor=(1, 0.7))
fig0.tight_layout(pad=2.0)
fig0.savefig("../cscsw_PCA.png", format='png')
plt.clf()

#print(f"symmetrical cscs: {scipy.linalg.issymmetric(cscs_u)}")
#print(f"symmetrical weights: {scipy.linalg.issymmetric(eW)}")

heatmap_W(weights_fixed_alpha, "fixed_alpha")
heatmap_W(weights_unfixed_alpha, "unfixed_alpha")

eM = cscs_u * eW
M01 = cscs_u * W01

contours(eM, title="cscs_eAlpha")
contours(M01, title="cscs_alpha01")
gradient_plot_3D(eM, title="cscs_eAlpha")
gradient_plot_3D(M01, title="cscs_alpha01")
"""