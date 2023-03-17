import scipy.sparse as sparse
import scipy
import numpy as np
import pandas as pd
import random, time, math, pickle
import matplotlib.pyplot as plt
import itertools, os, mkl
import skbio
import seaborn as sns
import multiprocessing as mp
from sklearn.metrics.pairwise import cosine_similarity

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
    fig.savefig(f"../{title}_contours.png", format='png')

def gradient_plot_2D(M, title):
    x, y = np.gradient(M)
    fig, ax = plt.subplots()
    im = ax.imshow(M, cmap="viridis")
    ax.quiver(x, y)
    ax.set_title(f"Gradient of {title}")
    fig.savefig(f"../{title}_2D_GD.png", format='png')

def gradient_plot_3D(M, title):
    grad_x, grad_y = np.gradient(cscs_u)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(*np.meshgrid(np.arange(cscs_u.shape[0]), np.arange(cscs_u.shape[0])), cscs_u, cmap='viridis', linewidth=0)
    # Controls the arrows
    ax.quiver(*np.meshgrid(np.arange(cscs_u.shape[0]), np.arange(cscs_u.shape[0])), np.zeros_like(cscs_u), \
        grad_x, grad_y, np.zeros_like(cscs_u),\
            # Parameters for arrows
            length=0.1, normalize=True, color='r')
    # Set the labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f"Gradient of {title}")
    fig.savefig(f"../{title}_3D_GD.png", format='png')

def pca(X):
    mean = X.mean(axis=0) 
    center = X - mean 
    _, stds, pcs = np.linalg.svd(center/np.sqrt(X.shape[0])) 
    return stds**2, pcs

# Parallel C interface optimization
os.environ["USE_INTEL_MKL"] = "1"
mkl.set_num_threads(4)

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

#---------------------------------------------------------------------------------------------------------------------#
# simulated data
#---------------------------------------------------------------------------------------------------------------------#

# Set the dimensions of the data
p = 10   # number of features
n = 1000  # number of samples# Set the mean and covariance of the data
mean = np.zeros(p)
cov = np.eye(p)   # covariance matrix# Generate data from the multivariate normal distribution
data = np.random.multivariate_normal(mean, cov, n)# Add a binary covariate to the data
x = np.random.binomial(n=1, p=0.5, size=n)  # binary covariate
data[:, 0] += x  # add x to the first feature

# Compute the similarity matrix
css = np.cov(data)

#---------------------------------------------------------------------------------------------------------------------#
# Parallelization
#---------------------------------------------------------------------------------------------------------------------#

cscs_u = Parallelize(cscs, np.absolute(data), css)
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

def eig_loss(gradient):
    eigval, eigvec = np.linalg.eig(gradient)
    alpha = 2 / (eigval[0]+eigval[1])
    loss = 0
    k = np.max(eigval) / np.min(eigval)
    error = np.arccos(np.dot(eigval[0],eigval[1])/(np.linalg.norm(eigval[0])*np.linalg.norm(eigval[1])))
    loss = sum(error * ((1-(alpha*eigval[0]))**k) * eigvec[0] + error * ((1-(alpha*eigval[1]))**k) * eigvec[1])
    return alpha, loss, eigval

def GD_eigen(cscs_u):

    #WW = np.zeros([cscs_u.shape[0], cscs_u.shape[0]])

#for p, q in itertools.combinations(range(0, cscs_u.shape[1]), 2):
    # two samples
    #pair = np.outer(cscs_u[:,p], cscs_u[:,q])
    #dA = cscs_u[:,p]
    #dB = np.outer(cscs_u[:,q], cscs_u[:,q])

    # approximate alpha and beta
    sample_mean = np.mean(cscs_u)
    sample_var = np.var(cscs_u, ddof=1)
    alpha = sample_mean * (sample_mean * (1 - sample_mean) / sample_var - 1)
    if alpha < 0:
        alpha *= -1
    beta = (1 - sample_mean) * (sample_mean * (1 - sample_mean) / sample_var - 1)
    if beta < 0:
        beta *= -1
    #print(f"alpha: {alpha}\t beta: {beta}")
    # weights sample with constraint
    w = np.random.beta(alpha, beta, size=cscs_u.shape[0])
    W = np.outer(w, w)
    W.astype(np.float64)
    #W = np.clip(W, 0, 1, dtype="float32")

    # storing parameters
    eigval1 = []
    eigval2 = []
    weights = []
    alphas = []

    # storing best weights based on alpha
    best_alpha = 1.0
    best_W = 0
    best_iter = 0

    # storing best weights based on eigenvalue 1
    best_eig1 = 0
    eig1_W = 0
    eig1_it = 0
    eig1_alpha = 0

    for i in range(100):
        # Eigen decomposition        
            #Bu_eigval,_ = np.linalg.eig(dB)
            #_, Aw_eigvec = np.linalg.eig(np.outer(dA, w))
            ## Generalized symmetric pair of matrices
            #grad = Aw_eigvec.T * (dA - (dB * np.outer(Bu_eigval, Bu_eigval))) * Aw_eigvec
            M = cscs_u * W
            _, eigvec_w = np.linalg.eig(M)
            grad = eigvec_w * cscs_u * eigvec_w.T

            # gradient error & alpha
            alpha, loss, eigval = eig_loss(grad)
            
            # stores parameters
            eigval1.append(np.real(eigval[0]))
            eigval2.append(np.real(eigval[1]))
            weights.append(np.sum(np.real(W)))
            alphas.append(np.real(alpha))
            
            print(f"iter: {str(i)}\t alpha: {np.real(alpha)}\t eigval 1: {np.real(eigval[0])}\t eigval 2: {np.real(eigval[1])}\t loss: {loss}")
            # this means PC1 and PC2 cover 99% of the cov.
            if 0 <= alpha < best_alpha:
                best_alpha = alpha
                best_W = W
                best_iter = i

            if eigval[0] > best_eig1:
                best_eig1 = eigval[0]
                eig1_W = W
                eig1_it = i
                eig1_alpha = alpha

            # Update weights
            # parenthesis to allow complex to float casting
            W = (W + alpha * grad)
            #W = np.clip(W, 0, 1)

    return eigval1, eigval2, weights, alphas, best_alpha, best_W, best_iter, eig1_W, eig1_it, eig1_alpha

#---------------------------------------------------------------------------------------------------------------------#
# Bare bones gradient descent
#---------------------------------------------------------------------------------------------------------------------#
def variance_explained(gradient):
    eigval, _ = np.linalg.eigh(gradient)
    var_explained = np.sum(eigval[:2])/np.sum(eigval)
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
    W = np.outer(w, w)
    W.astype(np.float64)
    return W

def grad_function(X, W):
    M = X * W
    _, eigvec_w = np.linalg.eigh(M)
    grad = eigvec_w * X * eigvec_w.T
    return grad

def Bare_bone(X, alpha, num_iters, epss = np.finfo(np.float64).eps):
    W = initialize_theta(X)
    df1 = pd.DataFrame(columns=["iter", "variance_explained", "abs_diff", "eigval1", "eigval2"])

    prev_var, _ = variance_explained(X)
    for i in range(num_iters):
        get_grad = grad_function(X, W)
        
        current_var, eigval = variance_explained(get_grad)
        abs_diff = np.absolute(prev_var - current_var)
        # epss is based on the machine precision of float 64
        df1.loc[i] = [i, np.real(current_var), np.real(abs_diff), np.real(eigval[0]), np.real(eigval[1])]

        if abs_diff < epss:
            break

        if current_var > prev_var:
            best_W = W
            iter = i
        
        W = W + (alpha * get_grad)
        prev_var = current_var
    
    return df1, best_W, iter

def GD_alpha(X, num_iters, epss = np.finfo(np.float64).eps):
    W = initialize_theta(X)
    df1 = pd.DataFrame(columns=["iter", "variance_explained", "abs_diff", "eigval1", "eigval2"])

    prev_var, _ = variance_explained(X)
    for i in range(num_iters):
        get_grad = grad_function(X, W)
        
        current_var, eigval = variance_explained(get_grad)
        abs_diff = np.absolute(prev_var - current_var)
        alpha = np.sum(eigval) / np.sum(eigval[:2])
        # epss is based on the machine precision of float 64
        df1.loc[i] = [i, np.real(current_var), np.real(abs_diff), np.real(eigval[0]), np.real(eigval[1])]

        if abs_diff < epss:
            break

        if current_var > prev_var:
            best_W = W
            iter = i

        W = W + (alpha * get_grad)
        prev_var = current_var
    
    return df1, best_W, iter

it = 20
df_emse3, W01, i_W01 = Bare_bone(cscs_u, alpha=0.01, num_iters=it)
df_emse, eW, i_eW = GD_alpha(cscs_u, it)

fig, (ax0, ax1, ax2, ax3) = plt.subplots(4)
fig.set_size_inches(15, 10)
ax0.plot(df_emse3["iter"], df_emse3["variance_explained"], label="a=0.1")
ax0.plot(df_emse["iter"], df_emse["variance_explained"], label="a=2/e1+e2")
ax0.axvline(x=i_W01, ls='--', c="red", label="a = 0.01")
ax0.axvline(x=i_eW, ls='--', c="blue", label="a=2/e1+e2")
ax0.set_xlabel(f"iterations")
ax0.set_title("Variance explained")
ax1.plot(df_emse3["iter"], df_emse3["abs_diff"], label="a=0.1")
ax1.plot(df_emse["iter"], df_emse["abs_diff"], label="a=2/e1+e2")
ax1.set_xlabel(f"iterations")
ax1.set_title("Absolute difference")
ax2.plot(df_emse3["iter"], df_emse3["eigval1"], label="a=0.1")
ax2.plot(df_emse["iter"], df_emse["eigval1"], label="a=2/e1+e2")
ax2.set_xlabel(f"iterations")
ax2.set_title("Eigenvalue 1")
ax3.plot(df_emse3["iter"], df_emse3["eigval2"], label="a=0.1")
ax3.plot(df_emse["iter"], df_emse["eigval2"], label="a=2/e1+e2")
ax3.set_xlabel(f"iterations")
ax3.set_title("Eigenvalue 2")
ax1.legend()
fig.tight_layout(pad=2.0)
fig.savefig("../cscsw_simulated_par.png", format='png')

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

contours(cscs_u, title="CSCS_unweighted")
gradient_plot_2D(cscs_u, title="CSCS_unweighted")
gradient_plot_3D(cscs_u, title="CSCS_unweighted")

#---------------------------------------------------------------------------------------------------------------------#
# Visualizing simulated data
#---------------------------------------------------------------------------------------------------------------------#
"""
pca_color = sns.color_palette(None, cscs_u.shape[1])
eigval1, eigval2, weights, alphas, alpha, alpha_W, alpha_it, eig1_W, eig1_it, eig_alpha = GD_eigen(cscs_u)

print(f"alpha: {alpha}\t eigenvalue alpha: {eig_alpha}")

var_u, pcs_u = pca(cscs_u)
var_w, pcs_w = pca(cscs_u*alpha_W)
var_eig_w, pcs_eig_w = pca(cscs_u*eig1_W)
### subplot 2 ###
# font size
font_size = 15
plt.rcParams.update({"font.size": 12})

fig0, (ax1, ax2) = plt.subplots(2)
fig0.set_size_inches(15, 10)
# unweigthed cscs
for i in range(cscs_u.shape[1]):
    ax1.scatter(pcs_u[0][i], pcs_u[1][i], color=pca_color[i], s=10, label=f"{i+1}")
    ax1.annotate(f"{str(i+1)}", (pcs_u[0][i], pcs_u[1][i]))
ax1.set_xlabel(f"PC1: {round(var_u[0]/np.sum(var_u)*100,2)}%")
ax1.set_ylabel(f"PC2: {round(var_u[1]/np.sum(var_u)*100,2)}%")
ax1.legend(loc='center left', bbox_to_anchor=(1, 0.7))

# Weighted cscs
for i in range(cscs_u.shape[1]):
    ax2.scatter(pcs_eig_w[0][i], pcs_eig_w[1][i], color=pca_color[i], s=10, label=f"{i+1}")
    ax2.annotate(f"{str(i+1)}", (pcs_eig_w[0][i], pcs_eig_w[1][i]))
ax2.set_xlabel(f"PC1: {round(var_eig_w[0]/np.sum(var_eig_w)*100,2)}%")
ax2.set_ylabel(f"PC2: {round(var_eig_w[1]/np.sum(var_eig_w)*100,2)}%")
ax2.legend(loc='center left', bbox_to_anchor=(1, 0.7))
fig0.savefig("../cscsw_PCA_eig.png", format='png')


fig1, (ax1, ax2) = plt.subplots(2)
fig1.set_size_inches(15, 10)
# unweigthed cscs
for i in range(cscs_u.shape[1]):
    ax1.scatter(pcs_u[0][i], pcs_u[1][i], color=pca_color[i], s=10, label=f"{i+1}")
    ax1.annotate(f"{str(i+1)}", (pcs_u[0][i], pcs_u[1][i]))
ax1.set_xlabel(f"PC1: {round(var_u[0]/np.sum(var_u)*100,2)}%")
ax1.set_ylabel(f"PC2: {round(var_u[1]/np.sum(var_u)*100,2)}%")
ax1.legend(loc='center left', bbox_to_anchor=(1, 0.7))

# Weighted cscs
for i in range(cscs_u.shape[1]):
    ax2.scatter(pcs_w[0][i], pcs_w[1][i], color=pca_color[i], s=10, label=f"{i+1}")
    ax2.annotate(f"{str(i+1)}", (pcs_w[0][i], pcs_w[1][i]))
ax2.set_xlabel(f"PC1: {round(var_w[0]/np.sum(var_w)*100,2)}%")
ax2.set_ylabel(f"PC2: {round(var_w[1]/np.sum(var_w)*100,2)}%")
ax2.legend(loc='center left', bbox_to_anchor=(1, 0.7))
fig1.savefig("../cscsw_PCA_alpha.png", format='png')

### subplot 2 ###
fig2, (ax1, ax2) = plt.subplots(2)
fig2.set_size_inches(15, 10)
palette = sns.color_palette(None, 4)
X = [i for i in range(len(eigval1))]
# visualization
ax1.scatter(X, eigval1, color=palette[0], s=5, label=f"eigenvalue 1")
ax1.scatter(X, eigval2, color=palette[1], s=5, label=f"eigenvalue 2")
ax1.axvline(x=alpha_it, ls='--') # point of best alpha
ax1.axvline(x=eig1_it, ls='-')
ax1.set_ylabel(f"Eigenvalues")
ax1.legend(loc='center left', bbox_to_anchor=(1, 0.7))

ax2.scatter(X, weights, color=palette[2], s=5, label=f"weights")
ax2.scatter(X, alphas, color=palette[3], s=5, label=f"alpha")
ax2.axvline(x=alpha_it, ls='--')
ax2.axvline(x=eig1_it, ls='-')
ax2.set_xlabel(f"iterations")
ax2.set_ylabel(f"weight & alpha")
ax2.legend(loc='upper left', bbox_to_anchor=(1, 0.8))
fig2.savefig("../cscsw_parameters.png", format='png')
"""