import pandas as pd
import glob
import os
import numpy as np
import scipy
import gc

# -----------------------------------------------------------------------------------------------------------
#    Pre-processing Methods
# -----------------------------------------------------------------------------------------------------------

'''
    get_persistent_ids
Get particles IDs that persist over the whole time series defined in input_path.
'''
def get_persistent_ids(input_path,
                       id_column=6,
                       max_search=int(3e5)):
    
    files = glob.glob(input_path+"*.txt")

    # Get IDs for each time-step
    ids = []
    for i in range(len(files)):
        df = pd.read_csv(files[i], 
                        sep=" ",
                        header=1) 
        
        curr_ids = df.iloc[:,id_column].values
        ids.append(curr_ids)
    
    def mask_arr(id):
        mask = np.full((max_search,1), False)
        mask[id-1] = True
        return mask

    persistent_ids = np.full((max_search,1), True)

    for i in range(len(ids)):
        persistent_ids = persistent_ids * mask_arr(ids[i])
        
    return persistent_ids
    

'''
    reformat
Basic reformating of .txt files to .csv files. Originally to deal with DNS 3900 data that Paraview
had difficulty reading. Can also supply specified particle IDs to track as a list.
'''
def reformat(input_path,
            output_path,
            particle_ids=None,  # Ignore by default.
            id_column=6):
    
    files = glob.glob(input_path+"*.txt") # .txt extension by default.

    for i in range(len(files)):
        file = files[i]
        file_name = os.path.basename(file)
        file_name_noext = os.path.splitext(file_name)[0]

        df = pd.read_csv(files[i], 
                    sep=" ",
                    header=1)
        
        # test this...
        if not particle_ids == None: 
            mask = df.loc[:,id_column].isin(particle_ids)

        df.to_csv(output_path+file_name_noext+".csv",
              header=None,
              index=False)
        
# -----------------------------------------------------------------------------------------------------------
#    Numerical Methods
# -----------------------------------------------------------------------------------------------------------

class Trajectory:
    def __init__(self, X=None, t=None, sub_inds=None):
        self.X = X
        self.t = np.array(t)
        self.df = None

        self.n_s = None
        self.n_t = len(t)

        self.sub_inds = sub_inds

        if isinstance(self.X, list): # paths
            self.grab_mode = 1
            # checks
            assert len(self.X) == len(self.t)
            # all trajectories present across t?

            # load parameters
            self.Xat(0)

        elif isinstance(self.X, np.ndarray): # idk yet
            self.grab_mode = 0

            assert len(np.shape(X)) == 3
            assert np.shape(X)[2] == len(self.t)

            self.Xat(0)
        else:
            raise Exception()

    def Xat(self,t_i):
        if self.grab_mode == 0: # 3D-Array.
            self.n_s = np.shape(self.X)[0]
            return self.X[:,:,t_i]
        elif self.grab_mode == 1: # Load from files.
            self.df = pd.read_csv(self.X[t_i])
            if self.sub_inds == None:
                temp = self.df.iloc[:,1:].values
            else:
                temp = self.df.iloc[self.sub_inds,1:].values
            self.n_s = len(temp)
            return temp
        else:
            raise Exception("Grab mode not recognised.")


'''

Notes:
- https://relate.cs.illinois.edu/course/cs450-s19/f/demos/upload/bvp_odes/Sparse%20matrices.html
'''
def dynamic_laplacian(traj,eps=None,alpha=1,include_mult=1):
    if not isinstance(traj,Trajectory):
        raise Exception("traj must be a Trajectory object!")
    
    n_s = traj.n_s
    n_t = traj.n_t
    # alpha = 1 # see Coifman/Lafon

    # f(x) = iso_kernel(x,0,epsilon) # ???

    if (n_t>1):
        temp = traj.t[1:]-traj.t[:-1]
        # coeff = np.zeros((n_t,1))
        # coeff[0] = temp[0]
        # coeff[1:-1] = temp[:-1] + temp[1:]
        # coeff[-1] = temp[-1]
        # coeff = coeff * 0.5 / (traj.t[-1] - traj.t[0])

        coeff = np.ones((n_t,)) # temp.
    else:
        # temp = 1.
        coeff = [1]

    frange = range(n_t)
    # if n_t == 1:
    #     frange = iter([0])
    # else:
    #     frange = range(n_t)

    P_final = scipy.sparse.coo_matrix((n_s,n_s))
    # row = []
    # col = []
    # data = []
    for t_i in frange:
        X_t = traj.Xat(t_i)
        # print(np.shape(dynamic_laplacian_sl(X_t,eps=eps)))
        P_final = P_final + coeff[t_i]*dynamic_laplacian_sl(X_t,eps=eps,alpha=alpha,include_mult=include_mult)


    """
        add lagrange multipliers
    """

    return P_final/n_t

    '''
    


    # p = Progress(n_t,desc="Calculating dynamic Laplacian...",barglyphs=BarGlyphs("[=> ]"),barlen=50,color=:yellow)

    
    P_final = zeros(n_s,n_s)
    temp = similar(P_final)
    for ti = 1:n_t
        # next!(p)
        temp .= pairwise(metric,traj.X[:,ti,:],traj.X[:,ti,:],dims=1)
        # temp .= broadcast(f,temp)
        temp .= _kernel.(temp,epsilon)
        temp .*= broadcast(>=,temp,threshold) # threshold the kernel matrix using broadcasting
        
        # # diagnose
        # println(sum(P_final.>=1e-8)/(n_s^2))
        # display(heatmap(temp))

        q_bar = (sum(temp, dims=2)./n_s).^alpha
        temp ./= (q_bar .* q_bar')
        d_barsum = sum(temp, dims=1)./n_s
        temp ./= d_barsum
        P_final .+= coeff[ti].*temp./n_s
    end
    # P_final /= traj.t[end]-traj.t[1] # make sure this hasn't changed anything
    P_final

    if (boundary_idxs != [])
        for idx in boundary_idxs
            P_final[:,idx] = zeros(n_s)
            P_final[idx,:] = zeros(n_s)
        end
    end
    '''

"""
    nndist
Pre-processing step to determine kernel size for dynamic Laplacian construction.
"""
def nndist(X,k=1,metric="Euclidean"):
    kdtree = scipy.spatial.KDTree(X)
    nearest_dist, _ = kdtree.query(X, k=k+1)

    del kdtree
    gc.collect()

    avg = np.mean(nearest_dist[:,k])

    return avg

'''
    dynamic_laplacian_sl
Calculate Dynamic Laplacian for a time of time. X must therefore be 2-dimensional.
'''
def dynamic_laplacian_sl(X, eps=None, alpha=1, include_mult=1, mode="sparse"):
    assert len(np.shape(X)) == 2

    if eps == None: # Calculate default...
        eps = nndist(X,k=7)/np.sqrt(2) # ... based on 5-point stencil

    n_s = np.shape(X)[0] # X is a square matrix

    kdtree = scipy.spatial.KDTree(X)
    temp = kdtree.sparse_distance_matrix(kdtree,eps*include_mult).tocoo()
    temp.data = iso_kernel(temp.data,0,eps)

    # q_bar = np.power((np.sum(temp, axis=1)/n_s),alpha)
    # temp.data = np.divide(temp.data,np.multiply(q_bar[temp.row], q_bar[temp.col]).T).A1
    # d_barsum = np.sum(temp, axis=0)/n_s
    # temp.data = np.divide(temp.data,d_barsum.A1[temp.col])

    q_bar = np.power((np.sum(temp, axis=1)),alpha)
    qq_bar = np.multiply(q_bar[temp.row], q_bar[temp.col]).A1
    temp.data = np.divide(temp.data, qq_bar)
    d_barsum = np.sum(temp, axis=1)
    temp.data = np.divide(temp.data,d_barsum[temp.col].T).A1

    return temp

    # kdtree = scipy.spatial.KDTree(X)
    # temp = kdtree.sparse_distance_matrix(kdtree,1).tocoo()
    # temp.data = iso_kernel(temp.data,0,eps)

    # q_bar = np.power((np.sum(temp, axis=1)),alpha)
    # qq_bar = np.multiply(q_bar[temp.row], q_bar[temp.col]).A1
    # temp.data = np.divide(temp.data, qq_bar)
    # d_barsum = np.sum(temp, axis=1)
    # temp.data = np.divide(temp.data,d_barsum[temp.col].T).A1

    # Garbage collect; necessary?
    # del temp
    # del kdtree
    # gc.collect()


    '''
    q_bar = (sum(temp, dims=2)./n_s).^alpha
        temp ./= (q_bar .* q_bar')
        d_barsum = sum(temp, dims=1)./n_s
        temp ./= d_barsum
        P_final .+= coeff[ti].*temp./n_s
    '''


def metric(mode="Euclidean"):
    if mode == "Euclidean":
        return 0
    else:
        raise Exception("Mode not recognised.")
    
def iso_kernel(x_1, x_2, epsilon):
    # assert np.shape(x_1) == np.shape(x_2)

    diff = x_1-x_2
    if len(np.shape(x_1)) == 1:
        diff = diff[:,None]
    
    return np.exp(-np.linalg.norm(diff,axis=1)**2/epsilon)

"""
    SEBA
Python implementation of the SEBA algorithm.
Adapted from Julia code written by Gary Froyland.
"""
def SEBA(V, Rinit = None):
    # V is pxr matrix (r vectors of length p as columns)
    # Rinit is an (optional) initial rotation matrix.

    # Outputs:
    # S is pxr matrix with columns approximately spanning the column space of V
    # R is the optimal rotation that acts on V, which followed by thresholding, produces S

    # Begin SEBA algorithm
    maxiter = 5000   # maximum number of iterations allowed
    F,_ = np.linalg.qr(V) # Enforce orthonormality
    V = F # (!) needed?
    (p,r) = np.shape(V)
    mu = 0.99 / np.sqrt(p)

    S = np.zeros(np.shape(V))

    # Perturb near-constant vectors
    for j in range(r):
            if np.max(V[:, j]) - np.min(V[:, j]) < 1e-14:
                    V[:, j] = V[:, j] + (np.random.random((p, 1)) - 1 / 2) * 1e-12

    # is R correct?

    # ...
    # Initialise rotation
    if Rinit == None:
            Rnew = np.eye(r) # depends on context?
    else:
            # Ensure orthonormality of Rinit
            U, _, Vt = np.linalg.svd(Rinit)
            Rnew = np.matmul(U , Vt)

    #preallocate matrices
    R = np.zeros((r, r))
    Z = np.zeros((p, r))
    Si = np.zeros((p, 1))

    iter = 0
    while np.linalg.norm(Rnew - R) > 1e-14 and iter < maxiter:
            iter = iter + 1
            R = Rnew
            Z = np.matmul(V , R.T)

            # Threshold to solve sparse approximation problem
            for i in range(r):
                    Si = soft_threshold(Z[:,i], mu)
                    S[:, i] = Si / np.linalg.norm(Si)
            # Polar decomposition to solve Procrustes problem
            U, _, Vt = np.linalg.svd(np.matmul(S.T , V), full_matrices=False)
            Rnew = np.matmul(U , Vt)

    # Choose correct parity of vectors and scale so largest value is 1
    for i in range(r):
            S[:, i] = S[:, i] * np.sign(sum(S[:, i]))
            S[:, i] = S[:, i] / np.max(S[:, i])

    # Sort so that most reliable vectors appear first
    ind = np.argsort(np.min(S, axis=0))
    S = S[:, ind]

    return S, R

def soft_threshold(z, mu):
            assert len(np.shape(z)) <= 1 # only accept scalars or vectors

            temp = np.zeros(np.shape(z))
            if len(np.shape(z)) == 1:
                    for i in range(len(z)):
                            temp[i] = np.sign(z[i]) * np.max([np.abs(z[i]) - mu, 0])
            else:
                    temp = np.sign(z) * np.max([np.abs(z) - mu, 0])        
            
            return temp