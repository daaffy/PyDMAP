import sys
sys.path.insert(1,'./src/') 

import unittest
import numpy as np
import methods
import glob

class TestPreprocessing(unittest.TestCase):

    # * methods.nndist()
    def test_nndist(self):
        # Generate random points in 3D
        n = 100
        d = 3
        X = np.random.random((n,d))

        methods.nndist(X,k=1)

class TestProcessing(unittest.TestCase):

    # * methods.dynamic_laplacian_sl()
    def test_dynamics_laplacian_sl(self):
        # setup.
        X_paths = glob.glob("/Users/jackh/Documents/2023/Turbulence Research/Data/trajs/*.csv")
        traj = methods.Trajectory(X=X_paths,t=range(len(X_paths)))
        X = traj.Xat(0)

        methods.dynamic_laplacian_sl(X)

if __name__ == "__main__":
    unittest.main()


