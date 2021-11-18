import numpy as np

def BuildProjectionConstraintMatrix(points2D, points3D):

  # TODO
  # For each correspondence, build the two rows of the constraint matrix and stack them

  num_corrs = points2D.shape[0]
  constraint_matrix = np.zeros((num_corrs * 2, 12))


  for i in range(num_corrs):
    # TODO Add your code here
    X = np.append(points3D[i,:], [1])

    constraint_matrix[2*i,:] = np.append(np.zeros(4), np.append(-X, points2D[i,1] * X))
    constraint_matrix[2*i + 1,:] = np.append(X, np.append(np.zeros(4), -points2D[i,0] * X))

  return constraint_matrix