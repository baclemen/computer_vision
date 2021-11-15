import time
import os
import random
import math
import torch
import numpy as np

# run `pip install scikit-image` to install skimage, if you haven't done so
from skimage import io, color
from skimage.transform import rescale

def distance(x, X):
    dist3d = X - x
    dist1d = torch.sum(torch.square(dist3d), axis=1)
    # dist1d = torch.einsum('ij->i', dist3d)
    # dist1d = dist3d[:,0] ** 2 + dist3d[:,1] ** 2 + dist3d[:,2] ** 2
    return dist1d

def distance_batch(x, X, batch_size):
    # dist3d = torch.zeros([batch_size, X.shape[0], X.shape[1]])
    # for i in range(batch_size):
    #     dist3d[i,:,:] = X - x[i]
    dist3d = X.repeat(batch_size,1,1) - x.unsqueeze(1).repeat(1, len(X), 1)
    dist1d = torch.sum(torch.square(dist3d), axis=2)
    # dist1d = dist3d[:,0] ** 2 + dist3d[:,1] ** 2 + dist3d[:,2] ** 2
    return dist1d

def gaussian(dist, bandwidth):
    return torch.exp(-dist / (2 * (bandwidth ** 2)))

def update_point(weight, X):
    temp = torch.mul(weight.unsqueeze(dim=1), X)
    return torch.sum(temp, dim=0) / torch.sum(weight)
    
    # torch.FloatTensor([sum(weight * X[:,0]), sum(weight * X[:,1]), sum(weight * X[:,2])])

def update_point_batch(weight, X, batch_size):
    # temp = torch.zeros([batch_size,3])
    # temp = torch.sum(torch.mul(weight.unsqueeze(dim=2), X), dim=0) / 
    w_pts = torch.mul(weight.unsqueeze(dim=2), X.repeat(batch_size,1,1))
    weights = torch.sum(weight, axis = 1)
    # for i in range(batch_size):
    #     temp[i] = torch.sum(torch.mul(weight[i].unsqueeze(dim=1), X), dim=0) / torch.sum(weight[i])
    return torch.einsum('ji,j->ji',torch.sum(w_pts, dim=1), (1/weights))


def meanshift_step(X, bandwidth=2.5):
    X_ = X.clone()
    for i, x in enumerate(X):
        dist = distance(x, X)
        weight = gaussian(dist, bandwidth)
        X_[i] = update_point(weight, X)
    return X_

def meanshift_step_batch(X, bandwidth=2.5):
    X_ = X.clone()
    batch_size = 500

    for i in range(0, len(X) - batch_size + 1, batch_size):
        dist = distance_batch(X[i:i+batch_size], X, batch_size)
        weight = gaussian(dist, bandwidth)
        X_[i:i+batch_size] = update_point_batch(weight, X, batch_size)

    if(len(X) % batch_size > 0):
        final_batch_size = len(X) % batch_size
        dist = distance_batch(X[-final_batch_size:], X, final_batch_size)
        weight = gaussian(dist, bandwidth)
        X_[-final_batch_size:] = update_point_batch(weight, X, final_batch_size)


    return X_

def meanshift(X):
    X = X.clone()
    for _ in range(20):
        # X = meanshift_step(X)   # slow implementation
        X = meanshift_step_batch(X)   # fast implementation
    return X

scale = 0.25    # downscale the image to run faster
torch.set_grad_enabled(False) 

# Load image and convert it to CIELAB space
image = rescale(io.imread('cow.jpg'), scale, multichannel=True)
image_lab = color.rgb2lab(image)
shape = image_lab.shape # record image shape
image_lab = image_lab.reshape([-1, 3])  # flatten the image

# Run your mean-shift algorithm
t = time.time()
# X = meanshift(torch.from_numpy(image_lab)).detach().cpu().numpy()

X = meanshift(torch.from_numpy(image_lab).cuda()).detach().cpu().numpy()  # you can use GPU if you have one

t = time.time() - t
print ('Elapsed time for mean-shift: {}'.format(t))

# Load label colors and draw labels as an image
colors = np.load('colors.npz')['colors']
colors[colors > 1.0] = 1
colors[colors < 0.0] = 0

centroids, labels = np.unique((X / 4).round(), return_inverse=True, axis=0)

result_image = colors[labels].reshape(shape)
result_image = rescale(result_image, 1 / scale, order=0, multichannel=True)     # resize result image to original resolution
result_image = (result_image * 255).astype(np.uint8)
io.imsave('result.png', result_image)
