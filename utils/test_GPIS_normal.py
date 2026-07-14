## Description
# This Script loads a 3D point cloud in pcd format with its normals.
# Then performs a 3D Reconstruction using GPIS.

## Add libraries
# import pcl
import open3d as o3d
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import gpytorch
import sys
import time
import torch
import itertools
import plotly.graph_objects as go
import scipy.spatial.distance as metric

# import mcubes
# import plyfile
# import compute_depth_normals
# import compute_tactile_normals

n_pts=128
d = 0.2
d_pos = 0.05
d_neg = 0.05 # 0.2
npar = 0.001 # 0.03

# Grid resolution
res = 64 # 150 # grid resolution # 50
# Note to add all depndencies folders or libraries
def scatter3D(points_list, colors):
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')

	for points, color in zip(points_list, colors):
		ax.scatter(points[:,0], points[:,1], points[:,2], c=color, marker='o')

	ax.set_xlabel('X Label')
	ax.set_ylabel('Y Label')
	ax.set_zlabel('Z Label')

	plt.show()

def _generate_ply_data(points, faces):
    """

    :param points:
    :param faces:
    :return:
    """
    vertices = [(point[0], point[1], point[2]) for point in points]
    faces = [(point,) for point in faces]

    vertices_np = np.array(vertices, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    faces_np = np.array(faces, dtype=[('vertex_indices', 'i4', (3,))])

    vertex_element = plyfile.PlyElement.describe(vertices_np, 'vertex')
    face_element = plyfile.PlyElement.describe(faces_np, 'face')

    return plyfile.PlyData([vertex_element, face_element], text=True)


# Place matlab in the working directory

## Load 3D Object
depth_pcd_filename = 'mustard/depth_cloud_cf.pcd'
tactile_pcd_filename = 'mustard/tactile_cloud_cf.pcd'
outputfile_obj = './data/outputs/Bunny.ply'

pcd = o3d.io.read_point_cloud("bunny_ascii.pcd")

# print(np.asarray(pcd.points))
# print(np.asarray(pcd.normals))
v_points = np.asarray(pcd.points)
v_normals = np.asarray(pcd.normals)

sel = np.random.choice(len(v_normals),n_pts, replace=False)
v_points = v_points[sel]
v_normals = v_normals[sel]

# v_pcd = pcl.PointCloud()
# v_pcd.from_file(depth_pcd_filename)
# v_points = v_pcd.to_array()
# v_normals = compute_depth_normals.compute_normals(v_pcd, ksearch=10, search_radius=0)

# t_points = v_points = v_points[::10]
# t_normals = v_normals = v_normals[::10]

# print("Downsampled depth cloud size: " + str(v_points.shape))

# t_pcd = pcl.PointCloud()
# t_pcd.from_file(tactile_pcd_filename)
# t_points = t_pcd.to_array()
# t_normals = compute_tactile_normals.compute_normals(t_pcd)

# print("tactile cloud size: " + str(t_points.shape))

# vt_points = np.concatenate([v_points, t_points])
# vt_normals = np.concatenate([v_normals, t_normals])

# D_frompcd = pcl.PointCloud()
# D_frompcd.from_file(filename_obj)
# Format: x y z nx ny nz radius

## Prepare Data (Computing Constraints) % Assuming normals are correct ...
# Parameters can be computed automatically as a function of the size of the object(e.g. 10% of the size defined by a bounding box, see (Wendland, 2002, Surface Reconstructions from unorganized point clouds).


## Computing inside and outside constraints based on normals
normals = v_normals # D_frompcd.calc_normals(0.1, 20)
points = v_points # D_frompcd.to_array()

points_out = points + d_neg * normals
points_in = points - d_pos * normals

print("Expected ((166, 3), (166, 3), (166, 3))")
print(points.shape, points_in.shape, points_out.shape)

## Prepare f(x) as signace distance function
# fone=ones(1,size(points_in',1))*1;
# fminus=-1*ones(1,size(points_out',1))*1;
fone   = np.ones((points_in.shape[0], 1)) #* d_pos
fminus = -1 * np.ones((points_out.shape[0], 1))# * d_neg
X      = points
fzero  = np.zeros((X.shape[0], 1))

print("Expected ((166, 1), (166, 1), (166, 3), (166, 1))")
print(fone.shape, fminus.shape, X.shape, fzero.shape)

## Visualize Object (Cube)
# Notice that the scale of the Sphere goes from -20 to 20
# scatter3D([points, points_out, points_in], ['r', 'g', 'b'])
# exit()

# Training data

minx = np.min(X, axis=0) - 0.6 # We extend the boundaries of the object a bit to evaluate a little bit further
maxx = np.max(X, axis=0) + 0.6 # the 0.6 value can be adjusted dependeing the size of the bounding box, and if for example you are interested in regions outside the boundaries of the object modelled by the sensors.

print("Expected ((3,), (3,))")
print(minx.shape, maxx.shape)

pcd2 = o3d.geometry.PointCloud()
pcd2.points = o3d.utility.Vector3dVector(v_points)

kdtree = o3d.geometry.KDTreeFlann(pcd2)
x_new = []
y_new = []
for i in range(200):
    p = np.random.uniform() * (maxx - minx) + minx
    # print(p)
    [k, idx, dis_sqr] = kdtree.search_knn_vector_3d(p, 1)
    # print(dis_sqr, p, v_points[idx[0]], np.linalg.norm(p-v_points[idx[0]])**2 )
    val = dis_sqr[0] ** 0.5
    vec1 = v_points[idx[0]]-p
    vec2 = v_normals[idx[0]]
    diff = metric.cosine(vec1,vec2)
    val=diff
    # if val>0.5:
    #     val=0
    # else:
    #     continue
        # val = (1- val)
    x_new.append(p)
    y_new.append(val)

x_new = np.array(x_new)
y_new = np.array(y_new)
x_new = np.concatenate([np.array(x_new), v_points])
y_new = np.concatenate([np.array(y_new), fone.squeeze()])
# X = np.concatenate([X, points_in, points_out])
# y = np.concatenate([fzero, fone, fminus])
print("Expected ((498, 3), (498, 1))")
X = x_new
y = y_new
print(X.shape, y.shape)

# Evaluation limits


# Filling the query vector
scale_x_min = minx[0]
scale_x_max = maxx[0]
scale_y_min = minx[1]
scale_y_max = maxx[1]
scale_z_min = minx[2]
scale_z_max = maxx[2]

xstar = np.zeros((res**3, 3))

for j in range(res):
	for i in range(res):
		d = j * res**2 # Delta
		axis_min = d + res * i
		axis_max = res * (i + 1) + d

		xstar[axis_min:axis_max, 0] = np.linspace(scale_x_min, scale_x_max, num=res) # in X
		xstar[axis_min:axis_max, 1] = scale_y_min + i * ((scale_y_max - scale_y_min) / res) # in X
		xstar[axis_min:axis_max, 2] = scale_z_min + ((j + 1) * ((scale_z_max - scale_z_min) / res))

tsize = res
xeva = np.reshape(xstar[:, 0], (tsize, tsize, tsize))
yeva = np.reshape(xstar[:, 1], (tsize, tsize, tsize))
zeva = np.reshape(xstar[:, 2], (tsize, tsize, tsize))

print("Expected ((100, 100, 100), (100, 100, 100), (100, 100, 100))")
print(xeva.shape, yeva.shape, zeva.shape)

# GP Setup
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        #self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RQKernel())
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# ker = GPy.kern.Exponential(3)
# ker=GPy.kern.RatQuad(3,power=0.8)
X = torch.FloatTensor(X).cuda()
y = torch.FloatTensor(y).squeeze().cuda()
likelihood = gpytorch.likelihoods.GaussianLikelihood().cuda()
model =ExactGPModel(X,y,likelihood).cuda()
model.train()
likelihood.train()
# Query GP
print("Predicting")
start = time.time()
# xstar  = torch.FloatTensor(xstar)
# prediction, x_star_var = m(xstar)
# x_star_std = np.sqrt(x_star_var)
training_iter = 200
optimizer = torch.optim.Adam([
    {'params': model.parameters()},  # Includes GaussianLikelihood parameters
], lr=0.1)
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

for i in range(training_iter):
    # Zero gradients from previous iteration
    optimizer.zero_grad()
    # Output from model
    output = model(X)
    # Calc loss and backprop gradients
    loss = -mll(output, y)
    loss.backward()
    print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
        i + 1, training_iter, loss.item(),
        model.covar_module.base_kernel.lengthscale.item(),
        model.likelihood.noise.item()
    ))
    optimizer.step()

xstar = torch.FloatTensor(xstar).cuda()

model.eval()
likelihood.eval()

x = torch.FloatTensor(v_points).cuda()
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    observed_pred = likelihood(model(x))
    xtrain_prediction = observed_pred.mean.cpu().numpy()
    # print(pred.shape)
    xtrain_confidence_lower, xtrain_confidence_upper = observed_pred.confidence_region()

_ = time.time()
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    # test_x = torch.linspace(0, 1, 51)
    observed_pred = likelihood(model(xstar))
    prediction = observed_pred.mean.cpu().numpy()
    # print(pred.shape)
    confidence_lower, confidence_upper = observed_pred.confidence_region()
    # print(region.shape)
print(time.time()-_)
print("grid loop")
output_grid = np.zeros((res, res, res))
for counter, (i, j, k) in enumerate(itertools.product(range(res), range(res), range(res))):
    output_grid[i][j][k] = prediction[counter]

print("statistical data")
print(np.min(prediction))
print(np.max(prediction))
x_star_std = np.std(prediction)

print("plotting")
fig = go.Figure(data=
            [go.Isosurface(
                x=xeva.flatten(),
                y=yeva.flatten(),
                z=zeva.flatten(),
                value=prediction.flatten(),
                isomin=-np.std(xtrain_prediction)/4 + np.mean(xtrain_prediction),
                isomax=np.std(xtrain_prediction)/4 + np.mean(xtrain_prediction),
                caps=dict(x_show=False, y_show=False)
            ),
            go.Scatter3d(
                x=v_points[:,0],
                y=v_points[:,1],
                z=v_points[:,2],
                mode='markers',
                marker=dict(color="blue")
            ),
            # go.Scatter3d(
            #     x=points_in[:,0],
            #     y=points_in[:,1],
            #     z=points_in[:,2],
            #     mode='markers',
            #     marker=dict(color="green")
            # ),
            # go.Scatter3d(
            #     x=points_out[:,0],
            #     y=points_out[:,1],
            #     z=points_out[:,2],
            #     mode='markers',
            #     marker=dict(color="red")
            # )
            ]
    )
fig.show()