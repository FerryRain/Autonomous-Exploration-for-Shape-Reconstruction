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
from gpytorch.kernels import Kernel
from gpytorch.constraints import GreaterThan # Import the constraint
# import mcubes
# import plyfile
# import compute_depth_normals
# import compute_tactile_normals

n_pts=128
d = 0.2
d_pos = 0.05
d_neg = 0.05 # 0.2
npar = 0.001 # 0.03
training_iter = 40

display_percentile_low=0
display_percentile_high=100-display_percentile_low

# Grid resolution
res = 90 # 150 # grid resolution # 50
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

# Place matlab in the working directory


pcd = o3d.io.read_point_cloud("bunny_ascii.pcd")
if not pcd.has_normals():
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pcd.orient_normals_towards_camera_location(camera_location=np.asarray(pcd.points).mean(axis=0))
v_points = np.asarray(pcd.points)# scale if needed.
v_normals = np.asarray(pcd.normals)

# points = np.load(f"D:\Code\AMMH\Env\Results\Outer\object_0\merged\merged_x_4.npy")
# val = np.load(f"D:\Code\AMMH\Env\Results\Outer\object_0\merged\merged_y_4.npy")
# points = np.load(f"D:\Code\AMMH\Env\Results\RealExp_1\Realexp1_irregular\explored_x_120.npy")
# val = np.load(f"D:\Code\AMMH\Env\Results\RealExp_1\Realexp1_irregular\explored_y_120.npy")
# points = np.load(f"D:\Code\AMMH\Env\Results\Realexp_2\Real_exp_2(sphere)\explored_x_110.npy")
# val = np.load(f"D:\Code\AMMH\Env\Results\Realexp_2\Real_exp_2(sphere)\explored_y_110.npy")
# v_points = points[val==0]

# centroid = np.mean(v_points, axis=0)
# centroid[0] = 1.2
# centroid[1] = -0.6
# num_samples = min(200, len(v_points))
# sample_indices = np.random.choice(len(v_points), num_samples, replace=False)
# v_points = v_points[sample_indices]

# directions = v_points - centroid  # (N, 3)

# norms = np.linalg.norm(directions, axis=1, keepdims=True)
# unit_dirs = directions / (norms + 1e-8)  # 避免除零





## Prepare Data (Computing Constraints) % Assuming normals are correct ...
# Parameters can be computed automatically as a function of the size of the object(e.g. 10% of the size defined by a bounding box, see (Wendland, 2002, Surface Reconstructions from unorganized point clouds).

## Computing inside and outside constraints based on normals
normals = v_normals # D_frompcd.calc_normals(0.1, 20)
points = v_points # D_frompcd.to_array()

points_out = points + d_neg * normals
points_in = points - d_pos * normals
# points_out = v_points + 5.0 * unit_dirs
# points_in = v_points - 5.0 * unit_dirs


# print("Expected ((166, 3), (166, 3), (166, 3))")
# print(points.shape, points_in.shape, points_out.shape)

## Prepare f(x) as signace distance function
# fone=ones(1,size(points_in',1))*1;
# fminus=-1*ones(1,size(points_out',1))*1;
fone   = np.ones((points_out.shape[0], 1)) #* d_pos
fminus = -1 * np.ones((points_in.shape[0], 1))# * d_neg
X      = points
fzero  = np.zeros((X.shape[0], 1))
print("Expected ((166, 1), (166, 1), (166, 3), (166, 1))")
# print(fone.shape, fminus.shape, X.shape, fzero.shape)

## Visualize Object (Cube)
# Notice that the scale of the Sphere goes from -20 to 20
# scatter3D([points, points_out, points_in], ['r', 'g', 'b'])
# exit()

# Training data
X_0 = np.concatenate([X])
y_0 = np.concatenate([fzero])
X_0 = torch.FloatTensor(X_0).cuda()
y_0 = torch.FloatTensor(y_0).squeeze().cuda()
X = np.concatenate([X, points_out])
y = np.concatenate([fzero, fone])
# X = np.concatenate([X, points_in, points_out])
# y = np.concatenate([fzero, fminus, fone])
print("Expected ((498, 3), (498, 1))")
print(X.shape, y.shape)

# Evaluation limits
minx = np.min(X, axis=0) - 0.6 # We extend the boundaries of the object a bit to evaluate a little bit further
maxx = np.max(X, axis=0) + 0.6 # the 0.6 value can be adjusted dependeing the size of the bounding box, and if for example you are interested in regions outside the boundaries of the object modelled by the sensors.

print("Expected ((3,), (3,))")
print(minx.shape, maxx.shape)

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

print("Expected ((90, 90, 90), (90, 90, 90), (90, 90, 90))")
print(xeva.shape, yeva.shape, zeva.shape)

class SqrtMultiquadricKernel(Kernel):
    has_lengthscale = True

    def __init__(self, **kwargs):
        # We enforce that the lengthscale must be greater than a small number to avoid division by zero
        super(SqrtMultiquadricKernel, self).__init__(
            lengthscale_constraint=GreaterThan(1e-4), **kwargs
        )

    def forward(self, x1, x2, diag=False, **params):
        dist_sq = self.covar_dist(x1, x2, square_dist=True, diag=diag)
        # Using the new inverse multiquadric formula from your code
        res = 1 / torch.sqrt(dist_sq + self.lengthscale)
        return res

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.mean_module.initialize(constant=1.)
        # Initialize the kernel without the incorrect argument
        self.covar_module = gpytorch.kernels.ScaleKernel(SqrtMultiquadricKernel())

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
model.likelihood.noise = 1e-2
model.covar_module.base_kernel.lengthscale = 0.2 # Set to 0.2 as requested
model.train()
likelihood.train()
# Query GP
print("Predicting")
start = time.time()
# xstar  = torch.FloatTensor(xstar)
# prediction, x_star_var = m(xstar)
# x_star_std = np.sqrt(x_star_var)

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

with torch.no_grad(), gpytorch.settings.fast_pred_var():
    x = X_0
    observed_pred = likelihood(model(x))
    original_prediction = observed_pred.mean.cpu().numpy()
    # original_confidence_lower, original_confidence_upper = observed_pred.confidence_region()

original_pred_low, original_pred_high = np.percentile(original_prediction, [display_percentile_low, display_percentile_high])
print(original_pred_low, original_pred_high)
_ = time.time()
prediction_buf = []
uncertainty_buf = []
isplit=0
step_length = 10000
model.eval()
likelihood.eval()
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    # test_x = torch.linspace(0, 1, 51)
    while isplit*step_length<xstar.shape[0]:
        isplit += 1
        split_min = step_length * (isplit-1)
        split_max = np.minimum(step_length * isplit, xstar.shape[0])

        # xstar_tensor = torch.FloatTensor(xstar[split_min:split_max,:]).cuda()
        observed_pred = likelihood(model(xstar[split_min:split_max,:]))
        prediction = observed_pred.mean.cpu().numpy()
        prediction_buf.append(prediction)
        # confidence_lower, confidence_upper = observed_pred.confidence_region()
        # confidence_lower = confidence_lower.cpu().numpy()
        # confidence_upper = confidence_upper.cpu().numpy()
        # uncertainty_buf.append(confidence_upper - confidence_lower)


# with torch.no_grad(), gpytorch.settings.fast_pred_var():
#     observed_pred = likelihood(model(xstar))
#     prediction = observed_pred.mean.cpu().numpy()
# BUG FIX: The following line caused a RuntimeError and is not needed for the plot.
# confidence_lower, confidence_upper = observed_pred.confidence_region()
print(time.time()-_)
prediction = np.concatenate([np.asarray(x) for x in prediction_buf], axis=0)
print("grid loop")
output_grid = np.zeros((res, res, res))
for counter, (i, j, k) in enumerate(itertools.product(range(res), range(res), range(res))):
    output_grid[i][j][k] = prediction[counter]

print("statistical data")
print(np.min(prediction))
print(np.max(prediction))
x_star_std = np.std(prediction)

# This block of code was also unnecessary as it was only used to set isomin/isomax,
# which are now correctly set to 0.0 for an implicit surface.
# with torch.no_grad(), gpytorch.settings.fast_pred_var():
#     x = torch.FloatTensor(v_points).cuda()
#     observed_pred = likelihood(model(x))
#     original_prediction = observed_pred.mean.cpu().numpy()
#     original_confidence_lower, original_confidence_upper = observed_pred.confidence_region()
# original_pred_low, original_pred_high = np.percentile(original_prediction, [0, 100])

# mask = (prediction ==0)
mask = (prediction >=-0.0001) & (prediction  <=0.0001)
# mask = (prediction > original_pred_low) & (prediction < original_pred_high)
# mask = (prediction > 1-0.01) & (prediction < 1+0.01)
estimated_surface = xstar[mask].cpu().numpy()

import open3d as o3d



point_cloud = o3d.geometry.PointCloud()

estimated_surface = estimated_surface[estimated_surface[:, -1] <=20]
estimated_surface = estimated_surface[estimated_surface[:, -1] >=-20]
point_cloud.points = o3d.utility.Vector3dVector(estimated_surface)
o3d.visualization.draw_geometries([point_cloud])
np.save("Diress_normal.npy", estimated_surface)


draw_points = points

# draw_points = points[points[:, -1] <=2]
# draw_points = draw_points[draw_points[:, -1] >=-2]
print("plotting")
fig = go.Figure(data=
[
    go.Isosurface(
        x=xeva.flatten(),
        y=yeva.flatten(),
        z=zeva.flatten(),
        value=prediction.flatten(),
        isomin=0.0,
        isomax=0.0,
        caps=dict(x_show=False, y_show=False),
        # colorscale='RdBu',
        surface= dict(show=True,count=1, fill=0.7),

    ),
    # go.Scatter3d(
    #     x=draw_points[:,0],
    #     y=draw_points[:,1],
    #     z=draw_points[:,2],
    #     mode='markers',
    #     marker=dict(color="blue", size=2)
    # ),

    go.Scatter3d(
        x=points[:,0],
        y=points[:,1],
        z=points[:,2],
        mode='markers',
        marker=dict(color="blue", size=2)
    ),
    # go.Scatter3d(
    #     x=points_in[:,0],
    #     y=points_in[:,1],
    #     z=points_in[:,2],
    #     mode='markers',
    #     marker=dict(color="green", size=2)
    # ),
    go.Scatter3d(
        x=points_out[:,0],
        y=points_out[:,1],
        z=points_out[:,2],
        mode='markers',
        marker=dict(color="red", size=2)
    )
]
)
# fig.update_layout(
#     scene=dict(
#         zaxis=dict(range=[-20, 20]),    # 限制 z 轴范围
#         xaxis=dict(range=[xeva.min(), xeva.max()]),  # 可选：限制 x 范围
#         yaxis=dict(range=[yeva.min(), yeva.max()]),  # 可选：限制 y 范围
#         aspectmode='data'  # 保持比例一致
#     )
# )
import plotly.io as pio
pio.renderers.default = "browser"    # 或 "chrome"/"firefox"/"safari"
fig.show()
