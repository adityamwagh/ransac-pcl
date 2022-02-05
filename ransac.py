import math

import numpy as np
import open3d as o3d
import plotly.graph_objects as go


def ransac(data, sample_size=3, threshold=0.05, iterations=1000):
    """
    RANSAC algorithm to detect outliers and fit a 3D plane
    :param data: numpy array of shape (N, 3)
    :param threshold: threshold for inliers
    :param iterations: number of iterations
    :return: numpy array of shape (1, 4) containing parameters of
             the best fit plane and the (M, 3) array of inliers
    """

    # numpy arrays to store inliers and parameters
    best_inliers = np.array([])
    best_plane_parameters = np.array([])

    for _ in range(iterations):

        # randomly sample 3 points
        i, j, k = np.random.randint(0, data.shape[0] - 1, sample_size)
        p1, p2, p3 = data[i], data[j], data[k]
        x1, y1, z1 = p1[0], p1[1], p1[2]
        x2, y2, z2 = p2[0], p2[1], p2[2]
        x3, y3, z3 = p3[0], p3[1], p3[2]

        # compute plane parameters - ax + by + cz + d = 0
        a = (y2 - y1) * (z3 - z1) - (z2 - z1) * (y3 - y1)
        b = (z2 - z1) * (x3 - x1) - (x2 - x1) * (z3 - z1)
        c = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
        d = -(a * x1 + b * y1 + c * z1)
        current_plane_parameters = [a, b, c, d]

        # compute distance of a point to the plane
        norm_of_plane_normal = math.sqrt(a * a + b * b + c * c)
        distances = (
            np.abs(a * data[:, 0] + b * data[:, 1] + c * data[:, 2] + d)
            / norm_of_plane_normal
        )

        # Select points where distance is bigger than the threshold
        inlier_index = np.argwhere(distances <= threshold)
        current_inliers = data[inlier_index]
        if len(current_inliers) > len(best_inliers):
            best_inliers = current_inliers
            best_plane_parameters = current_plane_parameters

    return best_plane_parameters, best_inliers


# read and print data using open3d
data = o3d.io.read_point_cloud("record_00348.pcd")
print(f"Data : {repr(np.asarray(data.points))}")

# convert data to numpy array
data = np.asarray(data.points)

# Fit a plane to the dataset using RANSAC with a threshold of 0.05 and for 100 iterations
params, inliers = ransac(data, threshold=0.01, iterations=1000)
print(f"Plane Parameters : {repr(params)}")
print(f"Inliers : {inliers, inliers.shape}")
print(f"Inlier ratio : {repr(inliers.shape[0]/ data.shape[0])}")

# Setup plane parameters to visualize the plane
X, Y = np.meshgrid(*np.array([data.min(axis=0), data.max(axis=0)])[:, :2].T)
Z = -(params[0] / params[2]) * X - (params[1] / params[2]) * Y - params[3] / params[2]

# plot the best fit plane along with the point cloud data
fig = go.Figure(
    [
        go.Scatter3d(
            x=data[:, 0],
            y=data[:, 1],
            z=data[:, 2],
            mode="markers",
            showlegend=False,
            marker=dict(color=None, size=2.5, line=dict(color="green", width=1)),
        ),
        go.Surface(
            x=X,
            y=Y,
            z=Z,
            opacity=0.5,
            showscale=False,
            surfacecolor=np.zeros(Z.shape),
            colorscale=[[0, "grey"]],
        ),
    ]
)

# show the plot
fig.show()

# write the plot to a jpg file
fig.write_image("plot.jpg")
