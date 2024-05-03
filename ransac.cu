#include <cmath>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iostream>
#include <open3d/Open3D.h>
#include <vector>

__device__ void compute_plane_params(float x1, float y1, float z1, float x2,
                                     float y2, float z2, float x3, float y3,
                                     float z3, float *a, float *b, float *c,
                                     float *d) {
  *a = (y2 - y1) * (z3 - z1) - (z2 - z1) * (y3 - y1);
  *b = (z2 - z1) * (x3 - x1) - (x2 - x1) * (z3 - z1);
  *c = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1);
  *d = -(*a * x1 + *b * y1 + *c * z1);
}

__global__ void ransac_kernel(
    float *data, 
    int N, 
    int sample_size,
    float threshold, int iterations,
    float *best_plane_params, 
    float *inliers,
    int *inlier_count) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= iterations)
    return;

  curandState_t rand_state;
  curand_init(clock64() + idx, 0, 0, &rand_state);

  int i, j, k;
  float a, b, c, d;
  float x1, y1, z1, x2, y2, z2, x3, y3, z3;
  float norm_of_plane_normal;
  int current_inlier_count = 0;

  for (int iter = 0; iter < iterations; iter++) {
    i = curand(&rand_state) % N;
    j = curand(&rand_state) % N;
    k = curand(&rand_state) % N;

    x1 = data[i * 3];
    y1 = data[i * 3 + 1];
    z1 = data[i * 3 + 2];
    x2 = data[j * 3];
    y2 = data[j * 3 + 1];
    z2 = data[j * 3 + 2];
    x3 = data[k * 3];
    y3 = data[k * 3 + 1];
    z3 = data[k * 3 + 2];

    compute_plane_params(x1, y1, z1, x2, y2, z2, x3, y3, z3, &a, &b, &c, &d);

    norm_of_plane_normal = sqrt(a * a + b * b + c * c);
    if (norm_of_plane_normal < 1e-6) continue;  // Avoid division by zero

    current_inlier_count = 0;
    for (int n = 0; n < N; n++) {
      float dist = fabs(a * data[n * 3] + b * data[n * 3 + 1] +
                        c * data[n * 3 + 2] + d) /
                   norm_of_plane_normal;
      if (dist <= threshold) {
        current_inlier_count++;
      }
    }

    if (current_inlier_count > *inlier_count) {
      atomicExch(inlier_count, current_inlier_count);
      best_plane_params[0] = a;
      best_plane_params[1] = b;
      best_plane_params[2] = c;
      best_plane_params[3] = d;

      int inlier_idx = 0;
      for (int n = 0; n < N; n++) {
        float dist = fabs(a * data[n * 3] + b * data[n * 3 + 1] +
                          c * data[n * 3 + 2] + d) /
                     norm_of_plane_normal;
        if (dist <= threshold) {
          inliers[inlier_idx * 3] = data[n * 3];
          inliers[inlier_idx * 3 + 1] = data[n * 3 + 1];
          inliers[inlier_idx * 3 + 2] = data[n * 3 + 2];
          inlier_idx++;
        }
      }
    }
  }
}

void ransac(float *data, int N, int sample_size, float threshold,
            int iterations, float *best_plane_params, float *inliers,
            int *inlier_count) {
  float *dev_data, *dev_best_plane_params, *dev_inliers;
  int *dev_inlier_count;

  cudaMalloc(&dev_data, N * sizeof(float) * 3);
  cudaMalloc(&dev_best_plane_params, 4 * sizeof(float));
  cudaMalloc(&dev_inliers, N * sizeof(float) * 3);
  cudaMalloc(&dev_inlier_count, sizeof(int));

  cudaMemcpy(dev_data, data, N * sizeof(float) * 3, cudaMemcpyHostToDevice);
  cudaMemset(dev_best_plane_params, 0, 4 * sizeof(float));
  cudaMemset(dev_inliers, 0, N * sizeof(float) * 3);
  cudaMemset(dev_inlier_count, 0, sizeof(int));

  int blockSize = 256;
  int numBlocks = (iterations + blockSize - 1) / blockSize;

  ransac_kernel<<<numBlocks, blockSize>>>(dev_data, N, sample_size, threshold,
                                          iterations, dev_best_plane_params,
                                          dev_inliers, dev_inlier_count);

  cudaMemcpy(best_plane_params, dev_best_plane_params, 4 * sizeof(float),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(inlier_count, dev_inlier_count, sizeof(int),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(inliers, dev_inliers, *inlier_count * sizeof(float) * 3,
             cudaMemcpyDeviceToHost);

  cudaFree(dev_data);
  cudaFree(dev_best_plane_params);
  cudaFree(dev_inliers);
  cudaFree(dev_inlier_count);
}

int main() {
  // Read point cloud data from a .pcd file
  open3d::geometry::PointCloud pcd;
  open3d::io::ReadPointCloudFromPCD("record_00348.pcd", pcd);

  // Convert Open3D point cloud to a flat array
  std::vector<float> data(pcd.points_.size() * 3);
  for (size_t i = 0; i < pcd.points_.size(); ++i) {
    data[i * 3] = pcd.points_[i].x();
    data[i * 3 + 1] = pcd.points_[i].y();
    data[i * 3 + 2] = pcd.points_[i].z();
  }

  // Allocate memory for inliers
  float *inliers = (float *)malloc(data.size() * sizeof(float));

  // Call ransac function with the point cloud data
  float best_plane_params[4];
  int inlier_count;
  ransac(data.data(), data.size() / 3, 3, 0.01, 1000, best_plane_params,
         inliers, &inlier_count);

  // Print the results
  std::cout << "Plane Parameters: " << best_plane_params[0] << ", "
            << best_plane_params[1] << ", " << best_plane_params[2] << ", "
            << best_plane_params[3] << std::endl;
  std::cout << "Inlier Count: " << inlier_count << std::endl;

  // Free the memory for inliers
  free(inliers);

  return 0;
}