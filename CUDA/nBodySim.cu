#include <stdio.h>
#include <stdlib.h>

#define EPS2 0.00000001
#define N 100

// Global variables
__device__ float4 globalX[N];
__device__ float4 globalV[N];
__device__ float4 globalA[N];

// Function prototypes
__global__ void calculate_forces(float4 globalX[N], float4 globalA[N]);
__device__ float3 tile_calculation(float4 myPosition, float3 accel);
__device__ float3 bodyBodyInteraction(float4 bi, float4 bj, float3 ai);

// Kernel definitions
__global__ void calculate_forces(float4 globalX[N], float4 globalA[N])
{
  // A shared memory buffer to store the body positions.
  __shared__ float4 shPosition[N];
   
  float4 myPosition;
  int i, tile;
 
  float3 acc = {0.0f, 0.0f, 0.0f};
  // Global thread ID (represent the unique body index in the simulation) 
  int gtid = blockIdx.x * blockDim.x + threadIdx.x;
 
  // This is the position of the body we are computing the acceleration for.
  myPosition = globalX[gtid];
 
  for (i = 0, tile = 0; i < N; i += blockDim.x, tile++) 
  {
    int idx = tile * blockDim.x + threadIdx.x;
 
    // Each thread in the block participates in populating the shared memory buffer.
    shPosition[threadIdx.x] = globalX[idx];
    // Synchronize guarantees all thread in the block have updated the shared memory
    // buffer.
    __syncthreads();
 
    acc = tile_calculation(myPosition, acc);
 
    // Synchronize again to make sure all threads have used the shared memory
    // buffer before we overwrite the values for the next tile.
    __syncthreads();
  }
   
  // Save the total acceleration in global memory for the integration step.
  float4 acc4 = {acc.x, acc.y, acc.z, 0.0f};
  globalA[gtid] = acc4;
}

// tile_calculation
__device__ float3 tile_calculation(float4 myPosition, float3 accel)
{
  int i;
  __shared__ float4 shPosition[N];
 
  for (i = 0; i < blockDim.x; i++) 
  {
    accel = bodyBodyInteraction(myPosition, shPosition[i], accel);
  }
   
  return accel;
}

// Body Interaction
__device__ float3 bodyBodyInteraction(float4 bi, float4 bj, float3 ai)
{
  float3 r;
 
  // r_ij  [3 FLOPS]
  r.x = bj.x - bi.x;
  r.y = bj.y - bi.y;
  r.z = bj.z - bi.z;
 
  // distSqr = dot(r_ij, r_ij) + EPS^2  [6 FLOPS]
  float distSqr = r.x * r.x + r.y * r.y + r.z * r.z + EPS2;
 
  // invDistCube =1/distSqr^(3/2)  [4 FLOPS (2 mul, 1 sqrt, 1 inv)]
  float distSixth = distSqr * distSqr * distSqr;
  float invDistCube = 1.0f/sqrtf(distSixth);
 
  // s = m_j * invDistCube [1 FLOP]
  float s = bj.w * invDistCube;
 
  // a_i =  a_i + s * r_ij [6 FLOPS]
  ai.x += r.x * s;
  ai.y += r.y * s;
  ai.z += r.z * s;
 
  return ai;
}

// Main method
int main(int argc, char* argv[])
{
  //float4* h_A = (float*)malloc(size);





	return 0;
}