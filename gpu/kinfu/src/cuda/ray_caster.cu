/*
 * Software License Agreement (BSD License)
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2011, Willow Garage, Inc.
 *
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Willow Garage, Inc. nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 */
#include "device.hpp"
#include <stdio.h>
namespace pcl
{
  namespace device
  {
    __device__ __forceinline__ float
    getMinTime (const float3& volume_max, const float3& origin, const float3& dir, const float3& cell_size, const int3& shift)
    {
      float txmin = ( (dir.x > 0 ? 0.f + shift.x*cell_size.x: volume_max.x + shift.x*cell_size.x) - origin.x) / dir.x;
      float tymin = ( (dir.y > 0 ? 0.f + shift.y*cell_size.y: volume_max.y + shift.y*cell_size.y) - origin.y) / dir.y;
      float tzmin = ( (dir.z > 0 ? 0.f + shift.z*cell_size.z: volume_max.z + shift.z*cell_size.z) - origin.z) / dir.z;

      return fmax ( fmax (txmin, tymin), tzmin);
    }

    __device__ __forceinline__ float
    getMaxTime (const float3& volume_max, const float3& origin, const float3& dir, const float3& cell_size, const int3& shift)
    {
      float txmax = ( (dir.x > 0 ? volume_max.x + shift.x*cell_size.x: 0.f + shift.x*cell_size.x) - origin.x) / dir.x;
      float tymax = ( (dir.y > 0 ? volume_max.y + shift.y*cell_size.y: 0.f + shift.y*cell_size.y) - origin.y) / dir.y;
      float tzmax = ( (dir.z > 0 ? volume_max.z + shift.z*cell_size.z: 0.f + shift.z*cell_size.z) - origin.z) / dir.z;

      return fmin (fmin (txmax, tymax), tzmax);
    }

    struct RayCaster
    {
      enum { CTA_SIZE_X = 32, CTA_SIZE_Y = 8 };

      Mat33 Rcurr;
      float3 tcurr;

      float time_step;
      float3 volume_size;

      float3 cell_size;
      int cols, rows;

      PtrStep<short2> volume;

      Intr intr;

      int3 shift;

      mutable PtrStep<float> nmap;
      mutable PtrStep<float> vmap;
      mutable PtrStep<int3> ray_cubes;

      PtrStepSz<ushort> depth_raw; //depth in mm
      float tranc_dist;

      bool first;

      __device__ __forceinline__ float3
      get_ray_next (int x, int y) const
      {
        float3 ray_next;
        ray_next.x = (x - intr.cx) / intr.fx;
        ray_next.y = (y - intr.cy) / intr.fy;
        ray_next.z = 1;
        return ray_next;
      }

      __device__ __forceinline__ bool
      checkInds (const int3& g) const
      {
        return (g.x >= 0 && g.y >= 0 && g.z >= 0 && g.x < VOLUME_X && g.y < VOLUME_Y && g.z < VOLUME_Z);
      }

      __device__ __forceinline__ bool
      checkFeasible (const int3& g) const
      {
        int minx = min(0, -shift.x);
        int miny = min(0, -shift.y);
        int minz = min(0, -shift.z);
        int maxx = max(VOLUME_X-1, -shift.x);
        int maxy = max(VOLUME_Y-1, -shift.y);
        int maxz = max(VOLUME_Z-1, -shift.z);
        return (g.x >= minx && g.y >= miny && g.z >= minz && g.x <= maxx && g.y <= maxy && g.z <= maxz);
      }

      __device__ __forceinline__ float
      readTsdf (int x, int y, int z) const
      {
        return unpack_tsdf (volume.ptr (VOLUME_Y * z + y)[x]);
      }

      __device__ __forceinline__ int3
      getVoxel (float3 point) const
      {
        int vx = __float2int_rd (point.x / cell_size.x);        // round to negative infinity
        int vy = __float2int_rd (point.y / cell_size.y);
        int vz = __float2int_rd (point.z / cell_size.z);

        return make_int3 (vx - shift.x, vy - shift.y, vz - shift.z);
      }

      __device__ __forceinline__ void
      getCubes () const
      {
        int col = threadIdx.x + blockIdx.x * CTA_SIZE_X;
        int row = threadIdx.y + blockIdx.y * CTA_SIZE_Y;

        if (col >= cols || row >= rows)
          return;

        float3 ray_start = tcurr;
        float3 ray_next = Rcurr * get_ray_next (col, row) + tcurr;

        float3 ray_dir = ray_next - ray_start;

        //depth in mm
        int depth = depth_raw.ptr (row)[col];
        if (depth < 4000 && depth > .0001)
        {
          float zx = ray_dir.z/ray_dir.x;
          float yx = ray_dir.y/ray_dir.x;
          float x = __int2float_rn(depth)/sqrt(zx*zx + yx*yx+1);
          x = ray_dir.x >= 0 ? x : -x;
          float y = yx*x;
          float z = zx*x;
          //cell_size in m
          //tcurr inm
          ray_cubes.ptr(row)[col].x = __float2int_rd((x + tcurr.x*1000)/(cell_size.x*1000)/(VOLUME_X - 7))*(VOLUME_X - 7);
          ray_cubes.ptr(row)[col].y = __float2int_rd((y + tcurr.y*1000)/(cell_size.y*1000)/(VOLUME_Y - 7))*(VOLUME_Y - 7);
          ray_cubes.ptr(row)[col].z = __float2int_rd((z + tcurr.z*1000)/(cell_size.z*1000)/(VOLUME_Z - 7))*(VOLUME_Z - 7);
        }
        else
        {
          ray_cubes.ptr(row)[col].x = -1;
          ray_cubes.ptr(row)[col].y = -1;
          ray_cubes.ptr(row)[col].z = -1;
        }
      }


      __device__ __forceinline__ float
      interpolateTrilineary (const float3& origin, const float3& dir, float time) const
      {
        return interpolateTrilineary (origin + dir * time);
      }

      __device__ __forceinline__ float
      interpolateTrilineary (const float3& point) const
      {
        int3 g = getVoxel (point);

        if (g.x <= 0 || g.x >= VOLUME_X - 1)
          return numeric_limits<float>::quiet_NaN ();

        if (g.y <= 0 || g.y >= VOLUME_Y - 1)
          return numeric_limits<float>::quiet_NaN ();

        if (g.z <= 0 || g.z >= VOLUME_Z - 1)
          return numeric_limits<float>::quiet_NaN ();

        float vx = (g.x + shift.x + 0.5f) * cell_size.x;
        float vy = (g.y + shift.y + 0.5f) * cell_size.y;
        float vz = (g.z + shift.z + 0.5f) * cell_size.z;

        g.x = (point.x < vx) ? (g.x - 1) : g.x;
        g.y = (point.y < vy) ? (g.y - 1) : g.y;
        g.z = (point.z < vz) ? (g.z - 1) : g.z;

        float a = (point.x - (g.x + shift.x + 0.5f) * cell_size.x) / cell_size.x;
        float b = (point.y - (g.y + shift.y + 0.5f) * cell_size.y) / cell_size.y;
        float c = (point.z - (g.z + shift.z + 0.5f) * cell_size.z) / cell_size.z;

        float res = readTsdf (g.x + 0, g.y + 0, g.z + 0) * (1 - a) * (1 - b) * (1 - c) +
                    readTsdf (g.x + 0, g.y + 0, g.z + 1) * (1 - a) * (1 - b) * c +
                    readTsdf (g.x + 0, g.y + 1, g.z + 0) * (1 - a) * b * (1 - c) +
                    readTsdf (g.x + 0, g.y + 1, g.z + 1) * (1 - a) * b * c +
                    readTsdf (g.x + 1, g.y + 0, g.z + 0) * a * (1 - b) * (1 - c) +
                    readTsdf (g.x + 1, g.y + 0, g.z + 1) * a * (1 - b) * c +
                    readTsdf (g.x + 1, g.y + 1, g.z + 0) * a * b * (1 - c) +
                    readTsdf (g.x + 1, g.y + 1, g.z + 1) * a * b * c;
        return res;
      }

      __device__ __forceinline__ void
      operator () () const
      {
        int x = threadIdx.x + blockIdx.x * CTA_SIZE_X;
        int y = threadIdx.y + blockIdx.y * CTA_SIZE_Y;

        if (x >= cols || y >= rows)
          return;

        if (first) {
          vmap.ptr (y)[x] = numeric_limits<float>::quiet_NaN ();
          nmap.ptr (y)[x] = numeric_limits<float>::quiet_NaN ();
          vmap.ptr (y + rows)[x] = numeric_limits<float>::quiet_NaN ();
          nmap.ptr (y + rows)[x] = numeric_limits<float>::quiet_NaN ();
          vmap.ptr (y + 2*rows)[x] = numeric_limits<float>::quiet_NaN ();
          nmap.ptr (y + 2*rows)[x] = numeric_limits<float>::quiet_NaN ();
        }

        float3 ray_start = tcurr;
        float3 ray_next = Rcurr * get_ray_next (x, y) + tcurr;

        float3 ray_dir = normalized (ray_next - ray_start);

        //ensure that it isn't a degenerate case
        ray_dir.x = (ray_dir.x == 0.f) ? 1e-15 : ray_dir.x;
        ray_dir.y = (ray_dir.y == 0.f) ? 1e-15 : ray_dir.y;
        ray_dir.z = (ray_dir.z == 0.f) ? 1e-15 : ray_dir.z;

        // computer time when entry and exit volume
        float time_start_volume = getMinTime (volume_size, ray_start, ray_dir, cell_size, shift);
        float time_exit_volume = getMaxTime (volume_size, ray_start, ray_dir, cell_size, shift);

        const float min_dist = 0.f;         //in meters
        time_start_volume = fmax (time_start_volume, min_dist);
        if (time_start_volume >= time_exit_volume)
          return;

        float time_curr = time_start_volume;
        int3 g = getVoxel (ray_start + ray_dir * time_curr);
        g.x = max (0, min (g.x, VOLUME_X - 1));
        g.y = max (0, min (g.y, VOLUME_Y - 1));
        g.z = max (0, min (g.z, VOLUME_Z - 1));

        float tsdf = readTsdf (g.x, g.y, g.z);

        //infinite loop guard
        // changed
        const float max_time = 15 * (volume_size.x + volume_size.y + volume_size.z);
        for (; time_curr < max_time; time_curr += time_step)
        {
          float tsdf_prev = tsdf;

          int3 g = getVoxel (  ray_start + ray_dir * (time_curr + time_step)  );
          //if (!checkFeasible(g))
          //  break;
          if (!checkInds (g))
              break;
          //  continue;
          
          tsdf = readTsdf (g.x, g.y, g.z);

          if (tsdf_prev < 0.f && tsdf > 0.f)
            break;

          if (tsdf_prev > 0.f && tsdf < 0.f)           //zero crossing
          {
            float Ftdt = interpolateTrilineary (ray_start, ray_dir, time_curr + time_step);
            if (isnan (Ftdt))
              break;

            float Ft = interpolateTrilineary (ray_start, ray_dir, time_curr);
            if (isnan (Ft))
              break;

            //float Ts = time_curr - time_step * Ft/(Ftdt - Ft);
            float Ts = time_curr - time_step * Ft / (Ftdt - Ft);

            float3 vetex_found = ray_start + ray_dir * Ts;
            float3 prev_vetex;
            prev_vetex.x = vmap.ptr (y       )[x];
            prev_vetex.y = vmap.ptr (y + rows)[x];
            prev_vetex.z = vmap.ptr (y + 2 * rows)[x];
            
            bool valid_ray = false;
            if (isnan(prev_vetex.x)) {
              valid_ray = true;
            }
            else {
              float3 test_vetex = vetex_found - prev_vetex;
              float magnitude = test_vetex.x*test_vetex.x + test_vetex.y*test_vetex.y + test_vetex.z + test_vetex.z;
              if (magnitude > 0) valid_ray = true;
            }
            //printf("valid ray: %d\n", valid_ray);
            if (valid_ray) {
              vmap.ptr (y       )[x] = vetex_found.x;
              vmap.ptr (y + rows)[x] = vetex_found.y;
              vmap.ptr (y + 2 * rows)[x] = vetex_found.z;

              int3 g = getVoxel ( ray_start + ray_dir * time_curr );
              if (g.x > 1 && g.y > 1 && g.z > 1 && g.x < VOLUME_X - 3 && g.y < VOLUME_Y - 3 && g.z < VOLUME_Z - 3)
              {
                float3 t;
                float3 n;

                t = vetex_found;
                t.x += cell_size.x;
                float Fx1 = interpolateTrilineary (t);

                t = vetex_found;
                t.x -= cell_size.x;
                float Fx2 = interpolateTrilineary (t);

                n.x = (Fx1 - Fx2);

                t = vetex_found;
                t.y += cell_size.y;
                float Fy1 = interpolateTrilineary (t);

                t = vetex_found;
                t.y -= cell_size.y;
                float Fy2 = interpolateTrilineary (t);

                n.y = (Fy1 - Fy2);

                t = vetex_found;
                t.z += cell_size.z;
                float Fz1 = interpolateTrilineary (t);

                t = vetex_found;
                t.z -= cell_size.z;
                float Fz2 = interpolateTrilineary (t);

                n.z = (Fz1 - Fz2);

                n = normalized (n);

                nmap.ptr (y       )[x] = n.x;
                nmap.ptr (y + rows)[x] = n.y;
                nmap.ptr (y + 2 * rows)[x] = n.z;
              }
            }
            break;
          }

        }          /* for(;;)  */


      }
    };

    __global__ void
    rayCastKernel (const RayCaster rc) {
      rc ();
    }

    __global__ void
    findCubeKernel (const RayCaster rc) {
      rc.getCubes();
    }
  }
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
pcl::device::raycast (const Intr& intr, const Mat33& Rcurr, const float3& tcurr, 
                      float tranc_dist, const float3& volume_size,
                      const PtrStep<short2>& volume,
                      MapArr& vmap, MapArr& nmap)
{
  RayCaster rc;

  rc.Rcurr = Rcurr;
  rc.tcurr = tcurr;

  rc.time_step = tranc_dist * 0.8f;

  rc.volume_size = volume_size;

  rc.cell_size.x = volume_size.x / VOLUME_X;
  rc.cell_size.y = volume_size.y / VOLUME_Y;
  rc.cell_size.z = volume_size.z / VOLUME_Z;

  rc.cols = vmap.cols ();
  rc.rows = vmap.rows () / 3;

  rc.intr = intr;

  rc.volume = volume;
  rc.vmap = vmap;
  rc.nmap = nmap;

  rc.first = true;

  dim3 block (RayCaster::CTA_SIZE_X, RayCaster::CTA_SIZE_Y);
  dim3 grid (divUp (rc.cols, block.x), divUp (rc.rows, block.y));

  rayCastKernel<<<grid, block>>>(rc);
  cudaSafeCall (cudaGetLastError ());
  //cudaSafeCall(cudaDeviceSynchronize());
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
pcl::device::raycast (const Intr& intr, const Mat33& Rcurr, const float3& tcurr, 
                      float tranc_dist, const float3& volume_size,
                      const PtrStep<short2>& volume, const int3& shift, 
                      MapArr& vmap, MapArr& nmap, bool first)
{
  RayCaster rc;

  rc.Rcurr = Rcurr;
  rc.tcurr = tcurr;

  rc.time_step = tranc_dist * 0.8f;

  rc.volume_size = volume_size;

  rc.cell_size.x = volume_size.x / VOLUME_X;
  rc.cell_size.y = volume_size.y / VOLUME_Y;
  rc.cell_size.z = volume_size.z / VOLUME_Z;

  rc.cols = vmap.cols ();
  rc.rows = vmap.rows () / 3;

  rc.intr = intr;

  rc.shift = shift;

  rc.volume = volume;
  rc.vmap = vmap;
  rc.nmap = nmap;

  rc.first = first;

  dim3 block (RayCaster::CTA_SIZE_X, RayCaster::CTA_SIZE_Y);
  dim3 grid (divUp (rc.cols, block.x), divUp (rc.rows, block.y));

  rayCastKernel<<<grid, block>>>(rc);
  cudaSafeCall (cudaGetLastError ());
  //cudaSafeCall(cudaDeviceSynchronize());
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
pcl::device::generateNumCubeRays(const Intr& intr, const Mat33& Rcurr, const float3& tcurr, const float3& volume_size, 
              const PtrStepSz<ushort>& depth_raw, int rows, int cols, DeviceArray2D<int3>& ray_cubes)
{
  RayCaster rc;

  rc.Rcurr = Rcurr;
  rc.tcurr = tcurr;

  rc.volume_size = volume_size;

  rc.cell_size.x = volume_size.x / VOLUME_X;
  rc.cell_size.y = volume_size.y / VOLUME_Y;
  rc.cell_size.z = volume_size.z / VOLUME_Z;

  rc.cols = cols;
  rc.rows = rows;

  rc.intr = intr;

  rc.ray_cubes = ray_cubes;

  rc.depth_raw = depth_raw;

  dim3 block (RayCaster::CTA_SIZE_X, RayCaster::CTA_SIZE_Y);
  dim3 grid (divUp (rc.cols, block.x), divUp (rc.rows, block.y));

  findCubeKernel<<<grid, block>>>(rc);
  cudaSafeCall (cudaGetLastError ());
  //cudaSafeCall(cudaDeviceSynchronize());
}