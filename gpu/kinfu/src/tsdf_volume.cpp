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

#include <pcl/gpu/kinfu/tsdf_volume.h>
#include "internal.h"
#include <algorithm>
#include <Eigen/Core>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/gpu/kinfu/kinfu.h>

using namespace pcl;
using namespace pcl::gpu;
using namespace Eigen;
using pcl::device::device_cast;

static int num_volumes_ = 0;
static DeviceArray<PointXYZ> cloud_buffer_device_;
static DeviceArray<PointNormal> combined_device_;
static DeviceArray<Normal> normals_device_;
static DeviceArray<RGB> point_colors_device_; 

namespace pcl
{
  namespace gpu
  {
    void mergePointNormal (const DeviceArray<PointXYZ>& cloud, const DeviceArray<Normal>& normals, DeviceArray<PointNormal>& output);
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

pcl::gpu::TsdfVolume::TsdfVolume(const Vector3i& resolution) : resolution_(resolution)
{
  int volume_x = resolution_(0);
  int volume_y = resolution_(1);
  int volume_z = resolution_(2);

  volume_.create (volume_y * volume_z, volume_x);

  const Vector3f default_volume_size = Vector3f::Constant (3.f); //meters
  const float    default_tranc_dist  = 0.03f; //meters
  const Vector3i default_shift = Vector3i::Constant(0);

  setSize(default_volume_size);
  setTsdfTruncDist(default_tranc_dist);
  setShift(default_shift);
  volume_downloaded_ = std::vector<int>(volume_x*volume_y*volume_z,0);

  reset();

  //During initialization we do want to reset volumes, so need to make sure reset 
  //isn't called with num_volumes == 1
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

pcl::gpu::TsdfVolume::TsdfVolume(const Vector3i& resolution, bool first) : resolution_(resolution)
{
  int volume_x = resolution_(0);
  int volume_y = resolution_(1);
  int volume_z = resolution_(2);
  volume_.create (volume_y * volume_z, volume_x);

  const Vector3f default_volume_size = Vector3f::Constant (3.f); //meters
  const float    default_tranc_dist  = 0.03f; //meters
  const Vector3i default_shift = Vector3i::Constant(0);

  setSize(default_volume_size);
  setTsdfTruncDist(default_tranc_dist);
  setShift(default_shift);
  volume_downloaded_ = std::vector<int>(volume_x*volume_y*volume_z,0);
  device::initVolume(volume_);
  downloadTsdfAndWeightsInt ();
  volume_.release();
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void
pcl::gpu::TsdfVolume::setSize(const Vector3f& size)
{  
  size_ = size;
  setTsdfTruncDist(tranc_dist_);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void
pcl::gpu::TsdfVolume::setTsdfTruncDist (float distance)
{
  float cx = size_(0) / resolution_(0);
  float cy = size_(1) / resolution_(1);
  float cz = size_(2) / resolution_(2);

  tranc_dist_ = std::max (distance, 2.1f * std::max (cx, std::max (cy, cz)));  

  /*if (tranc_dist_ != distance)
	  PCL_WARN ("Tsdf truncation distance can't be less than 2 * voxel_size. Passed value '%f', but setting minimal possible '%f'.\n", distance, tranc_dist_);*/
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
pcl::gpu::TsdfVolume::setShift (const Vector3i& shift)
{
  shift_ = shift;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

pcl::gpu::DeviceArray2D<int> 
pcl::gpu::TsdfVolume::data() const
{
  return volume_;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

const Eigen::Vector3f&
pcl::gpu::TsdfVolume::getSize() const
{
    return size_;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

const Eigen::Vector3i&
pcl::gpu::TsdfVolume::getResolution() const
{
  return resolution_;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

const Eigen::Vector3f
pcl::gpu::TsdfVolume::getVoxelSize() const
{    
  return size_.array () / resolution_.array().cast<float>();
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

float
pcl::gpu::TsdfVolume::getTsdfTruncDist () const
{
  return tranc_dist_;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

const Eigen::Vector3i&
pcl::gpu::TsdfVolume::getShift () const
{
  return shift_;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

const std::vector<int>&
pcl::gpu::TsdfVolume::getTsdfAndWeightsInt () const
{
  return volume_downloaded_;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void 
pcl::gpu::TsdfVolume::reset()
{
  if (num_volumes_ == 1) {
    device::initVolume(volume_);
  }
  else {
    uploadTsdfAndWeightsInt();
    device::initVolume(volume_);
    downloadTsdfAndWeightsInt ();
    volume_.release();
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void
pcl::gpu::TsdfVolume::release()
{
  volume_.release();
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void
pcl::gpu::TsdfVolume::fetchCloudHost (PointCloud<PointType>& cloud, bool connected26) const
{
  int volume_x = resolution_(0);
  int volume_y = resolution_(1);
  int volume_z = resolution_(2);

  int cols;
  std::vector<int> volume_host;
  volume_.download (volume_host, cols);

  cloud.points.clear ();
  cloud.points.reserve (10000);

  const int DIVISOR = device::DIVISOR; // SHRT_MAX;

#define FETCH(x, y, z) volume_host[(x) + (y) * volume_x + (z) * volume_y * volume_x]

  Array3f cell_size = getVoxelSize();

  for (int x = 1; x < volume_x-1; ++x)
  {
    for (int y = 1; y < volume_y-1; ++y)
    {
      for (int z = 0; z < volume_z-1; ++z)
      {
        int tmp = FETCH (x, y, z);
        int W = reinterpret_cast<short2*>(&tmp)->y;
        int F = reinterpret_cast<short2*>(&tmp)->x;

        if (W == 0 || F == DIVISOR)
          continue;

        Vector3f V = ((Array3i(x, y, z).cast<float>() + 0.5f) * cell_size).matrix ();

        if (connected26)
        {
          int dz = 1;
          for (int dy = -1; dy < 2; ++dy)
            for (int dx = -1; dx < 2; ++dx)
            {
              int tmp = FETCH (x+dx, y+dy, z+dz);

              int Wn = reinterpret_cast<short2*>(&tmp)->y;
              int Fn = reinterpret_cast<short2*>(&tmp)->x;
              if (Wn == 0 || Fn == DIVISOR)
                continue;

              if ((F > 0 && Fn < 0) || (F < 0 && Fn > 0))
              {
                Vector3f Vn = ((Array3i (x+dx, y+dy, z+dz).cast<float>() + 0.5f) * cell_size).matrix ();
                Vector3f point = (V * (float)abs (Fn) + Vn * (float)abs (F)) / (float)(abs (F) + abs (Fn));

                pcl::PointXYZ xyz;
                xyz.x = point (0);
                xyz.y = point (1);
                xyz.z = point (2);

                cloud.points.push_back (xyz);
              }
            }
          dz = 0;
          for (int dy = 0; dy < 2; ++dy)
            for (int dx = -1; dx < dy * 2; ++dx)
            {
              int tmp = FETCH (x+dx, y+dy, z+dz);

              int Wn = reinterpret_cast<short2*>(&tmp)->y;
              int Fn = reinterpret_cast<short2*>(&tmp)->x;
              if (Wn == 0 || Fn == DIVISOR)
                continue;

              if ((F > 0 && Fn < 0) || (F < 0 && Fn > 0))
              {
                Vector3f Vn = ((Array3i (x+dx, y+dy, z+dz).cast<float>() + 0.5f) * cell_size).matrix ();
                Vector3f point = (V * (float)abs(Fn) + Vn * (float)abs(F))/(float)(abs(F) + abs (Fn));

                pcl::PointXYZ xyz;
                xyz.x = point (0);
                xyz.y = point (1);
                xyz.z = point (2);

                cloud.points.push_back (xyz);
              }
            }
        }
        else /* if (connected26) */
        {
          for (int i = 0; i < 3; ++i)
          {
            int ds[] = {0, 0, 0};
            ds[i] = 1;

            int dx = ds[0];
            int dy = ds[1];
            int dz = ds[2];

            int tmp = FETCH (x+dx, y+dy, z+dz);

            int Wn = reinterpret_cast<short2*>(&tmp)->y;
            int Fn = reinterpret_cast<short2*>(&tmp)->x;
            if (Wn == 0 || Fn == DIVISOR)
              continue;

            if ((F > 0 && Fn < 0) || (F < 0 && Fn > 0))
            {
              Vector3f Vn = ((Array3i (x+dx, y+dy, z+dz).cast<float>() + 0.5f) * cell_size).matrix ();
              Vector3f point = (V * (float)abs (Fn) + Vn * (float)abs (F)) / (float)(abs (F) + abs (Fn));

              pcl::PointXYZ xyz;
              xyz.x = point (0);
              xyz.y = point (1);
              xyz.z = point (2);

              cloud.points.push_back (xyz);
            }
          }
        } /* if (connected26) */
      }
    }
  }
  
#undef FETCH
  cloud.width  = (int)cloud.points.size ();
  cloud.height = 1;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
pcl::gpu::DeviceArray<pcl::gpu::TsdfVolume::PointType>
pcl::gpu::TsdfVolume::fetchCloud (DeviceArray<PointType>& cloud_buffer) const
{
  if (cloud_buffer.empty ())
    cloud_buffer.create (DEFAULT_CLOUD_BUFFER_SIZE);

  float3 device_volume_size = device_cast<const float3> (size_);
  size_t size = device::extractCloud (volume_, device_volume_size, cloud_buffer);
  return (DeviceArray<PointType> (cloud_buffer.ptr (), size));
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void
pcl::gpu::TsdfVolume::fetchNormals (const DeviceArray<PointType>& cloud, DeviceArray<PointType>& normals) const
{
  normals.create (cloud.size ());
  const float3 device_volume_size = device_cast<const float3> (size_);
  device::extractNormals (volume_, device_volume_size, cloud, (device::PointType*)normals.ptr ());
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void
pcl::gpu::TsdfVolume::fetchNormals (const DeviceArray<PointType>& cloud, DeviceArray<NormalType>& normals) const
{
  normals.create (cloud.size ());
  const float3 device_volume_size = device_cast<const float3> (size_);
  device::extractNormals (volume_, device_volume_size, cloud, (device::float8*)normals.ptr ());
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void
pcl::gpu::TsdfVolume::downloadTsdf (std::vector<float>& tsdf) const
{
  tsdf.resize (volume_.cols() * volume_.rows());
  volume_.download(&tsdf[0], volume_.cols() * sizeof(int));

#pragma omp parallel for
  for(int i = 0; i < (int) tsdf.size(); ++i)
  {
    float tmp = reinterpret_cast<short2*>(&tsdf[i])->x;
    tsdf[i] = tmp/device::DIVISOR;
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void
pcl::gpu::TsdfVolume::downloadTsdfAndWeighs (std::vector<float>& tsdf, std::vector<short>& weights) const
{
  int volumeSize = volume_.cols() * volume_.rows();
  tsdf.resize (volumeSize);
  weights.resize (volumeSize);
  volume_.download(&tsdf[0], volume_.cols() * sizeof(int));

#pragma omp parallel for
  for(int i = 0; i < (int) tsdf.size(); ++i)
  {
    short2 elem = *reinterpret_cast<short2*>(&tsdf[i]);
    tsdf[i] = (float)(elem.x)/device::DIVISOR;    
    weights[i] = (short)(elem.y);    
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void
pcl::gpu::TsdfVolume::downloadTsdfAndWeightsInt () {
  volume_.download(&volume_downloaded_[0], volume_.cols() * sizeof(int));
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void
pcl::gpu::TsdfVolume::uploadTsdfAndWeightsInt () {
  volume_.upload(volume_downloaded_, resolution_(2));
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

PointCloud<PointXYZ>::Ptr
pcl::gpu::TsdfVolume::getPointCloudNoNormal() {
  PointCloud<PointXYZ>::Ptr cloud_ptr = PointCloud<PointXYZ>::Ptr(new PointCloud<PointXYZ>);
  if (getNumVolumes() != 1) 
  {
    uploadTsdfAndWeightsInt();
  }
  
  DeviceArray<PointXYZ> extracted = fetchCloud (cloud_buffer_device_);
  extracted.download (cloud_ptr->points);
  cloud_ptr->width = (int)cloud_ptr->points.size ();
  cloud_ptr->height = 1;
  for (PointCloud<PointXYZ>::iterator it = cloud_ptr->begin(); it != cloud_ptr->end(); ++it)
  {
    it->x += shift_[0]*getVoxelSize()[0];
    it->y += shift_[1]*getVoxelSize()[1];
    it->z += shift_[2]*getVoxelSize()[2];
  }
  if (getNumVolumes() != 1) 
  {
    release();
  }
  return cloud_ptr;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

PointCloud<PointNormal>::Ptr
pcl::gpu::TsdfVolume::getPointCloud () {
  PointCloud<PointNormal>::Ptr combined_ptr = PointCloud<PointNormal>::Ptr(new PointCloud<PointNormal>);
  PointCloud<PointXYZ>::Ptr cloud_ptr = PointCloud<PointXYZ>::Ptr(new PointCloud<PointXYZ>);
  if (getNumVolumes() != 1) 
  {
    uploadTsdfAndWeightsInt();
  }
  
  DeviceArray<PointXYZ> extracted = fetchCloud (cloud_buffer_device_);
  
  fetchNormals (extracted, normals_device_);
  pcl::gpu::mergePointNormal (extracted, normals_device_, combined_device_);
  combined_device_.download (combined_ptr->points);
  combined_ptr->width = (int)combined_ptr->points.size ();
  combined_ptr->height = 1;

  for (PointCloud<PointNormal>::iterator xyznormal_it = combined_ptr->begin(); xyznormal_it != combined_ptr->end();++xyznormal_it)
  {
    xyznormal_it->x += shift_[0]*getVoxelSize()[0];
    xyznormal_it->y += shift_[1]*getVoxelSize()[1];
    xyznormal_it->z += shift_[2]*getVoxelSize()[2];
  }

  if (getNumVolumes() != 1) 
  {
    release();
  }
  return combined_ptr;
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

PointCloud<PointXYZRGBNormal>::Ptr
pcl::gpu::TsdfVolume::getColorPointCloud () {
  PointCloud<PointNormal>::Ptr combined_ptr = PointCloud<PointNormal>::Ptr(new PointCloud<PointNormal>);
  PointCloud<RGB>::Ptr point_colors_ptr  = PointCloud<RGB>::Ptr(new PointCloud<RGB>);

  if (getNumVolumes() != 1) 
  {
    uploadTsdfAndWeightsInt();
    color_volume_->uploadColorAndWeightsInt();
  }

  DeviceArray<PointXYZ> extracted = fetchCloud (cloud_buffer_device_);
  fetchNormals (extracted, normals_device_);
  pcl::gpu::mergePointNormal (extracted, normals_device_, combined_device_);
  combined_device_.download (combined_ptr->points);
  combined_ptr->width = (int)combined_ptr->points.size ();
  combined_ptr->height = 1;

  color_volume_->fetchColors(extracted, point_colors_device_);
  point_colors_device_.download(point_colors_ptr->points);
  point_colors_ptr->width = (int)point_colors_ptr->points.size ();
  point_colors_ptr->height = 1;
  
  PointCloud<PointXYZRGBNormal>::Ptr out_cloud = PointCloud<PointXYZRGBNormal>::Ptr(new PointCloud<PointXYZRGBNormal>);
  PointCloud<RGB>::iterator color_it = point_colors_ptr->begin();
  for (PointCloud<PointNormal>::iterator xyznormal_it = combined_ptr->begin(); xyznormal_it != combined_ptr->end(); ++color_it, ++xyznormal_it)
  {
    RGB color = *color_it;
    PointNormal xyznormal = *xyznormal_it;
    PointXYZRGBNormal point;
    point.x = xyznormal.x + shift_[0]*getVoxelSize()[0];
    point.y = xyznormal.y + shift_[1]*getVoxelSize()[1];
    point.z = xyznormal.z + shift_[2]*getVoxelSize()[2];
    point.normal_x = xyznormal.normal_x;
    point.normal_y = xyznormal.normal_y;
    point.normal_z = xyznormal.normal_z;
    point.rgb = color.rgb;
    out_cloud->push_back(point);
  }
  out_cloud->width = (int)out_cloud->points.size();
  out_cloud->height = 1;
  //pcl::concatenateFields(*point_colors_ptr, *combined_ptr, out_cloud);
  if (getNumVolumes() != 1)
  {
    release();
    color_volume_->release();
  }
  return out_cloud;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

ColorVolume::Ptr
pcl::gpu::TsdfVolume::getColorVolume() 
{
  return color_volume_;
}

void
pcl::gpu::TsdfVolume::setColorVolume(int max_weight) 
{
  color_volume_ = pcl::gpu::ColorVolume::Ptr( new ColorVolume(*this, max_weight) );
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void
pcl::gpu::TsdfVolume::setNumVolumes(int num_volumes) {
  num_volumes_ = num_volumes;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int
pcl::gpu::TsdfVolume::getNumVolumes() {
  return num_volumes_;
}