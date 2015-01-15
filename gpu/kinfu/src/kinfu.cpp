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

#include <iostream>
#include <algorithm>

#include <pcl/common/time.h>
#include <pcl/gpu/kinfu/kinfu.h>
#include <pcl/io/pcd_io.h>
#include "internal.h"

#include <Eigen/Core>
#include <Eigen/SVD>
#include <Eigen/Cholesky>
#include <Eigen/Geometry>
#include <Eigen/LU>

#include <map>

#ifdef HAVE_OPENCV
  #include <opencv2/opencv.hpp>
  #include <opencv2/gpu/gpu.hpp>
#endif

using namespace std;
using namespace pcl::device;
using namespace pcl::gpu;

using Eigen::AngleAxisf;
using Eigen::Array3f;
using Eigen::Vector3i;
using Eigen::Vector3f;

bool automa = true;
namespace pcl
{
  namespace gpu
  {
    Eigen::Vector3f rodrigues2(const Eigen::Matrix3f& matrix);
    void mergePointNormal(const DeviceArray<PointXYZ>& cloud, const DeviceArray<Normal>& normals, DeviceArray<PointNormal>& output);
  }
}

struct volume_compare {
public:
  bool operator () (const int3 x, const int3 y) const 
  {
    if (x.x*x.x + x.y*x.y + x.z*x.z != y.x*y.x + y.y*y.y + y.z*y.z)
    {
      return x.x*x.x + x.y*x.y + x.z*x.z < y.x*y.x + y.y*y.y + y.z*y.z;
    }
    else if (x.x != y.x)
    {
      return x.x < y.x;
    }
    else if (x.y != y.y)
    {
      return x.y < y.y;
    }
    else
    {
      return x.z < y.z;
    }
  }
};

  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
pcl::gpu::KinfuTracker::KinfuTracker (int rows, int cols) : rows_(rows), cols_(cols), global_time_(0), max_icp_distance_(0), integration_metric_threshold_(0.f), disable_icp_(false)
{
  const Vector3f volume_size = Vector3f::Constant (VOLUME_SIZE);
  volume_size_ = volume_size;
  const Vector3i volume_resolution(VOLUME_X, VOLUME_Y, VOLUME_Z);
  volume_resolution_ = volume_resolution;
  
  setDepthIntrinsics (KINFU_DEFAULT_DEPTH_FOCAL_X, KINFU_DEFAULT_DEPTH_FOCAL_Y); // default values, can be overwritten
  
  init_Rcam_ = Eigen::Matrix3f::Identity ();// * AngleAxisf(-30.f/180*3.1415926, Vector3f::UnitX());
  init_tcam_ = volume_size * 0.5f - Vector3f (0, 0, volume_size (2) / 2 * 1.2f);

  const int iters[] = {10, 5, 4};
  std::copy (iters, iters + LEVELS, icp_iterations_);

  const float default_distThres = 0.10f; //meters
  const float default_angleThres = sin (20.f * 3.14159254f / 180.f);
  const float default_tranc_dist = 0.03f; //meters

  tranc_dist_ = default_tranc_dist;

  setIcpCorespFilteringParams (default_distThres, default_angleThres);

  tsdf_volume_ = TsdfVolume::Ptr( new TsdfVolume(volume_resolution, true) );
  tsdf_volume_->setSize(volume_size);
  tsdf_volume_->setTsdfTruncDist (default_tranc_dist);
  tsdf_volume_->setShift(Vector3i({0,0,600}));

  //tsdf_volume_list_.push_back(tsdf_volume_);

  std::list<Vector3i> shifts;

  if (!automa) {
    //shifts.push_back(Vector3i({(VOLUME_X/2 - 5),(VOLUME_Y/2 -5),0})); 
    //shifts.push_back(Vector3i({(VOLUME_X/2 - 5),-(VOLUME_Y/2 - 5),0})); 
    //shifts.push_back(Vector3i({-(VOLUME_X/2 -5),(VOLUME_Y/2 -20),0}));
    shifts.push_back(Vector3i({0,0,0}));
    //shifts.push_back(Vector3i({0,0,0}));
    //shifts.push_back(Vector3i({-470,-200,1400}));
    for (std::list<Vector3i>::iterator it = shifts.begin(); it != shifts.end(); ++it) {
      const Vector3i shift = *it;
      insertVolume(shift);
    }
  }
  allocateBufffers (rows, cols);
  rmats_.reserve (30000);
  tvecs_.reserve (30000);

  reset ();
  
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
pcl::gpu::KinfuTracker::setDepthIntrinsics (float fx, float fy, float cx, float cy)
{
  fx_ = fx;
  fy_ = fy;
  cx_ = (cx == -1) ? cols_/2-0.5f : cx;
  cy_ = (cy == -1) ? rows_/2-0.5f : cy;  
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
pcl::gpu::KinfuTracker::getDepthIntrinsics (float& fx, float& fy, float& cx, float& cy)
{
  fx = fx_;
  fy = fy_;
  cx = cx_;
  cy = cy_;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
pcl::gpu::KinfuTracker::setInitalCameraPose (const Eigen::Affine3f& pose)
{
  init_Rcam_ = pose.rotation ();
  init_tcam_ = pose.translation ();
  reset ();
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
pcl::gpu::KinfuTracker::setDepthTruncationForICP (float max_icp_distance)
{
  max_icp_distance_ = max_icp_distance;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
pcl::gpu::KinfuTracker::setCameraMovementThreshold(float threshold)
{
  integration_metric_threshold_ = threshold;  
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
pcl::gpu::KinfuTracker::setIcpCorespFilteringParams (float distThreshold, float sineOfAngle)
{
  distThres_  = distThreshold; //mm
  angleThres_ = sineOfAngle;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int
pcl::gpu::KinfuTracker::cols ()
{
  return (cols_);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int
pcl::gpu::KinfuTracker::rows ()
{
  return (rows_);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
pcl::gpu::KinfuTracker::reset()
{
  if (global_time_)
    cout << "Reset" << endl;

  global_time_ = 0;
  rmats_.clear ();
  tvecs_.clear ();

  rmats_.push_back (init_Rcam_);
  tvecs_.push_back (init_tcam_);

  for (std::list<TsdfVolume::Ptr>::iterator it = tsdf_volume_list_.begin(); it != tsdf_volume_list_.end(); ++it) {
    (*it)->reset();
    if (integrate_color_)
    {
      (*it)->getColorVolume().reset();
    }
  } 
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
pcl::gpu::KinfuTracker::allocateBufffers (int rows, int cols)
{    
  depths_curr_.resize (LEVELS);
  vmaps_g_curr_.resize (LEVELS);
  nmaps_g_curr_.resize (LEVELS);

  vmaps_g_prev_.resize (LEVELS);
  nmaps_g_prev_.resize (LEVELS);

  vmaps_curr_.resize (LEVELS);
  nmaps_curr_.resize (LEVELS);

  coresps_.resize (LEVELS);

  for (int i = 0; i < LEVELS; ++i)
  {
    int pyr_rows = rows >> i;
    int pyr_cols = cols >> i;

    depths_curr_[i].create (pyr_rows, pyr_cols);

    vmaps_g_curr_[i].create (pyr_rows*3, pyr_cols);
    nmaps_g_curr_[i].create (pyr_rows*3, pyr_cols);

    vmaps_g_prev_[i].create (pyr_rows*3, pyr_cols);
    nmaps_g_prev_[i].create (pyr_rows*3, pyr_cols);

    vmaps_curr_[i].create (pyr_rows*3, pyr_cols);
    nmaps_curr_[i].create (pyr_rows*3, pyr_cols);

    coresps_[i].create (pyr_rows, pyr_cols);
  }
  
  ray_cubes_.create(rows, cols);
  
  depthRawScaled_.create (rows, cols);
  // see estimate tranform for the magic numbers
  gbuf_.create (27, 20*60);
  sumbuf_.create (27);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
bool
pcl::gpu::KinfuTracker::operator() (const DepthMap& depth_raw, 
    Eigen::Affine3f *hint)
{  

  device::Intr intr (fx_, fy_, cx_, cy_);

  if (!disable_icp_)
  {
      {
        //ScopeTime time(">>> Bilateral, pyr-down-all, create-maps-all");
        //depth_raw.copyTo(depths_curr[0]);
        device::bilateralFilter (depth_raw, depths_curr_[0]);

        if (max_icp_distance_ > 0)
          device::truncateDepth(depths_curr_[0], max_icp_distance_);

        for (int i = 1; i < LEVELS; ++i)
          device::pyrDown (depths_curr_[i-1], depths_curr_[i]);

        for (int i = 0; i < LEVELS; ++i)
        {
          device::createVMap (intr(i), depths_curr_[i], vmaps_curr_[i]);
          //device::createNMap(vmaps_curr_[i], nmaps_curr_[i]);
          computeNormalsEigen (vmaps_curr_[i], nmaps_curr_[i]);
        }
        pcl::device::sync ();
      }
      //can't perform more on first frame
      if (global_time_ == 0)
      {
        Matrix3frm init_Rcam = rmats_[0]; //  [Ri|ti] - pos of camera, i.e.
        Vector3f   init_tcam = tvecs_[0]; //  transform from camera to global coo space for (i-1)th camera pose

        Mat33&  device_Rcam = device_cast<Mat33> (init_Rcam);
        float3& device_tcam = device_cast<float3>(init_tcam);

        float3 device_volume_size = device_cast<const float3> (volume_size_);
        Matrix3frm Rcurr_inv = init_Rcam.inverse ();
        Mat33&  device_Rcurr_inv = device_cast<Mat33> (Rcurr_inv);
        float3& device_tcurr = device_cast<float3> (init_tcam);
        if (automa) 
        {
          generateNumCubeRays(intr, device_Rcurr_inv, device_tcurr, device_volume_size, depth_raw, rows_, cols_, ray_cubes_);
          updateProcessedVolumes();
        }
        std::cout << tsdf_volume_list_.size() << std::endl;
        for (std::list<TsdfVolume::Ptr>::iterator it = tsdf_volume_list_.begin(); it != tsdf_volume_list_.end(); ++it) {
          TsdfVolume::Ptr cur_volume = *it;
          Matrix3frm init_Rcam_inv = init_Rcam.inverse ();
          Mat33&   device_Rcam_inv = device_cast<Mat33> (init_Rcam_inv);
          float3 device_volume_size = device_cast<const float3>(cur_volume->getSize());
          int3 device_shift = device_cast<const int3>(cur_volume->getShift());
          if (single_tsdf_)
          {
            device::integrateTsdfVolume(depth_raw, intr, device_volume_size, device_Rcam_inv, device_tcam, cur_volume->getTsdfTruncDist(), cur_volume->data(), depthRawScaled_, device_shift);
          }
          else 
          {
            cur_volume->uploadTsdfAndWeightsInt();
            device::integrateTsdfVolume(depth_raw, intr, device_volume_size, device_Rcam_inv, device_tcam, cur_volume->getTsdfTruncDist(), cur_volume->data(), depthRawScaled_, device_shift);

            //integrateTsdfVolume(depth_raw, intr, device_volume_size, device_Rcam_inv, device_tcam, tranc_dist, volume_);    
            cur_volume->downloadTsdfAndWeightsInt();
            cur_volume->release();
          }
        }

        for (int i = 0; i < LEVELS; ++i)
        {
          device::tranformMaps (vmaps_curr_[i], nmaps_curr_[i], device_Rcam, device_tcam, vmaps_g_prev_[i], nmaps_g_prev_[i]);
        }
        ++global_time_;

        return (false);
      }

      ///////////////////////////////////////////////////////////////////////////////////////////
      // Iterative Closest Point
      Matrix3frm Rprev = rmats_[global_time_ - 1]; //  [Ri|ti] - pos of camera, i.e.
      Vector3f   tprev = tvecs_[global_time_ - 1]; //  tranfrom from camera to global coo space for (i-1)th camera pose
      Matrix3frm Rprev_inv = Rprev.inverse (); //Rprev.t();

      //Mat33&  device_Rprev     = device_cast<Mat33> (Rprev);
      Mat33&  device_Rprev_inv = device_cast<Mat33> (Rprev_inv);
      float3& device_tprev     = device_cast<float3> (tprev);
      Matrix3frm Rcurr;
      Vector3f tcurr;
      if(hint)
      {
        Rcurr = hint->rotation().matrix();
        tcurr = hint->translation().matrix();
      }
      else
      {
        Rcurr = Rprev; // tranform to global coo for ith camera pose
        tcurr = tprev;
      }
      {
        //ScopeTime time("icp-all");
        for (int level_index = LEVELS-1; level_index>=0; --level_index)
        {
          int iter_num = icp_iterations_[level_index];

          MapArr& vmap_curr = vmaps_curr_[level_index];
          MapArr& nmap_curr = nmaps_curr_[level_index];

          //MapArr& vmap_g_curr = vmaps_g_curr_[level_index];
          //MapArr& nmap_g_curr = nmaps_g_curr_[level_index];

          MapArr& vmap_g_prev = vmaps_g_prev_[level_index];
          MapArr& nmap_g_prev = nmaps_g_prev_[level_index];

          //CorespMap& coresp = coresps_[level_index];

          for (int iter = 0; iter < iter_num; ++iter)
          {
            Mat33&  device_Rcurr = device_cast<Mat33> (Rcurr);
            float3& device_tcurr = device_cast<float3>(tcurr);

            Eigen::Matrix<double, 6, 6, Eigen::RowMajor> A;
            Eigen::Matrix<double, 6, 1> b;
    #if 0
            device::tranformMaps(vmap_curr, nmap_curr, device_Rcurr, device_tcurr, vmap_g_curr, nmap_g_curr);
            findCoresp(vmap_g_curr, nmap_g_curr, device_Rprev_inv, device_tprev, intr(level_index), vmap_g_prev, nmap_g_prev, distThres_, angleThres_, coresp);
            device::estimateTransform(vmap_g_prev, nmap_g_prev, vmap_g_curr, coresp, gbuf_, sumbuf_, A.data(), b.data());

            //cv::gpu::GpuMat ma(coresp.rows(), coresp.cols(), CV_32S, coresp.ptr(), coresp.step());
            //cv::Mat cpu;
            //ma.download(cpu);
            //cv::imshow(names[level_index] + string(" --- coresp white == -1"), cpu == -1);
    #else
            estimateCombined (device_Rcurr, device_tcurr, vmap_curr, nmap_curr, device_Rprev_inv, device_tprev, intr (level_index),
                              vmap_g_prev, nmap_g_prev, distThres_, angleThres_, gbuf_, sumbuf_, A.data (), b.data ());
    #endif
            //checking nullspace
            double det = A.determinant ();

            if (fabs (det) < 1e-15 || pcl_isnan (det))
            {
              if (pcl_isnan (det)) cout << "qnan" << endl;

              reset ();
              return (false);
            }
            //float maxc = A.maxCoeff();

            Eigen::Matrix<float, 6, 1> result = A.llt ().solve (b).cast<float>();
            //Eigen::Matrix<float, 6, 1> result = A.jacobiSvd(ComputeThinU | ComputeThinV).solve(b);

            float alpha = result (0);
            float beta  = result (1);
            float gamma = result (2);

            Eigen::Matrix3f Rinc = (Eigen::Matrix3f)AngleAxisf (gamma, Vector3f::UnitZ ()) * AngleAxisf (beta, Vector3f::UnitY ()) * AngleAxisf (alpha, Vector3f::UnitX ());
            Vector3f tinc = result.tail<3> ();

            //compose
            tcurr = Rinc * tcurr + tinc;
            Rcurr = Rinc * Rcurr;
          }
        }
      }
      //save tranform
      rmats_.push_back (Rcurr);
      tvecs_.push_back (tcurr);
  } 
  else /* if (disable_icp_) */
  {
      if (global_time_ == 0)
        ++global_time_;

      Matrix3frm Rcurr = rmats_[global_time_ - 1];
      Vector3f   tcurr = tvecs_[global_time_ - 1];

      rmats_.push_back (Rcurr);
      tvecs_.push_back (tcurr);

  }

  Matrix3frm Rprev = rmats_[global_time_ - 1];
  Vector3f   tprev = tvecs_[global_time_ - 1];

  Matrix3frm Rcurr = rmats_.back();
  Vector3f   tcurr = tvecs_.back();

  ///////////////////////////////////////////////////////////////////////////////////////////
  // Integration check - We do not integrate volume if camera does not move.  
  float rnorm = rodrigues2(Rcurr.inverse() * Rprev).norm();
  float tnorm = (tcurr - tprev).norm();  
  const float alpha = 1.f;
  bool integrate = (rnorm + alpha * tnorm)/2 >= integration_metric_threshold_;
  
  if (disable_icp_)
    integrate = true;

  ///////////////////////////////////////////////////////////////////////////////////////////
  // Volume integration
  bool first = true;
  Matrix3frm Rcurr_inv = Rcurr.inverse ();
  Mat33&  device_Rcurr_inv = device_cast<Mat33> (Rcurr_inv);
  float3& device_tcurr = device_cast<float3> (tcurr);
  for (std::list<TsdfVolume::Ptr>::iterator it = tsdf_volume_list_.begin(); it != tsdf_volume_list_.end(); ++it) {
    TsdfVolume::Ptr cur_volume = *it;
    //TsdfVolume::Ptr cur_volume = tsdf_volume_list_.front();
    if (!single_tsdf_)
    {
      cur_volume->uploadTsdfAndWeightsInt();
    }
    

    float3 device_volume_size = device_cast<const float3> (cur_volume->getSize());

    //Matrix3frm Rcurr_inv = Rcurr.inverse ();
    //Mat33&  device_Rcurr_inv = device_cast<Mat33> (Rcurr_inv);
    //float3& device_tcurr = device_cast<float3> (tcurr);
    int3 device_shift = device_cast<const int3>(cur_volume->getShift());
    if (integrate)
    {
      //ScopeTime time("tsdf");
      //integrateTsdfVolume(depth_raw, intr, device_volume_size, device_Rcurr_inv, device_tcurr, tranc_dist, volume_);
      integrateTsdfVolume (depth_raw, intr, device_volume_size, device_Rcurr_inv, device_tcurr, cur_volume->getTsdfTruncDist(), cur_volume->data(), depthRawScaled_, device_shift);
    }

    ///////////////////////////////////////////////////////////////////////////////////////////
    // Ray casting
    Mat33& device_Rcurr = device_cast<Mat33> (Rcurr);
    {
      //ScopeTime time("ray-cast-all");
      raycast (intr, device_Rcurr, device_tcurr, cur_volume->getTsdfTruncDist(), device_volume_size, cur_volume->data(), device_shift, vmaps_g_prev_[0], nmaps_g_prev_[0], first);
      for (int i = 1; i < LEVELS; ++i)
      {
        resizeVMap (vmaps_g_prev_[i-1], vmaps_g_prev_[i]);
        resizeNMap (nmaps_g_prev_[i-1], nmaps_g_prev_[i]);
      }
      pcl::device::sync ();
    }

    if (!single_tsdf_)
    {
      cur_volume->downloadTsdfAndWeightsInt();
      cur_volume->release();
    }
    first = false;
  }

  if (automa) {
    float3 device_volume_size = device_cast<const float3> (tsdf_volume_list_.front()->getSize());
    generateNumCubeRays(intr, device_Rcurr_inv, device_tcurr, device_volume_size, depth_raw, rows_, cols_, ray_cubes_);
    updateProcessedVolumes();
  }
  std::cout << "tcurr: " << "(" << tcurr[0] << "," << tcurr[1] << "," << tcurr[2] << ")" << std::endl;
  ++global_time_;
  std::cout << global_time_ << std::endl;
  return (true);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
Eigen::Affine3f
pcl::gpu::KinfuTracker::getCameraPose (int time) const
{
  if (time > (int)rmats_.size () || time < 0)
    time = rmats_.size () - 1;

  Eigen::Affine3f aff;
  aff.linear () = rmats_[time];
  aff.translation () = tvecs_[time];
  return (aff);
}
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

size_t
pcl::gpu::KinfuTracker::getNumberOfPoses () const
{
  return rmats_.size();
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

const TsdfVolume& 
pcl::gpu::KinfuTracker::volume() const 
{ 
  std::cout << "volume 1 called" << std::endl;
  return *tsdf_volume_; 
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

TsdfVolume& 
pcl::gpu::KinfuTracker::volume()
{
  return *tsdf_volume_;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

std::list<TsdfVolume::Ptr>&
pcl::gpu::KinfuTracker::volumeList()
{
  return tsdf_volume_list_;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

bool
pcl::gpu::KinfuTracker::integrateColor()
{
  return integrate_color_;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

bool
pcl::gpu::KinfuTracker::singleTsdf()
{
  return single_tsdf_;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void
pcl::gpu::KinfuTracker::insertVolume (const Eigen::Vector3i shift)
{
  if (single_tsdf_) {
    tsdf_volume_list_.front()->downloadTsdfAndWeightsInt();
    tsdf_volume_list_.front()->release();
    if (integrate_color_) {
      tsdf_volume_list_.front()->getColorVolume()->downloadColorAndWeightsInt();
      tsdf_volume_list_.front()->getColorVolume()->release();
    }
    single_tsdf_ = false;
  }
  TsdfVolume::Ptr tsdf_vol = TsdfVolume::Ptr( new TsdfVolume(volume_resolution_, true) );
  tsdf_vol->setSize(volume_size_);
  tsdf_vol->setShift(shift);
  tsdf_vol->setTsdfTruncDist (tranc_dist_);
  tsdf_volume_list_.push_back(tsdf_vol);
  TsdfVolume::setNumVolumes(TsdfVolume::getNumVolumes() + 1);
  if (integrate_color_) {
    tsdf_vol->setColorVolume(max_weight_);
  }
  if (TsdfVolume::getNumVolumes() == 1) {
    single_tsdf_ = true;
    tsdf_volume_list_.front()->uploadTsdfAndWeightsInt();
    if (integrate_color_) {
      tsdf_vol->getColorVolume()->uploadColorAndWeightsInt();
    }
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void
pcl::gpu::KinfuTracker::removeVolume(TsdfVolume::Ptr volume)
{
  tsdf_volume_list_.remove(volume);
  TsdfVolume::setNumVolumes(TsdfVolume::getNumVolumes() - 1);
  if (TsdfVolume::getNumVolumes() == 1) {
    single_tsdf_ = true;
    tsdf_volume_list_.front()->uploadTsdfAndWeightsInt();
    if (integrate_color_)
    {
      tsdf_volume_list_.front()->getColorVolume()->uploadColorAndWeightsInt();
    }
  }
  else 
  {
    single_tsdf_ = false;
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void
pcl::gpu::KinfuTracker::setVolumeResolution (Vector3i resolution) 
{
  volume_resolution_ = resolution;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void
pcl::gpu::KinfuTracker::setVolumeSize (Vector3f size)
{
  volume_size_ = size;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void
pcl::gpu::KinfuTracker::setTruncDist (float tranc_dist)
{
  tranc_dist_ = tranc_dist;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void
pcl::gpu::KinfuTracker::updateProcessedVolumes() 
{
  std::vector<int3> ray_cubes_cpu = std::vector<int3>(rows_*cols_);
  ray_cubes_.download(&ray_cubes_cpu[0], ray_cubes_.cols()*sizeof(int3));
  std::map<int3, int, volume_compare> cube_counts;
  for (int i = 0; i < rows_*cols_; i++)
  {
    for (int j = 0; j < 1; j++)
    {
      int3 cube_val = ray_cubes_cpu[i];
      
      if (cube_val.x == -1)
        continue;
      if (cube_counts.count(cube_val)>0)
      {
        cube_counts[cube_val] += 1;
      }
      else
      {
        cube_counts[cube_val] = 1;
      }
      
    }
  }
  
  for (std::map<int3, int>::iterator it = cube_counts.begin(); it != cube_counts.end(); it++ )
  {
   std::cout << "(" << it->first.x << ", " << it->first.y << ", " << it->first.z << "): " << it->second << std::endl;
  }
  
  int max_shift_count = 0;
  Eigen::Vector3i max_shift;
  vector<Eigen::Vector3i> to_be_added;
  vector<TsdfVolume::Ptr> to_be_removed;
  int add_threshold_ = 20000;
  int remove_threshold_ = 2000;

  for (std::list<TsdfVolume::Ptr>::iterator it = tsdf_volume_list_.begin(); it != tsdf_volume_list_.end(); ++it)
  {
    int3 cur_shift;
    cur_shift.x = (*it)->getShift()[0];
    cur_shift.y = (*it)->getShift()[1];
    cur_shift.z = (*it)->getShift()[2];
    if (cube_counts.count(cur_shift) <= 0)
    {
      to_be_removed.push_back(*it);
    }
    else if(cube_counts[cur_shift] <= remove_threshold_)
    {
      to_be_removed.push_back(*it);
    }
  }
  for (std::vector<TsdfVolume::Ptr>::iterator it = to_be_removed.begin(); it != to_be_removed.end(); it++) 
  {
    stringstream cloud_name;
    cloud_name << "cloud_" << (*it)->getShift()[0] << "_" << (*it)->getShift()[1] << "_" << (*it)->getShift()[2] << ".pcd";
    std::cout << cloud_name.str() << std::endl;
    if (integrate_color_) 
    {
      pcl::io::savePCDFile (cloud_name.str(), *(*it)->getColorPointCloud(), true);
    }
    else {
      downloadPointCloud(*it, cloud_name.str());
    }
    removeVolume(*it);
  }
  for (std::map<int3, int>::iterator it = cube_counts.begin(); it != cube_counts.end(); it++ )
  {
    if (it->second > add_threshold_) 
    {
      Eigen::Vector3i cur_shift(it->first.x, it->first.y, it->first.z);
      bool exists = false;
      for (std::list<TsdfVolume::Ptr>::iterator vol_it = tsdf_volume_list_.begin(); vol_it != tsdf_volume_list_.end(); ++vol_it)
      {
        if ((*vol_it)->getShift() == cur_shift)
        {
          exists = true;
          break;
        }
      }
      if (!exists)
      {
        insertVolume(cur_shift);
      }

    }
  }
  for (std::list<TsdfVolume::Ptr>::iterator it = tsdf_volume_list_.begin(); it != tsdf_volume_list_.end(); ++it)
  {
    std::cout << "(" << (*it)->getShift()[0] << ", " << (*it)->getShift()[1] << ", " << (*it)->getShift()[2] << ")" << std::endl;
  }
  std::cout << "Num Volumes: " << tsdf_volume_list_.size() << std::endl;
}
     
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
pcl::gpu::KinfuTracker::getImage (View& view) const
{
  //Eigen::Vector3f light_source_pose = tsdf_volume_->getSize() * (-3.f);
  Eigen::Vector3f light_source_pose = tvecs_[tvecs_.size () - 1];

  device::LightSource light;
  light.number = 1;
  light.pos[0] = device_cast<const float3>(light_source_pose);

  view.create (rows_, cols_);
  generateImage (vmaps_g_prev_[0], nmaps_g_prev_[0], light, view);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
pcl::gpu::KinfuTracker::getLastFrameCloud (DeviceArray2D<PointType>& cloud) const
{
  cloud.create (rows_, cols_);
  DeviceArray2D<float4>& c = (DeviceArray2D<float4>&)cloud;
  device::convert (vmaps_g_prev_[0], c);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
pcl::gpu::KinfuTracker::getLastFrameNormals (DeviceArray2D<NormalType>& normals) const
{
  normals.create (rows_, cols_);
  DeviceArray2D<float8>& n = (DeviceArray2D<float8>&)normals;
  device::convert (nmaps_g_prev_[0], n);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void 
pcl::gpu::KinfuTracker::disableIcp() { disable_icp_ = true; }


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void
pcl::gpu::KinfuTracker::initColorIntegration(int max_weight)
{    
  for (std::list<TsdfVolume::Ptr>::iterator it = tsdf_volume_list_.begin(); it != tsdf_volume_list_.end(); it++) {
    (*it)->setColorVolume(max_weight);
  }
  if (single_tsdf_)
  {
    tsdf_volume_list_.front()->getColorVolume()->uploadColorAndWeightsInt();
  }
  integrate_color_ = true;
  max_weight_ = max_weight;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
pcl::gpu::KinfuTracker::downloadPointCloud(TsdfVolume::Ptr volume, string name) 
{
  cloud_ptr_ = PointCloud<PointXYZ>::Ptr (new PointCloud<PointXYZ>);
  normals_ptr_ = PointCloud<Normal>::Ptr (new PointCloud<Normal>);
  combined_ptr_ = PointCloud<PointNormal>::Ptr (new PointCloud<PointNormal>);
  point_colors_ptr_ = PointCloud<RGB>::Ptr (new PointCloud<RGB>);

  if (single_tsdf_) 
  {
    volume->uploadTsdfAndWeightsInt();
  }
  
  DeviceArray<PointXYZ> extracted = volume->fetchCloud (cloud_buffer_device_);
  
  volume->fetchNormals (extracted, normals_device_);
  pcl::gpu::mergePointNormal (extracted, normals_device_, combined_device_);
  combined_device_.download (combined_ptr_->points);
  combined_ptr_->width = (int)combined_ptr_->points.size ();
  combined_ptr_->height = 1;

  for (PointCloud<PointNormal>::iterator xyznormal_it = combined_ptr_->begin(); xyznormal_it != combined_ptr_->end();++xyznormal_it)
  {
    xyznormal_it->x += volume->getShift()[0]*volume->getVoxelSize()[0];
    xyznormal_it->y += volume->getShift()[1]*volume->getVoxelSize()[1];
    xyznormal_it->z += volume->getShift()[2]*volume->getVoxelSize()[2];
  }

  if (single_tsdf_) 
  {
    volume->release();
  }
  pcl::io::savePCDFile (name, *combined_ptr_, true);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
bool 
pcl::gpu::KinfuTracker::operator() (const DepthMap& depth, const View& colors)
{ 
  bool res = (*this)(depth);

  if (res && integrate_color_)
  {
    std::list<TsdfVolume::Ptr>::iterator tsdfIt = tsdf_volume_list_.begin();
    for (std::list<TsdfVolume::Ptr>::iterator it = tsdf_volume_list_.begin(); it != tsdf_volume_list_.end(); ++it)
    {
      TsdfVolume::Ptr tsdf_volume = *it;
      ColorVolume::Ptr color_volume = tsdf_volume->getColorVolume();
      const float3 device_volume_size = device_cast<const float3> (tsdf_volume->getSize());
      device::Intr intr(fx_, fy_, cx_, cy_);

      Matrix3frm R_inv = rmats_.back().inverse();
      Vector3f   t     = tvecs_.back();
      
      Mat33&  device_Rcurr_inv = device_cast<Mat33> (R_inv);
      float3& device_tcurr = device_cast<float3> (t);

      int3 device_shift = device_cast<const int3>(tsdf_volume->getShift());
      if (!single_tsdf_)
        color_volume->uploadColorAndWeightsInt();
      device::updateColorVolume(intr, tsdf_volume_->getTsdfTruncDist(), device_Rcurr_inv, device_tcurr, vmaps_g_prev_[0], 
          colors, device_volume_size, color_volume->data(), device_shift, color_volume->getMaxWeight());
      if (!single_tsdf_)
      {
        color_volume->downloadColorAndWeightsInt ();
        color_volume->release();
      }
    }
  }
  return res;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

namespace pcl
{
  namespace gpu
  {
    PCL_EXPORTS void 
    paint3DView(const KinfuTracker::View& rgb24, KinfuTracker::View& view, float colors_weight = 0.5f)
    {
      device::paint3DView(rgb24, view, colors_weight);
    }

    PCL_EXPORTS void
    mergePointNormal(const DeviceArray<PointXYZ>& cloud, const DeviceArray<Normal>& normals, DeviceArray<PointNormal>& output)
    {
      const size_t size = min(cloud.size(), normals.size());
      output.create(size);

      const DeviceArray<float4>& c = (const DeviceArray<float4>&)cloud;
      const DeviceArray<float8>& n = (const DeviceArray<float8>&)normals;
      const DeviceArray<float12>& o = (const DeviceArray<float12>&)output;
      device::mergePointNormal(c, n, o);           
    }

    Eigen::Vector3f rodrigues2(const Eigen::Matrix3f& matrix)
    {
      Eigen::JacobiSVD<Eigen::Matrix3f> svd(matrix, Eigen::ComputeFullV | Eigen::ComputeFullU);    
      Eigen::Matrix3f R = svd.matrixU() * svd.matrixV().transpose();

      double rx = R(2, 1) - R(1, 2);
      double ry = R(0, 2) - R(2, 0);
      double rz = R(1, 0) - R(0, 1);

      double s = sqrt((rx*rx + ry*ry + rz*rz)*0.25);
      double c = (R.trace() - 1) * 0.5;
      c = c > 1. ? 1. : c < -1. ? -1. : c;

      double theta = acos(c);

      if( s < 1e-5 )
      {
        double t;

        if( c > 0 )
          rx = ry = rz = 0;
        else
        {
          t = (R(0, 0) + 1)*0.5;
          rx = sqrt( std::max(t, 0.0) );
          t = (R(1, 1) + 1)*0.5;
          ry = sqrt( std::max(t, 0.0) ) * (R(0, 1) < 0 ? -1.0 : 1.0);
          t = (R(2, 2) + 1)*0.5;
          rz = sqrt( std::max(t, 0.0) ) * (R(0, 2) < 0 ? -1.0 : 1.0);

          if( fabs(rx) < fabs(ry) && fabs(rx) < fabs(rz) && (R(1, 2) > 0) != (ry*rz > 0) )
            rz = -rz;
          theta /= sqrt(rx*rx + ry*ry + rz*rz);
          rx *= theta;
          ry *= theta;
          rz *= theta;
        }
      }
      else
      {
        double vth = 1/(2*s);
        vth *= theta;
        rx *= vth; ry *= vth; rz *= vth;
      }
      return Eigen::Vector3d(rx, ry, rz).cast<float>();
    }
  }
}
