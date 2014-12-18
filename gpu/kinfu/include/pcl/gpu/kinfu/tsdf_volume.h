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

#ifndef PCL_KINFU_TSDF_VOLUME_H_
#define PCL_KINFU_TSDF_VOLUME_H_

#include <pcl/pcl_macros.h>
#include <pcl/gpu/containers/device_array.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <Eigen/Core>
#include <vector>
#include <list>

namespace pcl
{
  namespace gpu
  {
    /** \brief TsdfVolume class
      * \author Anatoly Baskeheev, Itseez Ltd, (myname.mysurname@mycompany.com)
      */
    class PCL_EXPORTS TsdfVolume
    {
    public:
      typedef boost::shared_ptr<TsdfVolume> Ptr;

      /** \brief Supported Point Types */
      typedef PointXYZ PointType;
      typedef Normal  NormalType;

      /** \brief Default buffer size for fetching cloud. It limits max number of points that can be extracted */
      enum { DEFAULT_CLOUD_BUFFER_SIZE = 10 * 1000 * 1000 };
            
      /** \brief Constructor
        * \param[in] resolution volume resolution
        */
      TsdfVolume(const Eigen::Vector3i& resolution);

      /** \brief Constructor for case with multiple tsdf volumes
        * \param[in] resolution volume resolution
        * \param[in] first only initialize gpu if first volume
        */
      TsdfVolume(const Eigen::Vector3i& resolution, bool first);           
            
      /** \brief Sets Tsdf volume size for each dimention
        * \param[in] size size of tsdf volume in meters
        */
      void
      setSize(const Eigen::Vector3f& size);
      
      /** \brief Sets Tsdf truncation distance. Must be greater than 2 * volume_voxel_size
        * \param[in] distance TSDF truncation distance 
        */
      void
      setTsdfTruncDist (float distance);

      /** \brief Sets the translation of the tsdf volume from (0,0,0)
        * \param[in] [x,y,z] translation
        */
      void
      setShift (const Eigen::Vector3i& shift);

      /** \brief Returns tsdf volume container that point to data in GPU memroy */
      DeviceArray2D<int> 
      data() const;

      /** \brief Returns volume size in meters */
      const Eigen::Vector3f&
      getSize() const;
            
      /** \brief Returns volume resolution */
      const Eigen::Vector3i&
      getResolution() const;

      /** \brief Returns volume voxel size in meters */
      const Eigen::Vector3f
      getVoxelSize() const;
      
      /** \brief Returns tsdf truncation distance in meters */
      float
      getTsdfTruncDist () const;

      /** \brief Return the translation of the tsdf volume from (0,0,0) */
      const Eigen::Vector3i&
      getShift () const;

      /** \brief Returns the most recently downloaded volume representation */
      const std::vector<int>&
      getTsdfAndWeightsInt () const;
     
      /** \brief Resets tsdf volume data to uninitialized state */
      void 
      reset();

      /** \brief Releases this tsdf's data from gpu memory */
      void 
      release();

      /** \brief Generates cloud using CPU (downloads volumetric representation to CPU memory)
        * \param[out] cloud output array for cloud
        * \param[in] connected26 If false point cloud is extracted using 6 neighbor, otherwise 26.
        */
      void
      fetchCloudHost (PointCloud<PointType>& cloud, bool connected26 = false) const;

      /** \brief Generates cloud using GPU in connected6 mode only
        * \param[out] cloud_buffer buffer to store point cloud
        * \return DeviceArray with disabled reference counting that points to filled part of cloud_buffer.
        */
      DeviceArray<PointType>
      fetchCloud (DeviceArray<PointType>& cloud_buffer) const;

      /** \brief Computes normals as gradient of tsdf for given points
        * \param[in] cloud Points where normals are computed.
        * \param[out] normals array for normals
        */
      void
      fetchNormals (const DeviceArray<PointType>& cloud, DeviceArray<PointType>& normals) const;

      /** \brief Computes normals as gradient of tsdf for given points
        * \param[in] cloud Points where normals are computed.
        * \param[out] normals array for normals
        */
      void
      fetchNormals(const DeviceArray<PointType>& cloud, DeviceArray<NormalType>& normals) const;

      /** \brief Downloads tsdf volume from GPU memory.           
        * \param[out] tsdf Array with tsdf values. if volume resolution is 512x512x512, so for voxel (x,y,z) tsdf value can be retrieved as volume[512*512*z + 512*y + x];
        */
      void
      downloadTsdf (std::vector<float>& tsdf) const;

      /** \brief Downloads TSDF volume and according voxel weights from GPU memory
        * \param[out] tsdf Array with tsdf values. if volume resolution is 512x512x512, so for voxel (x,y,z) tsdf value can be retrieved as volume[512*512*z + 512*y + x];
        * \param[out] weights Array with tsdf voxel weights. Same size and access index as for tsdf. A weight of 0 indicates the voxel was never used.
        */
      void
      downloadTsdfAndWeighs(std::vector<float>& tsdf, std::vector<short>& weights) const;

      /** \brief Downloads TSDF volume and according voxel weights from GPU memory in int representation for the pair
        */

      void
      downloadTsdfAndWeightsInt ();

      /** \brief Uploads TSDF volume and according voxel weights to GPU memory from int representation for the pair
        */

      void
      uploadTsdfAndWeightsInt ();

      /** \brief Set number of volumes.
          \param[in] num_volume How many volumes there are
      */
      static void
      setNumVolumes(int num_volumes);

      /** \brief Gets number of volumes. */
      static int
      getNumVolumes();

    private:
      /** \brief tsdf volume size in meters */
      Eigen::Vector3f size_;
      
      /** \brief tsdf volume resolution */
      Eigen::Vector3i resolution_;      

      /** \brief tsdf volume data container */
      DeviceArray2D<int> volume_;

      /** \brief tsdf truncation distance */
      float tranc_dist_;

      /** \brief tsdf location shift */
      Eigen::Vector3i shift_;

      /** \brief CPU memory container for points */
      std::vector<int> volume_downloaded_;

public:
EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    };
  }
}

#endif /* PCL_KINFU_TSDF_VOLUME_H_ */