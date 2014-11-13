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

#ifndef PCL_KINFU_COLOR_VOLUME_H_
#define PCL_KINFU_COLOR_VOLUME_H_

#include <pcl/pcl_macros.h>
#include <pcl/gpu/containers/device_array.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <Eigen/Core>

namespace pcl
{
  namespace gpu
  {
    class TsdfVolume;

    /** \brief ColorVolume class
      * \author Anatoly Baskeheev, Itseez Ltd, (myname.mysurname@mycompany.com)
      */
    class PCL_EXPORTS ColorVolume
    {
    public:
      typedef PointXYZ PointType;
      typedef boost::shared_ptr<ColorVolume> Ptr;

      /** \brief Constructor
        * \param[in] tsdf tsdf volume to get parameters from
        * \param[in] max_weight max weight for running average. Can be less than 255. Negative means default.
        */
      ColorVolume(const TsdfVolume& tsdf, int max_weight = -1);

      /** \brief Desctructor */
      ~ColorVolume();

      /** \brief Resets color volume to uninitialized state */
      void
      reset();

      /** \brief Releases this color volume's data from gpu memory */
      void 
      release();

      /** \brief Returns running average length */
      int
      getMaxWeight() const;

      /** \brief Returns container with color volume in GPU memory */
      DeviceArray2D<int>
      data() const;

      /** \brief Computes colors from color volume
        * \param[in] cloud Points for which colors are to be computed.
        * \param[out] colors output array for colors
        */
      void
      fetchColors (const DeviceArray<PointType>& cloud, DeviceArray<RGB>& colors) const; 

      /** \brief Downloads TSDF volume and according voxel weights from GPU memory in int representation for the pair
        */

      void
      downloadColorAndWeightsInt ();

      /** \brief Uploads TSDF volume and according voxel weights to GPU memory from int representation for the pair
        */

      void
      uploadColorAndWeightsInt ();

    private:
      /** \brief Volume resolution */
      Eigen::Vector3i resolution_;

      /** \brief Volume size in meters */
      Eigen::Vector3f volume_size_;

      /** \brief Length of running average */
      int max_weight_;     

      /** \brief color volume data */
      DeviceArray2D<int> color_volume_;

      /** \brief tsdf location shift */
      Eigen::Vector3i shift_;

      /** \brief CPU memory container for points */
      std::vector<int> color_volume_downloaded_;

public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    };
  }
}

#endif /* PCL_KINFU_COLOR_VOLUME_H_ */