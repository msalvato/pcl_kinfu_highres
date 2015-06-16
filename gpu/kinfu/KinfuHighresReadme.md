Setup:

git clone https://github.com/msalvato/pcl_kinfu_highres

Follow these instructions to compile with GPU: http://pointclouds.org/documentation/tutorials/gpu_install.php

Note that you also need to set the OPENNI2 flag, this work allows for that.
Additionally you'll only want to use 'make pcl_kinfu_app', if you don't want to compile all of pcl as well.


Use:

There are a number of command line flags that can be set, all of them can be seen with 'pcl_kinfu_app -h'

We recommend always using the '-st' flags, which causes the trunctation of the TSDF to scale with size.

Example command 'pcl_kinfu_app -pcd pcd_files/ -r -ic -st -volume_size .5 -download_mesh 0 -num_volumes 8'

This command will read in the pcd files in the 'pcd_files' directory.
-ic allows for color, and -r turns on registration, which is necessary for color.
-st is scale trunction as mentioned above.
-volume_size defines the size of each volume, in this case .5m^3
-download_mesh 0 sets mesh download to false. Computers with less GPU memory may crash when downloading meshes. Pointclouds are still downloaded
-num_volumes sets the maximum number of TSDF volumes

This command will output a series of files with named 'cloud_x_y_z_frame.pcd', where x, y, and z represent the coordinates of the saved cloud, and
frame represents the frame the cloud was saved on. It's possible to save the same volume multiple times if it moves in and out of view. Any clouds
saved at the end of execution simply have the named 'cloud_x_y_z.pcd'. Meshes follow the same naming convention, but are instead '.ply' files.

To view progress while running, simply press 'h' while a window is highlighted to get a window of viewing commands.


Generating PCD files from LCM logs:

In order to generate a PCD file LCM logs such as those created with dustpan, use the lcmtopcd program located at
~/marine/projects/misc/msalvato/lcmtopcd

If you run 'convertToPcd -f lcmlog' on an lcmlog, it'll output a folder named 'pcd_files/' with the appropriate pcd files for use in kinfu_highres.