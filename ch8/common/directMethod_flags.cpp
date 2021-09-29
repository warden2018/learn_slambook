//
// Created by yang on 2021/3/29.
//

#include "directMethod_flags.h"

DEFINE_string(left_img_file, "../left.png","left image file dir.");
DEFINE_string(disparity_file, "../disparity.png","Disparity image of the left and right image.");
DEFINE_int32(nPoints,2000,"Sample number in an image.");
DEFINE_int32(nBorder,20,"Border size of sample points in image 1. To avoid too close to the border.");
DEFINE_int32(halfPatchSize,3,"Half patch size for an interest patch in image.");
DEFINE_int32(interations,10,"Iterations for the GN method.");
DEFINE_int32(img_number,5,"Image number of the camera2……");
DEFINE_int32(pyramids,4,"How many pyramids in DirectPoseEstimationMultiLayer.");
DEFINE_double(pyr_scale,0.5,"The scale between the neighbor pyramid images.");
