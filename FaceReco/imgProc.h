//
//  imgProc.h
//  FaceReco
//
//  Created by Yawo Kpeglo - Business Lab on 17/03/2014.
//  Copyright (c) 2014 Business Lab. All rights reserved.
//

#ifndef __FaceReco__imgProc__
#define __FaceReco__imgProc__

#include <iostream>
#include <iostream>
#include <fstream>
#include <sstream>

#include "opencv2/core/core.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"


#include "opencv/cv.h"
#include "opencv/cvaux.h"

# include <cmath>

#include <pthread.h>
#include "imgProcTest.h"

using namespace std;
using namespace cv;


Mat norm_0_255(InputArray _src) ;


Mat correctGamma( Mat src,double gamma );

Mat filterDoG( Mat src );

void dogCam(IplImage* src, IplImage* dst, int kernel1, int kernel2, int invert) ;// dogCam()

Mat dogFilter( Mat src, int kernel1, int kernel2);
Mat sqiFilter( Mat src );
Mat dogCamMat( Mat src, int kernel1, int kernel2);

// Fin dogCamMat


#endif /* defined(__FaceReco__imgProc__) */
