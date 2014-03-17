//
//  imgProcTest.h
//  FaceReco
//
//  Created by Yawo Kpeglo - Business Lab on 06/03/2014.
//  Copyright (c) 2014 Business Lab. All rights reserved.
//

#ifndef __FaceReco__imgProcTest__
#define __FaceReco__imgProcTest__

#include <iostream>

#include "imgProcTest.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "opencv2/core/core.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
using namespace std;

/***********************************************************************************************/
/***********************************************************************************************/
/************************************** TEST DES FILTRES ***************************************/
/***********************************************************************************************/
/***********************************************************************************************/



/************************************** GAUSSIAN BLUR ou FLOU GAUSSIEN ***************************************/

void GaussianBlurTest(Mat src, Mat dst, string textBuffer, string processFenetre);

/************************************** BLUR ***************************************/

void BlurTest(Mat src, Mat dst, string textBuffer, string processFenetre);

/************************************** BOX FILTER ***************************************/

void BoxFilterTest(Mat src, Mat dst, string textBuffer, string processFenetre);

/************************************** MEDIAN BLUR ***************************************/

void MedianBlurTest(Mat src, Mat dst, string textBuffer, string processFenetre);

/************************************** FILTER 2D ***************************************/

void Filter2DTest(Mat src, Mat dst, string textBuffer, string processFenetre);

/************************************** Laplacian ***************************************/

void LaplacianTest(Mat src, Mat dst, string textBuffer, string processFenetre);

/************************************** Sobel ***************************************/

void SobelTest(Mat src, Mat dst, string textBuffer, string processFenetre);

/************************************** Sobel ***************************************/

void Sobel2Test(Mat src, Mat dst, string textBuffer, string processFenetre);

/************************************** Scharr ***************************************/

void ScharrTest(Mat src, Mat dst, string textBuffer, string processFenetre);


#endif /* defined(__FaceReco__imgProcTest__) */
