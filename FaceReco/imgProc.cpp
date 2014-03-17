//
//  imgProc.cpp
//  FaceReco
//
//  Created by Yawo Kpeglo - Business Lab on 17/03/2014.
//  Copyright (c) 2014 Business Lab. All rights reserved.
//

#include "imgProc.h"

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


Mat norm_0_255(InputArray _src) {
    Mat src = _src.getMat();
    // Create and return normalized image:
    Mat dst;
    switch(src.channels()) {
        case 1:
            cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC1);
            break;
        case 3:
            cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC3);
            break;
        default:
            src.copyTo(dst);
            break;
    }
    return dst;
}



Mat correctGamma( Mat src,double gamma ) {
    double inverse_gamma = 1.0 / gamma;
    
    Mat lut_matrix(1, 256, CV_8UC1 );
    uchar * ptr = lut_matrix.ptr();
    for( int i = 0; i < 256; i++ )
        ptr[i] = (int)( pow( (double) i / 255.0, inverse_gamma ) * 255.0 );
    
    Mat dst;
    LUT( src, lut_matrix, dst );
    // imshow("dst", dst);
    //  imshow("src", src);
    
    return dst;
}


Mat filterDoG( Mat src )
{
    int thresh = 15;
    int max_thresh = 50;
    
    int intSigmaBig = 70;
    int intMaxSigmaBig = 120;
    
    int intSigmaSmall = 60;
    int intMaxSigmaSmall = 120;
    
    
    cv::Mat filterResponse;
    float sigmaBig = intSigmaBig / 10.0f;
    float sigmaSmall = intSigmaSmall / 100.0f;
    
    // sigma = 0.3*((ksize-1)*0.5 - 1) + 0.8
    int ksize = ceilf((sigmaBig-0.8f)/0.3f)*2 + 3;
    
    cv::Mat gauBig = cv::getGaussianKernel(ksize, sigmaBig, CV_32F);
    cv::Mat gauSmall = cv::getGaussianKernel(ksize, sigmaSmall, CV_32F);
    
    cv::Mat DoG = gauSmall - gauBig;
    cv::sepFilter2D(src, filterResponse, CV_32F, DoG.t(), DoG);
    
    //imshow("Dog", DoG);
    
    filterResponse = cv::abs(filterResponse);
    
    std::cout << thresh << " : " << sigmaBig << " : " << sigmaSmall << std::endl;
    
    Mat dst = filterResponse.clone();
    
    //    for( int j = 0; j < filterResponse.rows ; j++ ) {
    //        for( int i = 0; i < filterResponse.cols; i++ ) {
    //            cv::Vec3f absPixel  = filterResponse.at<cv::Vec3f>(j, i);
    //
    //            if( (absPixel[0]+absPixel[1]+absPixel[2])/3 >= thresh ) {
    //                circle( dst, cv::Point( i, j ), 1, cv::Scalar(0, 0, 255), 2, 8, 0 );
    //            }
    //        }
    //    }
    
    //cv::resize(dst, dst, cv::Size(round(700 * dst.cols/dst.rows), 700));
    
    //cv::namedWindow( "filterDog", CV_WINDOW_AUTOSIZE );
    //cv::imshow( "filterDog", dst );
    
    return dst;
}

void dogCam(IplImage* src, IplImage* dst, int kernel1, int kernel2, int invert) {
    
    // Difference-Of-Gaussians (DOG) works by performing two different Gaussian blurs on the image,
    
    // with a different blurring radius for each, and subtracting them to yield the result.
    
    //   http://en.wikipedia.org/wiki/Difference_of_Gaussians
    
    //   http://docs.gimp.org/en/plug-in-dog.html
    
    IplImage *dog_1 = cvCreateImage(cvGetSize(src), src->depth, src->nChannels);
    
    IplImage *dog_2 = cvCreateImage(cvGetSize(src), src->depth, src->nChannels);
    
    cvSmooth(src, dog_2, CV_GAUSSIAN, kernel1, kernel1); // Gaussian blur
    
    cvSmooth(src, dog_1, CV_GAUSSIAN, kernel2, kernel2);
    
    cvSub(dog_2, dog_1, dst, 0);
    
} // dogCam()

Mat dogFilter( Mat src, int kernel1, int kernel2)
{
    Mat dst;
    Mat dog_1(src.rows,src.cols,src.depth());
    Mat dog_2(src.rows,src.cols,src.depth());
    
    GaussianBlur(src, dog_1, Size(7,7), kernel1);
    GaussianBlur(src, dog_2, Size(7,7), kernel2);
    
    subtract(dog_2, dog_1, dst);
    imshow("dog2>dog1", dst);
    
    //
    //    subtract(dog_1, dog_2, dst);
    //    imshow("dog2<dog1", dst);
    
    return dst;
}

Mat sqiFilter( Mat src )
{
    Mat dst;
    Mat gaussMat;
    GaussianBlur(src, gaussMat, Size(7,7), 1.5,1.5);
    //original/gaussMat
    divide(gaussMat, src, dst);
    
    
    return dst;
}


Mat dogCamMat( Mat src, int kernel1, int kernel2)
{
    Mat dst;
    
    cvtColor(src, src, CV_BGR2GRAY);
    src=correctGamma(src, 2);
    
    
    Mat dog_1(src.rows,src.cols,src.depth());
    Mat dog_2(src.rows,src.cols,src.depth());
    
    GaussianBlur(src, dog_2, Size(11,11), kernel1);
    GaussianBlur(src, dog_1, Size(11,11), kernel2);
    
    subtract(dog_2, dog_1, dst);
    
    subtract(255, dst, dst);
    
    //dst= norm_0_255(dst);
    equalizeHist(dst, dst);
    
    
    //dst=correctGamma(dst, 0.2);
    
    imshow("dog2>dog1", dst);
    
    //
    //    subtract(dog_1, dog_2, dst);
    //    imshow("dog2<dog1", dst);
    //invert(dst, dst);
    return dst;
    
} // Fin dogCamMat

