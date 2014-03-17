//
//  main.cpp
//  FaceReco
//
//  Created by Yawo Kpeglo - Business Lab on 24/02/2014.
//  Copyright (c) 2014 Business Lab. All rights reserved.
//
//
#include "brouillon.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include "libface/LibfaceUtils.h"

#include "opencv2/core/core.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"


#include "opencv/cv.h"
#include "opencv/cvaux.h"

# include <cmath>

#include "Libface/libface.h"
#include <pthread.h>
#include "imgProcTest.h"
#include "imgProc.h"

using namespace std;
using namespace cv;
using namespace libface;


int main(int argc, const char *argv[]) {
    
    
	string fn_haar = "haarcascade_frontalface_alt.xml";
    // string fn_csv = "C:\\Users\\yawo\\OpenCV4Android\\FaceTracking\\src\\com\\blab\\opencv\\savedFaces\\face.txt";
    int deviceId = 0;
    
    CascadeClassifier haar_cascade;
    haar_cascade.load(fn_haar);
    // Get a handle to the Video device:
    VideoCapture cap(deviceId);
    // Check if we can use this device at all:
    if(!cap.isOpened()) {
        cerr << "Capture Device ID " << deviceId << "cannot be opened." << endl;
        return -1;
    }
    // Holds the current frame from the Video device:
    Mat frame;
    Mat src= imread("rachid.png",1);
    Mat dst;

    dst=dogCamMat(src, 15, 1);
    imshow("Rachid", dst);

    for(;;) {
        cap >> frame;
        // Clone the current frame:
        Mat original = frame.clone();
        // Convert the current frame to grayscale:
        Mat gray, frame_gray;
        
		if(! original.data )                              // Check for invalid input
        {
            cout <<  "Could not open or find the image" << std::endl ;
            //system("Pause");
            exit(-1);
		}
        
        Mat img;
        namedWindow( "img", cv::WINDOW_NORMAL );
        
        //        //namedWindow( "img", cv::WINDOW_NORMAL );
        //        namedWindow( "CvtColor", cv::WINDOW_NORMAL );
        //        namedWindow( "correctGamma", cv::WINDOW_NORMAL );
        //        namedWindow( "filterDoG", cv::WINDOW_NORMAL );
        //        namedWindow( "norm_0_255", cv::WINDOW_NORMAL );
        //        namedWindow( "equalizeHist", cv::WINDOW_NORMAL );
        //        namedWindow( "laplacien", cv::WINDOW_NORMAL );
        //        namedWindow( "sqiFilter", cv::WINDOW_NORMAL );
        //        namedWindow( "logtransform", cv::WINDOW_NORMAL );
        //
        //
        //
        //
        //        Mat img,img1,img2,img3,img4,img5, sqiMat, logMat,gaussBlurMat;
        //
        //
        //        // Passage en échelle de Gris
        //        cvtColor(original, img1, CV_BGR2GRAY);
        //        imshow("CvtColor", img1);
        //
        //        //imwrite("rachid.png",original);
        //
        //
        //        // Passage en Gamma
        //        img2=correctGamma(img1, 2.2);
        //        imshow("correctGamma", img2);
        //
        //        // Difference Gaussienne
        //        img3=filterDoG(img2);
        //        imshow("filterDoG", img3);
        //
        //        //log  transform
        //        log(img3,logMat);
        //        imshow("logtransform", logMat);
        //
        //        //        //Laplacien
        //        //        Laplacian( img2, img3, -1, 3,1.0,0.0,BORDER_DEFAULT);
        //        //        imshow("laplacien", img3);
        //
        //        // DogFilter
        //        img3=dogFilter(img2, 3, 11);
        //        imshow("DoGfilter", img3);
        //
        //        //        // sqiFilterFilter
        //        //        sqiMat=sqiFilter(img2);
        //        //        imshow("sqiFilter", sqiMat);
        //
        //
        //        //Normalisation
        //        img4=norm_0_255(img3);
        //        imshow("norm_0_255", img4);
        //
        //        //HistEqualizer
        //        equalizeHist(img4, img4);
        //        imshow("equalizeHist", img4);
        //        //imwrite("What's up.png", img4);
        //
        //        //Gaussian Blur
        //        GaussianBlur(img4, gaussBlurMat, Size(7,7), 1.5,1.5);
        //        imshow("GaussianBlur", gaussBlurMat);
        //
        //
        //        //Laplacien
        //        Laplacian( img4, img5, -1, 7,1.0,0.0,BORDER_DEFAULT);
        //        imshow("laplacien", img5);
        
        
        //My method
        
        img=dogCamMat(original, 15, 1);
        //GaussianBlur(img, img, Size(3,3), 0,0);

        imshow("img", img);
        
        
        
        namedWindow( "Original", cv::WINDOW_AUTOSIZE );
        pyrDown(original, original);
        imshow("Original", original);
        
        
        char key = (char) waitKey(20);
        // Exit this loop on escape:
        if(key == 27)
            break;
    }
	system("Pause");
    return 0;
}