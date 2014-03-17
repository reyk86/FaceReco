//
//  imgProcTest.cpp
//  FaceReco
//
//  Created by Yawo Kpeglo - Business Lab on 06/03/2014.
//  Copyright (c) 2014 Business Lab. All rights reserved.
//

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

void GaussianBlurTest(Mat src, Mat dst, string textBuffer, string processFenetre)
{
    
    String process_name="GaussianBlur";
    
    for ( int i = 1; i < 31; i = i + 2 )
    {
        //Copy text in textBuffer
        textBuffer= format("Kernel Size : %d x %d", i, i);
        
        //Apply GaussianBlur on the image in the "src" and save it to "dst"
        GaussianBlur( src, dst, Size( i, i ),0.0,0.0,BORDER_DEFAULT);
        
        //Put the text in textBuffer on "dst" image
        putText( dst, textBuffer, Point( src.cols/4, src.rows/8), CV_FONT_HERSHEY_COMPLEX, 1, Scalar(255, 255, 255) );
        
        //Show "dst" ilage with text
        imshow( processFenetre, dst );
        
        
        String filename = format("processed_images/%s %d x %d.png",process_name.c_str(), i, i);
        imwrite( filename, dst );
        
        //Wait 2 sec
        int c = waitKey(2000);
        
        //if "Esc" exit
        if (c == 27)
        {
            break;
        }
    }
    
}


/************************************** BLUR ***************************************/

void BlurTest(Mat src, Mat dst, string textBuffer, string processFenetre)
{
    
    String process_name="Blur";
    
    for ( int i = 1; i < 31; i = i + 2 )
    {
        //Copy text in textBuffer
        textBuffer= format("Kernel Size : %d x %d", i, i);
        
        //Apply GaussianBlur on the image in the "src" and save it to "dst"
        
        //GaussianBlur( src, dst, Size( i, i ),0.0,0.0,BORDER_DEFAULT);
        blur(src, dst, Size( i, i ));
        
        //Put the text in textBuffer on "dst" image
        putText( dst, textBuffer, Point( src.cols/4, src.rows/8), CV_FONT_HERSHEY_COMPLEX, 1, Scalar(255, 255, 255) );
        
        //Show "dst" ilage with text
        imshow( processFenetre, dst );
        
        
        String filename = format("processed_images/%s %d x %d.png",process_name.c_str(), i, i);
        imwrite( filename, dst );
        
        //Wait 2 sec
        int c = waitKey(2000);
        
        //if "Esc" exit
        if (c == 27)
        {
            break;
        }
    }
    
}

/************************************** BOX FILTER ***************************************/

void BoxFilterTest(Mat src, Mat dst, string textBuffer, string processFenetre)
{
    
    String process_name="Box Filter";
    
    for ( int i = 1; i < 31; i = i + 2 )
    {
        //Copy text in textBuffer
        textBuffer= format("Kernel Size : %d x %d", i, i);
        
        //Apply filter on the image in the "src" and save it to "dst"
        
        //GaussianBlur( src, dst, Size( i, i ),0.0,0.0,BORDER_DEFAULT);
        //blur(src, dst, Size( i, i ));
        
        boxFilter(src, dst, -1, Size( i, i ),Point(-1,-1));
        
        //Put the text in textBuffer on "dst" image
        putText( dst, textBuffer, Point( src.cols/4, src.rows/8), CV_FONT_HERSHEY_COMPLEX, 1, Scalar(255, 255, 255) );
        
        //Show "dst" ilage with text
        imshow( processFenetre, dst );
        
        
        String filename = format("processed_images/%s %d x %d.png",process_name.c_str(), i, i);
        imwrite( filename, dst );
        
        //Wait 2 sec
        int c = waitKey(2000);
        
        //if "Esc" exit
        if (c == 27)
        {
            break;
        }
    }
    
}
/************************************** MEDIAN BLUR ***************************************/

void MedianBlurTest(Mat src, Mat dst, string textBuffer, string processFenetre)
{
    
    String process_name="Median Blur";
    
    for ( int i = 1; i < 31; i = i + 2 )
    {
        //Copy text in textBuffer
        textBuffer= format("Aperture Size : %d" , i);
        
        //Apply filter on the image in the "src" and save it to "dst"
        
        //GaussianBlur( src, dst, Size( i, i ),0.0,0.0,BORDER_DEFAULT);
        //blur(src, dst, Size( i, i ));
        //boxFilter(src, dst, -1, Size( i, i ),Point(-1,-1));
        medianBlur(src, dst, i);
        
        //Put the text in textBuffer on "dst" image
        putText( dst, textBuffer, Point( src.cols/4, src.rows/8), CV_FONT_HERSHEY_COMPLEX, 1, Scalar(255, 255, 255) );
        
        //Show "dst" ilage with text
        imshow( processFenetre, dst );
        
        
        String filename = format("processed_images/%s %d.png",process_name.c_str(), i);
        imwrite( filename, dst );
        
        //Wait 2 sec
        int c = waitKey(2000);
        
        //if "Esc" exit
        if (c == 27)
        {
            break;
        }
    }
    
}

/************************************** FILTER 2D ***************************************/

void Filter2DTest(Mat src, Mat dst, string textBuffer, string processFenetre)
{
    
    String process_name="Filter2D";
    
    for ( int i = 1; i < 31; i = i + 2 )
    {
        //Copy text in textBuffer
        textBuffer= format("kernel Size : %d" , i);
        int kernel_size= i;
        Mat kernel = Mat::ones( kernel_size, kernel_size, CV_32F )/ (float)(kernel_size*kernel_size);
        
        //Apply filter on the image in the "src" and save it to "dst"
        
        //GaussianBlur( src, dst, Size( i, i ),0.0,0.0,BORDER_DEFAULT);
        //blur(src, dst, Size( i, i ));
        //boxFilter(src, dst, -1, Size( i, i ),Point(-1,-1));
        //medianBlur(src, dst, i);
        filter2D(src, dst, -1, kernel, Point( -1, -1 ), 0, BORDER_DEFAULT );
        
        //Put the text in textBuffer on "dst" image
        putText( dst, textBuffer, Point( src.cols/4, src.rows/8), CV_FONT_HERSHEY_COMPLEX, 1, Scalar(255, 255, 255) );
        
        //Show "dst" ilage with text
        imshow( processFenetre, dst );
        
        
        String filename = format("processed_images/%s %d.png",process_name.c_str(), i);
        imwrite( filename, dst );
        
        //Wait 2 sec
        int c = waitKey(2000);
        
        //if "Esc" exit
        if (c == 27)
        {
            break;
        }
    }
    
}

/************************************** Laplacian ***************************************/

void LaplacianTest(Mat src, Mat dst, string textBuffer, string processFenetre)
{
    
    String process_name="Laplacian";
    
    for ( int i = 1; i < 31; i = i + 2 )
    {
        //Copy text in textBuffer
        textBuffer= format("kernel Size : %d" , i);
        int kernel_size= i;
        Mat kernel = Mat::ones( kernel_size, kernel_size, CV_32F )/ (float)(kernel_size*kernel_size);
        
        //Apply filter on the image in the "src" and save it to "dst"
        
        //GaussianBlur( src, dst, Size( i, i ),0.0,0.0,BORDER_DEFAULT);
        //blur(src, dst, Size( i, i ));
        //boxFilter(src, dst, -1, Size( i, i ),Point(-1,-1));
        //medianBlur(src, dst, i);
        //filter2D(src, dst, -1, kernel, Point( -1, -1 ), 0, BORDER_DEFAULT );
        Laplacian( src, dst, -1, i,1.0,0.0,BORDER_DEFAULT);
        
        //Put the text in textBuffer on "dst" image
        putText( dst, textBuffer, Point( src.cols/4, src.rows/8), CV_FONT_HERSHEY_COMPLEX, 1, Scalar(255, 255, 255) );
        
        //Show "dst" ilage with text
        imshow( processFenetre, dst );
        
        
        String filename = format("processed_images/%s %d.png",process_name.c_str(), i);
        imwrite( filename, dst );
        
        //Wait 2 sec
        int c = waitKey(2000);
        
        //if "Esc" exit
        if (c == 27)
        {
            break;
        }
    }
    
}

/************************************** Sobel ***************************************/

void SobelTest(Mat src, Mat dst, string textBuffer, string processFenetre)
{
    
    String process_name="Sobel";
    
    for ( int i = 1; i < 31; i = i + 2 )
    {
        //Copy text in textBuffer
        textBuffer= format("kernel Size : %d" , i);
        int kernel_size= i;
        Mat kernel = Mat::ones( kernel_size, kernel_size, CV_32F )/ (float)(kernel_size*kernel_size);
        
        //Apply filter on the image in the "src" and save it to "dst"
        
        //GaussianBlur( src, dst, Size( i, i ),0.0,0.0,BORDER_DEFAULT);
        //blur(src, dst, Size( i, i ));
        //boxFilter(src, dst, -1, Size( i, i ),Point(-1,-1));
        //medianBlur(src, dst, i);
        //filter2D(src, dst, -1, kernel, Point( -1, -1 ), 0, BORDER_DEFAULT );
        //Laplacian( src, dst, -1, i,1.0,0.0,BORDER_DEFAULT);
        Sobel(src, dst, -1, 1,1,i,1.0,0.0,BORDER_DEFAULT);
        
        
        //Put the text in textBuffer on "dst" image
        putText( dst, textBuffer, Point( src.cols/4, src.rows/8), CV_FONT_HERSHEY_COMPLEX, 1, Scalar(255, 255, 255) );
        
        //Show "dst" ilage with text
        imshow( processFenetre, dst );
        
        
        String filename = format("processed_images/%s %d.png",process_name.c_str(), i);
        imwrite( filename, dst );
        
        //Wait 2 sec
        int c = waitKey(2000);
        
        //if "Esc" exit
        if (c == 27)
        {
            break;
        }
    }
    
}
/************************************** Sobel ***************************************/

void Sobel2Test(Mat src, Mat dst, string textBuffer, string processFenetre)
{
    
    String process_name="Sobel2_";
    
    for ( int i = 1; i < 31; i = i + 2 )
    {
        //Copy text in textBuffer
        textBuffer= format("Kernel: %d Scale: %i" , i, i);
        int kernel_size= i;
        Mat kernel = Mat::ones( kernel_size, kernel_size, CV_32F )/ (float)(kernel_size*kernel_size);
        
        //Apply filter on the image in the "src" and save it to "dst"
        
        //GaussianBlur( src, dst, Size( i, i ),0.0,0.0,BORDER_DEFAULT);
        //blur(src, dst, Size( i, i ));
        //boxFilter(src, dst, -1, Size( i, i ),Point(-1,-1));
        //medianBlur(src, dst, i);
        //filter2D(src, dst, -1, kernel, Point( -1, -1 ), 0, BORDER_DEFAULT );
        //Laplacian( src, dst, -1, i,1.0,0.0,BORDER_DEFAULT);
        Sobel(src, dst, -1, 1,1,i,i,0.0,BORDER_DEFAULT);
        
        
        //Put the text in textBuffer on "dst" image
        putText( dst, textBuffer, Point( src.cols/4, src.rows/8), CV_FONT_HERSHEY_COMPLEX, 1, Scalar(255, 255, 255) );
        
        //Show "dst" ilage with text
        imshow( processFenetre, dst );
        
        
        String filename = format("processed_images/%s k%d S%d.png",process_name.c_str(), i, i);
        imwrite( filename, dst );
        
        //Wait 2 sec
        int c = waitKey(2000);
        
        //if "Esc" exit
        if (c == 27)
        {
            break;
        }
    }
    
}

/************************************** Scharr ***************************************/

void ScharrTest(Mat src, Mat dst, string textBuffer, string processFenetre)
{
    
    String process_name="Scharr";
    
    for ( int i = 1; i < 31; i = i + 2 )
    {
        //Copy text in textBuffer
        textBuffer= format("scale : %d" , i);
        int kernel_size= i;
        Mat kernel = Mat::ones( kernel_size, kernel_size, CV_32F )/ (float)(kernel_size*kernel_size);
        
        //Apply filter on the image in the "src" and save it to "dst"
        
        //GaussianBlur( src, dst, Size( i, i ),0.0,0.0,BORDER_DEFAULT);
        //blur(src, dst, Size( i, i ));
        //boxFilter(src, dst, -1, Size( i, i ),Point(-1,-1));
        //medianBlur(src, dst, i);
        //filter2D(src, dst, -1, kernel, Point( -1, -1 ), 0, BORDER_DEFAULT );
        //Laplacian( src, dst, -1, i,1.0,0.0,BORDER_DEFAULT);
        //Sobel(src, dst, -1, 1,1,i,1.0,0.0,BORDER_DEFAULT);
        Scharr(src, dst, -1, 1,0,i,0.0,BORDER_DEFAULT);
        
        
        //Put the text in textBuffer on "dst" image
        putText( dst, textBuffer, Point( src.cols/4, src.rows/8), CV_FONT_HERSHEY_COMPLEX, 1, Scalar(255, 255, 255) );
        
        //Show "dst" ilage with text
        imshow( processFenetre, dst );
        
        
        String filename = format("processed_images/%s %d.png",process_name.c_str(), i);
        imwrite( filename, dst );
        
        //Wait 2 sec
        int c = waitKey(2000);
        
        //if "Esc" exit
        if (c == 27)
        {
            break;
        }
    }
    
}


/************************************** Differecnce of Gaussians ***************************************/
//
//static Mat norm_0_255(InputArray _src) {
//    Mat src = _src.getMat();
//    // Create and return normalized image:
//    Mat dst;
//    switch(src.channels()) {
//        case 1:
//            cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC1);
//            break;
//        case 3:
//            cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC3);
//            break;
//        default:
//            src.copyTo(dst);
//            break;
//    }
//    return dst;
//}
//
//
//
//Mat correctGamma( Mat src,double gamma ) {
//    double inverse_gamma = 1.0 / gamma;
//    
//    Mat lut_matrix(1, 256, CV_8UC1 );
//    uchar * ptr = lut_matrix.ptr();
//    for( int i = 0; i < 256; i++ )
//        ptr[i] = (int)( pow( (double) i / 255.0, inverse_gamma ) * 255.0 );
//    
//    Mat dst;
//    LUT( src, lut_matrix, dst );
//    // imshow("dst", dst);
//    //  imshow("src", src);
//    
//    return dst;
//}
//
//
//Mat dogCamMat( Mat src, int kernel1, int kernel2)
//{
//    Mat dst;
//    
//    cvtColor(src, src, CV_BGR2GRAY);
//    src=correctGamma(src, 2);
//
//    
//    Mat dog_1(src.rows,src.cols,src.depth());
//    Mat dog_2(src.rows,src.cols,src.depth());
//    
//    GaussianBlur(src, dog_2, Size(11,11), kernel1);
//    GaussianBlur(src, dog_1, Size(11,11), kernel2);
//    
//    subtract(dog_2, dog_1, dst);
//    
//    subtract(255, dst, dst);
//
//    dst= norm_0_255(dst);
//    equalizeHist(dst, dst);
//
//    
//    //dst=correctGamma(dst, 0.2);
//    
//    imshow("dog2>dog1", dst);
//    
//    //
//    //    subtract(dog_1, dog_2, dst);
//    //    imshow("dog2<dog1", dst);
//    //invert(dst, dst);
//    return dst;
//    
//} // Fin dogCamMat
//







/*int main(int argc, char** argv)
{
    String textBuffer, originFenetre="Original Image", processFenetre="Processed Image";
	// Création des fenêtres d'affichage
	namedWindow(originFenetre, CV_WINDOW_AUTOSIZE);
    namedWindow(processFenetre, CV_WINDOW_AUTOSIZE);
    
    
    // Chargement de l'image
	Mat src= imread("rachid.png",1);
    Mat dst;
    
	//Afficher l'image chargée
    imshow( originFenetre, src );
    
    // Application des filtres
    
    //GaussianBlurTest(src, dst,textBuffer,processFenetre);
    //BlurTest(src, dst, textBuffer, processFenetre);
    //BoxFilterTest(src, dst, textBuffer, processFenetre);
    //MedianBlurTest(src, dst, textBuffer, processFenetre);
    //Filter2DTest(src, dst, textBuffer, processFenetre);
    //LaplacianTest(src, dst, textBuffer, processFenetre);
    //SobelTest(src, dst, textBuffer, processFenetre);
    //Sobel2Test(src, dst, textBuffer, processFenetre);
    //ScharrTest(src, dst, textBuffer, processFenetre);

    dst=dogCamMat(src, 15, 1);
    
    dst = Mat::zeros( src.size(), src.type() );
    
    textBuffer=format("Press Any Key to Exit");
    
    putText( dst, textBuffer, Point( src.cols/4,  src.rows / 2), CV_FONT_HERSHEY_COMPLEX, 1, Scalar(255, 255, 255) );
    
    imshow( processFenetre, dst );
    
    waitKey(0);
    
    return 0;
    
} */