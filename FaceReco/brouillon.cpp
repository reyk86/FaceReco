////
////  brouillon.cpp
////  FaceReco
////
////  Created by Yawo Kpeglo - Business Lab on 25/02/2014.
////  Copyright (c) 2014 Business Lab. All rights reserved.
////
//
//#include "brouillon.h"
//
//#include <iostream>
//#include <fstream>
//#include <sstream>
//
//#include "opencv2/core/core.hpp"
//#include "opencv2/contrib/contrib.hpp"
//#include "opencv2/highgui/highgui.hpp"
//#include "opencv2/imgproc/imgproc.hpp"
//
//using namespace std;
//using namespace cv;
//////
//////static string fname(string filename)
//////{
//////    // Remove directory if present.
//////	// Do this before extension removal incase directory has a period character.
//////	const size_t last_slash_idx = filename.find_last_of("\\/");
//////	if (std::string::npos != last_slash_idx)
//////	{
//////		filename.erase(0, last_slash_idx + 1);
//////	}
//////    
//////	// Remove extension if present.
//////	const size_t period_idx = filename.rfind('_');
//////	if (std::string::npos != period_idx)
//////	{
//////		filename.erase(period_idx);
//////	}
//////    
//////	return filename;
//////}
//////
//////static void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, vector<string>& names, char separator = ';') {
//////    std::ifstream file(filename.c_str(), ifstream::in);
//////    if (!file) {
//////        string error_message = "No valid input file was given, please check the given filename.";
//////        CV_Error(CV_StsBadArg, error_message);
//////    }
//////    string line, path, classlabel, fileName;
//////    while (getline(file, line)) {
//////        stringstream liness(line);
//////        getline(liness, path, separator);
//////		fileName=fname(path);
//////        getline(liness, classlabel);
//////        if(!path.empty() && !classlabel.empty()&& !fileName.empty()) {
//////            images.push_back(imread(path, 0));
//////            labels.push_back(atoi(classlabel.c_str()));
//////			names.push_back(fileName.c_str());
//////        }
//////    }
//////}
//////
//////
//////int main(int argc, const char *argv[]) {
//////    // Check for valid command line arguments, print usage
//////    // if no arguments were given.
//////    /* if (argc != 4) {
//////     cout << "usage: " << argv[0] << " </path/to/haar_cascade> </path/to/csv.ext> </path/to/device id>" << endl;
//////     cout << "\t </path/to/haar_cascade> -- Path to the Haar Cascade for face detection." << endl;
//////     cout << "\t </path/to/csv.ext> -- Path to the CSV file with the face database." << endl;
//////     cout << "\t <device id> -- The webcam device id to grab frames from." << endl;
//////     exit(1);
//////     }*/
//////    // Get the path to your CSV:
//////    /*string fn_haar = string(argv[1]);
//////     string fn_csv = string(argv[2]);
//////     int deviceId = atoi(argv[3]);*/
//////    
//////    
//////	string fn_haar = "C:\\Users\\yawo\\OpenCV4Android\\FaceTracking\\src\\com\\blab\\opencv\\haarcascade_frontalface_alt.xml";
//////    string fn_csv = "C:\\Users\\yawo\\OpenCV4Android\\FaceTracking\\src\\com\\blab\\opencv\\savedFaces\\face.txt";
//////    int deviceId = 0;
//////    
//////    
//////	
//////    // These vectors hold the images and corresponding labels:
//////    vector<Mat> images;
//////    vector<int> labels;
//////	vector<string> names;
//////    
//////	
//////    
//////    
//////    // Read in the data (fails if no valid input filename is given, but you'll get an error message):
//////    try {
//////        read_csv(fn_csv, images, labels,names);
//////    } catch (cv::Exception& e) {
//////        cerr << "Error opening file \"" << fn_csv << "\". Reason: " << e.msg << endl;
//////        // nothing more we can do
//////        exit(1);
//////    }
//////    
//////    
//////	// Copy the names's vector into another with single name.
//////	vector<string> name;
//////	string actual_name="";
//////	for(string nom : names)
//////	{
//////		if(nom!=actual_name)
//////		{
//////			name.push_back(nom);
//////			actual_name=nom;
//////		}
//////		//cout<<"nom : " << nom << endl;
//////	}
//////    
//////	//cout<<"nom count: " << name.size() << endl;
//////	
//////    
//////    
//////    
//////    // Get the height from the first image. We'll need this
//////    // later in code to reshape the images to their original
//////    // size AND we need to reshape incoming faces to this size:
//////    int im_width = images[0].cols;
//////    int im_height = images[0].rows;
//////    
//////	//cout<<"images count: " << images.size() << " labels count: "  << labels.size()<< " images count: " << names.size() << endl;
//////	//cout<<"images width: " << im_width << " images height: "  << im_height << endl;
//////	/*int im_width = 125;
//////     int im_height = 150;*/
//////    
//////    // Create a FaceRecognizer and train it on the given images:
//////    Ptr<FaceRecognizer> model = createFisherFaceRecognizer();
//////    // Ptr<FaceRecognizer> model = createLBPHFaceRecognizer();
//////	Ptr<FaceRecognizer> Eigenmodel = createEigenFaceRecognizer();
//////	Ptr<FaceRecognizer> Fishermodel = createFisherFaceRecognizer();
//////    Ptr<FaceRecognizer> LBPHmodel = createLBPHFaceRecognizer();
//////    
//////    model->train(images, labels);
//////	Eigenmodel->train(images, labels);
//////	Fishermodel->train(images, labels);
//////	LBPHmodel->train(images, labels);
//////    
//////    // That's it for learning the Face Recognition model. You now
//////    // need to create the classifier for the task of Face Detection.
//////    // We are going to use the haar cascade you have specified in the
//////    // command line arguments:
//////    //
//////    CascadeClassifier haar_cascade;
//////    haar_cascade.load(fn_haar);
//////    // Get a handle to the Video device:
//////    VideoCapture cap(deviceId);
//////    // Check if we can use this device at all:
//////    if(!cap.isOpened()) {
//////        cerr << "Capture Device ID " << deviceId << "cannot be opened." << endl;
//////        return -1;
//////    }
//////    // Holds the current frame from the Video device:
//////    Mat frame;
//////    for(;;) {
//////        cap >> frame;
//////        // Clone the current frame:
//////        Mat original = frame.clone();
//////        // Convert the current frame to grayscale:
//////        Mat gray;
//////		if(! original.data )                              // Check for invalid input
//////        {
//////            cout <<  "Could not open or find the image" << std::endl ;
//////            //system("Pause");
//////            //exit(-1);
//////		}else
//////		{
//////            cvtColor(original, gray, CV_BGR2GRAY);
//////            
//////		}
//////        //cvtColor(original, gray, CV_BGR2GRAY);
//////        // Find the faces in the frame:
//////        vector< Rect_<int> > faces;
//////        haar_cascade.detectMultiScale(gray, faces);
//////        // At this point you have the position of the faces in
//////        // faces. Now we'll get the faces, make a prediction and
//////        // annotate it in the video. Cool or what?
//////        for(size_t i = 0; i < faces.size(); i++) {
//////            // Process face by face:
//////            Rect face_i = faces[i];
//////            // Crop the face from the image. So simple with OpenCV C++:
//////            Mat face = gray(face_i);
//////            // Resizing the face is necessary for Eigenfaces and Fisherfaces. You can easily
//////            // verify this, by reading through the face recognition tutorial coming with OpenCV.
//////            // Resizing IS NOT NEEDED for Local Binary Patterns Histograms, so preparing the
//////            // input data really depends on the algorithm used.
//////            //
//////            // I strongly encourage you to play around with the algorithms. See which work best
//////            // in your scenario, LBPH should always be a contender for robust face recognition.
//////            //
//////            // Since I am showing the Fisherfaces algorithm here, I also show how to resize the
//////            // face you have just found:
//////            Mat face_resized;
//////            cv::resize(face, face_resized, Size(im_width, im_height), 1.0, 1.0, INTER_CUBIC);
//////            // Now perform the prediction, see how easy that is:
//////            // int prediction = model->predict(face_resized);
//////			int predicted_label = -1;
//////			double predicted_confidence = 0.0;
//////			string detected="";
//////            
//////			
//////			int eprediction = Eigenmodel->predict(face_resized);
//////            int fprediction = Fishermodel->predict(face_resized);
//////            int lprediction = LBPHmodel->predict(face_resized);
//////            
//////            
//////			// Get the prediction and associated confidence from the model
//////			model->predict(face_resized, predicted_label, predicted_confidence);
//////            // And finally write all we've found out to the original image!
//////            // First of all draw a green rectangle around the detected face:
//////            rectangle(original, face_i, CV_RGB(0, 255,0), 1);
//////            // Create the text we will annotate the box with:
//////			detected= name.at(predicted_label);
//////            // string box_text = format("Prediction = %s Confidence = %d ", detected.c_str(), predicted_confidence);
//////            string box_text = format("%s", detected.c_str());
//////            // Calculate the position for annotated text (make sure we don't
//////            // put illegal values in there):
//////            int pos_x = std::max(face_i.tl().x - 10, 0);
//////            int pos_y = std::max(face_i.tl().y - 10, 0);
//////            // And now put it into the image:
//////            //putText(original, box_text, Point(pos_x, pos_y), FONT_HERSHEY_DUPLEX, 1.0, CV_RGB(0,255,0), 2.0);
//////			putText(original, box_text, Point(pos_x, pos_y), FONT_HERSHEY_DUPLEX, 1, CV_RGB(0,255,0), 2);
//////            
//////        }
//////        // Show the result:
//////        imshow("face_recognizer", original);
//////        // And display it:
//////        char key = (char) waitKey(20);
//////        // Exit this loop on escape:
//////        if(key == 27)
//////            break;
//////    }
//////	system("Pause");
//////    return 0;
//////}
////
//////
//////  main.cpp
//////  FaceReco
//////
//////  Created by Yawo Kpeglo - Business Lab on 24/02/2014.
//////  Copyright (c) 2014 Business Lab. All rights reserved.
//////
//////
////#include <iostream>
//////
//////int main(int argc, const char * argv[])
//////{
//////
//////    // insert code here...
//////    std::cout << "Hello, World!\n";
//////    return 0;
//////}
//////
////// Example showing how to read and write images
////#include <opencv2/opencv.hpp>
////#include <opencv2/highgui/highgui.hpp>
////#include <opencv/cvaux.h>
////
////using namespace cv;
////using namespace std;
////
////int main(int argc, char** argv)
////{
////    IplImage * pInpImg = 0;
////    
////    // Load an image from file - change this based on your image name
////    pInpImg = cvLoadImage("Original_Color.jpg", CV_LOAD_IMAGE_UNCHANGED);
////    if(!pInpImg)
////    {
////        fprintf(stderr, "failed to load input image\n");
////        return -1;
////    }
////    
////    // Write the image to a file with a different name,
////    // using a different image format -- .png instead of .jpg
////    if( !cvSaveImage("my_image_copy.png", pInpImg) )
////    {
////        fprintf(stderr, "failed to write image file\n");
////    }
////    
////    cvShowImage("image", pInpImg);
////    waitKey(0);
////    
////    
////    Mat src;
////    
////    /// Load an image
////    src = imread( "my_image_copy.png" );
////    if( !src.data )  { return -1; }
////    
////    imshow( "original", src );
////    
////    
////    int c= cvWaitKey(0);
////    // Remember to free image memory after using it!
////    cvReleaseImage(&pInpImg);
////    
////    if (c == 27)
////    {
////        return 0;
////    }
////}
//////int main ( int argc, char** argv )
//////{
//////    Mat src;
//////
//////    /// Load an image
//////    src = imread( "my_image_copy.png" );
//////    if( !src.data )  { return -1; }
//////
//////    imshow( "original", src );
//////    waitKey(0);
//////    return 0;
//////}
//
////
////  main.cpp
////  FaceReco
////
////  Created by Yawo Kpeglo - Business Lab on 24/02/2014.
////  Copyright (c) 2014 Business Lab. All rights reserved.
////
////
//#include "brouillon.h"
//
//#include <iostream>
//#include <fstream>
//#include <sstream>
//#include "libface/LibfaceUtils.h"
//
//#include "opencv2/core/core.hpp"
//#include "opencv2/contrib/contrib.hpp"
//#include "opencv2/highgui/highgui.hpp"
//#include "opencv2/imgproc/imgproc.hpp"
//
//#include "gaussianMat.h"
//
//#include "opencv/cv.h"
//#include "opencv/cvaux.h"
//
//# include <cmath>
//
//#include "Libface/libface.h"
//#include <pthread.h>
//
//using namespace std;
//using namespace cv;
//using namespace libface;
//
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
//void correctGamma( Mat src, Mat dst, double gamma ) {
//    double inverse_gamma = 1.0 / gamma;
//    
//    Mat lut_matrix(1, 256, CV_8UC1 );
//    uchar * ptr = lut_matrix.ptr();
//    for( int i = 0; i < 256; i++ )
//        ptr[i] = (int)( pow( (double) i / 255.0, inverse_gamma ) * 255.0 );
//    
//    //Mat result;
//    LUT( src, lut_matrix, dst );
//    imshow("dst", dst);
//    imshow("src", src);
//    
//}
//
//
//void filterDoG( Mat src , Mat dst )
//{
//    int thresh = 15;
//    int max_thresh = 50;
//    
//    int intSigmaBig = 70;
//    int intMaxSigmaBig = 120;
//    
//    int intSigmaSmall = 60;
//    int intMaxSigmaSmall = 120;
//    
//    
//    cv::Mat filterResponse;
//    float sigmaBig = intSigmaBig / 10.0f;
//    float sigmaSmall = intSigmaSmall / 100.0f;
//    
//    // sigma = 0.3*((ksize-1)*0.5 - 1) + 0.8
//    int ksize = ceilf((sigmaBig-0.8f)/0.3f)*2 + 3;
//    
//    cv::Mat gauBig = cv::getGaussianKernel(ksize, sigmaBig, CV_32F);
//    cv::Mat gauSmall = cv::getGaussianKernel(ksize, sigmaSmall, CV_32F);
//    
//    cv::Mat DoG = gauSmall - gauBig;
//    cv::sepFilter2D(src, filterResponse, CV_32F, DoG.t(), DoG);
//    
//    imshow("Dog", DoG);
//    
//    filterResponse = cv::abs(filterResponse);
//    
//    std::cout << thresh << " : " << sigmaBig << " : " << sigmaSmall << std::endl;
//    
//    dst = filterResponse.clone();
//    
//    for( int j = 0; j < filterResponse.rows ; j++ ) {
//        for( int i = 0; i < filterResponse.cols; i++ ) {
//            cv::Vec3f absPixel  = filterResponse.at<cv::Vec3f>(j, i);
//            
//            if( (absPixel[0]+absPixel[1]+absPixel[2])/3 >= thresh ) {
//                circle( dst, cv::Point( i, j ), 1, cv::Scalar(0, 0, 255), 2, 8, 0 );
//            }
//        }
//    }
//    
//    //cv::resize(dst, dst, cv::Size(round(700 * dst.cols/dst.rows), 700));
//    
//    cv::namedWindow( "filterDog", CV_WINDOW_AUTOSIZE );
//    cv::imshow( "filterDog", dst );
//}
//
//
//
//
//
//
//
//
//
//int main(int argc, const char *argv[]) {
//    
//    
//	string fn_haar = "haarcascade_frontalface_alt.xml";
//    // string fn_csv = "C:\\Users\\yawo\\OpenCV4Android\\FaceTracking\\src\\com\\blab\\opencv\\savedFaces\\face.txt";
//    int deviceId = 0;
//    
//    CascadeClassifier haar_cascade;
//    haar_cascade.load(fn_haar);
//    // Get a handle to the Video device:
//    VideoCapture cap(deviceId);
//    // Check if we can use this device at all:
//    if(!cap.isOpened()) {
//        cerr << "Capture Device ID " << deviceId << "cannot be opened." << endl;
//        return -1;
//    }
//    // Holds the current frame from the Video device:
//    Mat frame;
//    for(;;) {
//        cap >> frame;
//        // Clone the current frame:
//        Mat original = frame.clone();
//        // Convert the current frame to grayscale:
//        Mat gray, frame_gray;
//        
//		if(! original.data )                              // Check for invalid input
//        {
//            cout <<  "Could not open or find the image" << std::endl ;
//            //system("Pause");
//            exit(-1);
//		}
//        
//        
//        Mat gaussMat, sqiMat, gMat, dogMat, illMat;
//        
//        cv::GaussianBlur( original, gaussMat, cv::Size(5,5), 3, 3);
//        
//        //sqiMat = original/gaussMat;
//        
//        //namedWindow( "gaussMat", cv::WINDOW_AUTOSIZE );
//        //pyrDown(gaussMat, gaussMat);
//        //pyrDown(sqiMat, sqiMat);
//        //imshow("gaussianMat", gaussMat);
//        //imshow("sqiMat", sqiMat);
//        
//        //cvtColor(original, gray, CV_64FC1);
//        //cvtColor(gaussMat, frame_gray, CV_BGR2GRAY);
//        //Mat img;
//        
//        //cv::cvtColor(original, img, CV_BGR2YUV);
//        //std::vector<cv::Mat> channels;
//        //imshow("img", img);
//        
//        //cv::split(img, channels);
//        //cv::equalizeHist(channels[0], channels[0]);
//        //imshow("channels[0]", channels[0]);
//        
//        //cv::merge(channels, img);
//        //imshow("merge img", img);
//        
//        
//        
//        //cv::cvtColor(img, img, CV_BGR2GRAY);
//        //imshow("img Final", img);
//        
//        
//        cvtColor(original, illMat, CV_BGR2GRAY);
//        
//        //namedWindow( "Gamma", cv::WINDOW_NORMAL );
//        //Cv_Gamma(original, gMat, 2.2);
//        correctGamma( original,gMat, 2.2);
//        //filterDoG(original, dogMat);
//        //dog(original, dogMat);
//        
//        //        correctGamma( illMat,gMat, 0.2);
//        //        imshow("correctGamma", gMat);
//        //        filterDoG(gMat, dogMat);
//        //        imshow("filterDog", dogMat);
//        //        sqiMat=norm_0_255(dogMat);
//        //        imshow("norm_0_255", sqiMat);
//        //        equalizeHist(sqiMat, sqiMat);
//        //        imshow("equalizeHist", sqiMat);
//        // namedWindow( "SQI", cv::WINDOW_NORMAL );
//        
//        // imshow("SQI", sqiMat);
//        
//        
//        
//        
//        
//        //cvNamedWindow( "CvMat n", cv::WINDOW_NORMAL );
//        //namedWindow( "mf", cv::WINDOW_NORMAL );
//        //namedWindow( "mt", cv::WINDOW_NORMAL );
//        
//        
//        //        cv::Mat mf,mt;
//        //
//        //        // populate m
//        //
//        //        CvMat n = original; // cv::Mat::operator CvMat() const;
//        //
//        //        mf = cv::Mat(&n); // cv::Mat::Mat(const CvMat* m, bool copyData = false);
//        //        // or
//        //        mt = cv::Mat(&n, true); // to copy the data
//        
//        //cvShowImage("CvMat n", &n);
//        //imshow("mf", mf);
//        //imshow("mt", mt);
//        
//        
//        namedWindow( "Original", cv::WINDOW_AUTOSIZE );
//        pyrDown(original, original);
//        //imshow("Original", original);
//        //imshow("Gamma", gMat);
//        
//        
//        
//        //namedWindow( "gray", cv::WINDOW_AUTOSIZE );pyrDown(gray, gray);
//        //imshow("gray", gray);
//        
//        
//        //        equalizeHist(frame_gray, frame_gray) ;
//        //        namedWindow( "frame_gray", cv::WINDOW_AUTOSIZE );pyrDown(frame_gray, frame_gray);
//        //        imshow("frame_gray", frame_gray);
//        //
//        //        Mat norm_gray=norm_0_255(frame_gray);
//        //        imshow("norm_frame_gray", frame_gray);
//        
//        
//        
//        
//        
//        char key = (char) waitKey(20);
//        // Exit this loop on escape:
//        if(key == 27)
//            break;
//    }
//	system("Pause");
//    return 0;
//}
