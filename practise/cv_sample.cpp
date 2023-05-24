
#include <iostream>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

// compilation command on archlinux : 
// g++ -o elf_cv_sample cv_sample.cpp -I/usr/include/opencv4/ -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_ml

int main(int argc, char ** argv)
{
    // cv::Mat monoImg(320,240,CV_8U,Scalar::all(100)) ;
    // imshow("mono", monoImg);

    cv::Mat allM(500, 500, CV_8UC3, cv::Scalar::all(200));
    cv::line(allM, cv::Point(0, 400), cv::Point(500, 400), Scalar::all(50), 2);
    cv::line(allM, cv::Point(100, 0), cv::Point(100, 500), Scalar::all(100), 5);
    imshow("allM", allM);

    waitKey(0);
    return 0;
}
