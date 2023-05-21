
#if defined (__linux__) || defined(__linux) || defined(linux)
	#include <opencv2/core/core.hpp>
	#include <opencv2/highgui/highgui.hpp>
	#include <opencv2/imgproc/imgproc.hpp>
#endif

#include <iostream>
using namespace std;
namespace eason
{
	cv::Mat sigmoid(cv::Mat &x) ;								// sigmoid function
	cv::Mat tanh(cv::Mat &x) ;									// Tanh function
	cv::Mat ReLU(cv::Mat &x) ;									// ReLU function
	cv::Mat derivativeFunction(cv::Mat& fx, string func_type);	// Derivative function

	// cv::Mat leakyReLU()
	// cv::Mat Maxout()
	// cv::Mat softmax()

	void calcLoss(cv::Mat& output, cv::Mat& target, cv::Mat& output_error, float& loss);	// Objective function

	//Get sample_number samples in XML file,from the start column.
	void get_input_label(std::string filename, cv::Mat& input, cv::Mat& label, int sample_num, int start = 0);
	// Draw loss curve
	void draw_curve(cv::Mat& board, std::vector<double> points);
}
