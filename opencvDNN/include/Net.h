
#ifndef NET_H
#define NET_H

#include "methods.h"

using namespace std;

class Net {
	protected:
		float loss;
		cv::Mat target, board, output_error;
		vector<cv::Mat> layer, weights, bias, delta_err;

		// void initWeight(cv::Mat &dst, int type, double a, double b);	//initialise the weight matrix.if type =0,Gaussian.else uniform.
		cv::Mat activationFunction(cv::Mat &x, std::string func_type);	//Activation function

		void deltaError();												//Compute delta error
		void updateWeights();											//Update weights

	public:
        //Integer vector specifying the number of neurons in each layer including the input and output layers.
		vector<int> layer_neuron_num;
		vector<double> loss_vec;
		int output_interval = 10;
		float learning_rate, accuracy = 0. , fine_tune_factor = 1.01;
		string actvFunName = "sigmoid";

		Net() {};
		~Net() {};

		//Initialize net:: layer metrics, weights and bias metrics
		void initNet(std::vector<int> layer_neuron_num_);
		void initWeights(int type = 0, double a = 0., double b = 0.1);	//Initialize the weights metrics
		void initBias(const cv::Scalar& bias);								//Initialise the bias metrics.		// bias default all zero

		void forward();
		void backward();
		void train(cv::Mat input, cv::Mat target, float accuracy_threshold);							//Train,use accuracy_threshold
		void train(cv::Mat input, cv::Mat target_, float loss_threshold, bool draw_loss_curve = false);	//Train,use loss_threshold

		// train ---invoke--> forward() + backward() + draw_curve()
		// test ----invoke--> get_input_label() + predict():--->invode---> predict_one
		int predict_one(cv::Mat &input);					//Predict,just one sample
		vector<int> predict(cv::Mat &input);				//Predict,more  than one samples

		void load(std::string filename);					//Load model;
		void save(std::string filename);					//Save model;
		void test(cv::Mat &input, cv::Mat &target_);		//Test
};

#endif	// NET_H
