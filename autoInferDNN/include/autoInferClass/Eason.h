#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>
using namespace std;

#define input_i1 0.05
#define input_i2 0.1
#define output_o1 0.13
#define output_o2 0.99
#define PRECISION	9
#define MAGIC_EXCLUDE_double 98765.01234f

// name rule for suffix :
//		h1n2 means this parameter works on 1'st hidden layer with 2 neurons
//		end means this parameter works on end layers

typedef  double (*Fun_Ret_double_In_double)(double);

double actvFunSigmoid(double in);
double actvFunTanH(double in);
double derivativeCalc(Fun_Ret_double_In_double funPtr, double fun_in);
Fun_Ret_double_In_double getActivationFun(string funName);

class Eason
{
	private:
		vector<double> weights_h1n2 = {0.15,0.2,0.25,0.3}, weights_end = {0.40, 0.45, 0.50, 0.55};
		double bias_h1n2 = 0.35, bias_end=0.6 , learning_rate = 0.5 ;
		vector<double> layer_output_auxiliary;

		double auxiliary_out_hidden(unsigned char hLayerOrder);
		double auxiliary_out_end(Fun_Ret_double_In_double fun,unsigned char eLayerOrder);

		void updateWeights(Fun_Ret_double_In_double fun);
		void restoreWeights(void);
		void showLastWeights(string str);
		void forward(Fun_Ret_double_In_double fun);
		double backward(Fun_Ret_double_In_double fun, unsigned int turns, unsigned int last_turns, bool logOut,bool autoTrain, double mse_threshold);
	public:
		Eason();
		virtual ~Eason();
		vector<double> input = {0.05,0.10}, target_output = {0.01, 0.99};
		signed char setWeightBias(	vector<double> weights_h1n2_in, double bias_h1n2_in,
									vector<double> weights_end_in, double bias_end_in, double learning_rate_in);
		signed char setInput_TargetOutput(vector<double> input, vector<double> target_output);
		void train(unsigned int turns=20, bool logOut=true, bool autoTrain=false, double mse_threshold=0.0);
};
