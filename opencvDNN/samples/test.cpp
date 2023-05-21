#include "Net.h"

using namespace std;
using namespace cv;
using namespace eason;

#define SAMP_NUM			200
#define START 				800

#define CONTACT5A(a,b,c,d,e) a##b##c##d##e
#define CONTACT5B(a,b,c,d,e) CONTACT5A(a,b,c,d,e)
#define STR1(x) #x
#define STR2(x) STR1(x)

#define MS_P5 CONTACT5B(xmls/models/model_sigmoid_,START,_,SAMP_NUM,_test)
#define XML_MODEL_SIGMOID_TEST  STR2(MS_P5) ".xml"

#define XML_INPUT_LABEL			"xmls/data/input_label_1000.xml"

int main(int argc, char *argv[])
{
	//Set neuron number of every layer
	vector<int> layer_neuron_num = { 784,100,10 };
	//Set loss threshold,learning rate and activation function
	float loss_threshold = 0.5f;
	// Initialise Net and weights
	Net net;
	net.learning_rate = 0.3;
	net.output_interval = 2;
	net.actvFunName = "sigmoid";

	net.initNet(layer_neuron_num);
	net.initWeights(0, 0., 0.01);
	net.initBias(Scalar(0.05));

	//Get test samples and test samples 
	Mat input, label,test_input,test_label;
	int sample_number = 200;
	eason::get_input_label("../resources/" XML_INPUT_LABEL, input, label, sample_number);
	eason::get_input_label("../resources/" XML_INPUT_LABEL, test_input, test_label, SAMP_NUM, START);

	//convert label from 0---1 to -1---1,cause tanh function range is [-1,1]
	//label = 2 * label - 1;

	//Train,and draw the loss curve(cause the last parameter is ture) and test the trained net
	net.train(input, label, loss_threshold,true);
	net.test(test_input, test_label);

	//Save the model
	net.save("../resources/" XML_MODEL_SIGMOID_TEST);

	getchar();
	return 0;
}
