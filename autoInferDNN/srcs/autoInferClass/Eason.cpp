#include "Eason.h"
__asm__(".symver realpath,realpath@GLIBC_2.2.5");
double actvFunSigmoid(double in)
{
	return 1/(1+exp(-1*in));
}
double actvFunTanH(double in)
{
	return (exp(in)-exp(-1*in))/(exp(in)+exp(-1*in));
}
double derivativeCalc(Fun_Ret_double_In_double funPtr, double fun_in)
{
	if ( funPtr == actvFunSigmoid )
		return fun_in * ( 1 - fun_in );
	else if ( funPtr == actvFunTanH )
		return 1-pow(fun_in , 2);
	else
		return MAGIC_EXCLUDE_double;
}
Fun_Ret_double_In_double getActivationFun(string funName)
{
	if ( funName == "sigmoid" )
		return actvFunSigmoid;
	else if ( funName == "tanh")
		return actvFunTanH;
	else
		return nullptr;
}

double Eason::auxiliary_out_hidden(unsigned char hLayerOrder)
{
	if(1 == hLayerOrder)
		return this->weights_h1n2[1-1] * this->input[1-1] + this->weights_h1n2[2-1] * this->input[2-1] + this->bias_h1n2 ;
	else if (2 == hLayerOrder)
		return this->weights_h1n2[3-1] * this->input[1-1] + this->weights_h1n2[4-1] * this->input[2-1] + this->bias_h1n2 ;
	else return MAGIC_EXCLUDE_double;
}
double Eason::auxiliary_out_end(Fun_Ret_double_In_double fun, unsigned char eLayerOrder)
{
	if(1 == eLayerOrder)
		return 	this->weights_end[5-5]*(*fun)(auxiliary_out_hidden(1)) + \
				this->weights_end[6-5]*(*fun)(auxiliary_out_hidden(2)) + this->bias_end ;
	else if (2 == eLayerOrder)
		return 	this->weights_end[7-5]*(*fun)(auxiliary_out_hidden(1)) + \
				this->weights_end[8-5]*(*fun)(auxiliary_out_hidden(2)) + this->bias_end ;
	else return MAGIC_EXCLUDE_double ;
}
signed char Eason::setWeightBias(vector<double> weights_h1n2_in, double bias_h1n2_in,
	vector<double> weights_end_in, double bias_end_in, double learning_rate_in)
{
	if(weights_h1n2_in.size() == this->weights_h1n2.size()) {
		for(int i = 0 ; i< weights_h1n2_in.size(); i++) {
			this->weights_h1n2[i] = weights_h1n2_in[i];
		}
	} else return -1;

	if(weights_end_in.size() == this->weights_end.size()) {
		for(int i = 0 ; i< weights_end_in.size(); i++) {
			this->weights_end[i] = weights_end_in[i];
		}
	} else return -1;

	this->bias_h1n2 = bias_h1n2_in;
	this->bias_end = bias_end_in;
	this->learning_rate = learning_rate_in;
	return 0;
}
signed char Eason::setInput_TargetOutput(vector<double> input_in, vector<double> target_output_in)
{
	if(input_in.size() == this->input.size()) {
		for(int i = 0 ; i< input_in.size(); i++) {
			this->input[i] = input_in[i];
		}
	} else return -1;

	if(target_output_in.size() == this->target_output.size()) {
		for(int i = 0 ; i< target_output_in.size(); i++) {
			this->target_output[i] = target_output_in[i];
		}
	} else return -1;
	cout<<"\t\t\t\\ ++++++++++++++++++++++++ with Input and Target Output updated ++++++++++++++++++++++++ /"<<endl<<endl;
	return 0;
}

void Eason::restoreWeights(void)
{
	this->weights_h1n2[0] = 0.15 , this->weights_h1n2[1] = 0.20 ;
	this->weights_h1n2[2] = 0.25 , this->weights_h1n2[3] = 0.3 ;
	this->weights_end[0] = 0.40, this->weights_end[1] = 0.45 ;
	this->weights_end[2] = 0.50, this->weights_end[3] = 0.55 ;
}

void Eason::showLastWeights(string str)
{
	cout<<"last parameters group "<<str<<"w1= "<<this->weights_h1n2[0]<<" ; w2= "<<this->weights_h1n2[1]<<" ; w3= "<<this->weights_h1n2[2];
	cout<<" ; w4= "<<this->weights_h1n2[3]<<endl<<" \t\t\t w5= "<<this->weights_end[0]<<" ; w6= "<<weights_end[1];
	cout<<" ; w7= "<<weights_end[2]<<" ; w8= "<<weights_end[3]<<endl<<endl;
}

void Eason::forward(Fun_Ret_double_In_double fun)
{
	this->layer_output_auxiliary.push_back( (*fun)(auxiliary_out_hidden(1)) );
	this->layer_output_auxiliary.push_back( (*fun)(auxiliary_out_end(fun, 1)) );
	this->layer_output_auxiliary.push_back( (*fun)(auxiliary_out_hidden(2)) );
	this->layer_output_auxiliary.push_back( (*fun)(auxiliary_out_end(fun, 2)) );
}

void Eason::updateWeights(Fun_Ret_double_In_double fun)
{
	double out_h1 = this->layer_output_auxiliary[0], out_h2 = this->layer_output_auxiliary[2];
	double out_o1 = this->layer_output_auxiliary[1], out_o2 = this->layer_output_auxiliary[3];

	double derivate_h1= derivativeCalc(fun,out_h1), derivate_h2= derivativeCalc(fun,out_h2);
	double derivate_o1= derivativeCalc(fun,out_o1), derivate_o2= derivativeCalc(fun,out_o2);
	double w1= this->weights_h1n2[0], w2= this->weights_h1n2[1], w3= this->weights_h1n2[2], w4= this->weights_h1n2[3];
	double w5= this->weights_end[5-5], w6= this->weights_end[6-5], w7= this->weights_end[7-5], w8= this->weights_end[8-5];
	double r= this->learning_rate, i1= this->input[0], i2= this->input[1], t_o1= this->target_output[0], t_o2= this->target_output[1];

	// cout<<"w1= "<<w1<<" ; w2= "<<w2<<" ; w3= "<<w3<<" ; w4= "<<w4<<" ; w5= "<<w5<<" ; w6= "<<w6<<" ; w7= "<<w7<<" ; w8= "<<w8<<endl;
	// cout<<"r="<<r<<" ; i1= "<<i1<<" ; i2= "<<i2<<" ; target_o1= "<<t_o1<<" ; target_o2= "<<t_o2<<endl<<"++++++++++++++++++++++++++++++"<<endl;
	// cout<<"out_h1= "<<out_h1<<" ; out_h2= "<<out_h2<<" ; out_o1= "<<out_o1<<" ; out_o2= "<<out_o2<<endl;
	// cout<<"out_h1':net_h1= "<<derivate_h1<<" ; out_o1':net_o1= " << derivate_o1;
	// cout<< "out_h2':net_h2= "<<derivate_h2<<" ; out_o2':net_o2= " << derivate_o2 << endl;

	double E_total_derivate_out_h1 = (out_o1-t_o1)*derivate_o1*w5 + (out_o2-t_o2)*derivate_o2*w7;
	double E_total_derivate_out_h2 = (out_o1-t_o1)*derivate_o1*w6 + (out_o2-t_o2)*derivate_o2*w8;

	this->weights_h1n2[1-1] = w1 -  r * E_total_derivate_out_h1 * derivate_h1 * i1;
	this->weights_h1n2[2-1] = w2 -  r * E_total_derivate_out_h1 * derivate_h1 * i2;
	this->weights_h1n2[3-1] = w3 -  r * E_total_derivate_out_h2 * derivate_h2 * i1;
	this->weights_h1n2[4-1] = w4 -  r * E_total_derivate_out_h2 * derivate_h2 * i2;

	this->weights_end[5-5] = w5 -  r * (out_o1 - t_o1) * derivate_o1 * out_h1;
	this->weights_end[6-5] = w6 -  r * (out_o1 - t_o1) * derivate_o1 * out_h2;
	this->weights_end[7-5] = w7 -  r * (out_o2 - t_o2) * derivate_o2 * out_h1;
	this->weights_end[8-5] = w8 -  r * (out_o2 - t_o2) * derivate_o2 * out_h2;
}

double Eason::backward(Fun_Ret_double_In_double fun , unsigned int current_turns, unsigned int last_turns,
					bool logOut=true, bool autoTrain=false, double mse_threshold=0.0f )
{
	this->updateWeights(fun);

	double t_o1_updated = (*fun)(auxiliary_out_end(fun, 1));
	double t_o2_updated = (*fun)(auxiliary_out_end(fun, 2));
	double mse_updated = ( pow(t_o1_updated-this->target_output[0],2) + pow(t_o2_updated-this->target_output[1],2) )/2;

	this->layer_output_auxiliary.clear();

	if ( ! autoTrain ) {
		if(true == logOut || current_turns == last_turns) {
			cout<< fixed << setprecision(PRECISION) <<"mse_updated: " << mse_updated << " ; ( target_o1_updated: " << t_o1_updated ;
			cout<< " ; target_o2_updated: "<<t_o2_updated <<" ) "<< endl;
			cout << "\\ =========================================== turns : "<<current_turns<<" ========================================== /"<<endl<<endl;
		}
		if(current_turns == last_turns) {
			cout<<"\t  Layer-end-output \"Wanted\" output_o1 : "<< this->target_output[0] <<" ; output_o2 : "<<this->target_output[1]<<endl;
		}
	}

	return mse_updated;
}

void Eason::train(unsigned int turns, bool logOut, bool autoTrain, double mse_threshold)
{
	Fun_Ret_double_In_double fun = getActivationFun("sigmoid");
	string str("in turn("+to_string(turns)+") : ");
	if ( ! autoTrain ) {
		for ( int i = 0 ;i < turns ; i++) {
			this->forward(fun);
			this->backward(fun,i+1, turns,logOut, 0, 0.0 );
		}
	} else {
		double mse = 0.0;
		unsigned long long epoch=0;
		do {
			this->forward(fun);
			mse = this->backward(fun, 0, 0, logOut, true, mse_threshold);
			epoch+=1;
		}	while(mse > mse_threshold);
		double t_o1_updated = (*fun)(auxiliary_out_end(fun, 1));
		double t_o2_updated = (*fun)(auxiliary_out_end(fun, 2));
		cout<<fixed<<setprecision(PRECISION)<<"\tmse: "<<mse<<" ; ( target_o1_updated: "<<t_o1_updated<<" ; target_o2_updated: "<<t_o2_updated<<"; epoch: "<<epoch<<" ) "<<endl;
		cout<<"\t  Layer-end-output \"Wanted\" output_o1 : "<< this->target_output[0] <<" ; output_o2 : "<<this->target_output[1]<<endl;
		str="after inferring automatically: ";
	}
	this->showLastWeights(str);
	this->restoreWeights();
}

Eason::Eason()
{
	cout << "===== @@ Constructor() to show author \"Eason\" when releasing for python "<<endl;
}
Eason::~Eason()
{
	cout << "===== @@ Destructor reserved by author Eason " << endl;
}
