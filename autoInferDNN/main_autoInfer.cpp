#include "Eason.h"
int main(int argc, char ** argv)
{
	Eason element;
//	element.train(80000);	//turns=1 : mse : 0.291027774	// turns=2 : mse : 0.283547133
	element.setInput_TargetOutput({input_i1,input_i2},{output_o1,output_o2});
	if ( argc > 1) {
		element.train(0,false,true,stod(string(argv[1])));
	} else {
		element.train(8000,false);
	}

	cout<<"DNN for a 3 layer neural network"<<fixed<<setprecision(2)<<endl;
	cout<<"Layer-start-input input_i1 : "<< element.input[0] <<" ; input_i2 : "<< element.input[1] <<endl;
	cout<<"Layer-end-output \"Wanted\" output_o1 : "<< element.target_output[0] <<" ; output_o2 : "<<element.target_output[1]<<endl;
	return 0;
}
