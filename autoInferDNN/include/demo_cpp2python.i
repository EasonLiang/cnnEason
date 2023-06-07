%module AutoInfer_cpp2python_ArchLinux

%{
#include "autoInferClass/Eason.h"
%} 

%include "std_vector.i"
%include "std_string.i"

namespace std{
	%template(IntVector) vector<int>;
	%template(UInt32Vector) vector<unsigned int>;
	%template(StringVector) vector<string>;
	%template(FloatVector) vector<float>;
	%template(DoubleVector) vector<double>;
}

%include "autoInferClass/Eason.h"

using namespace std;
