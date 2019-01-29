
system('nvcc -c flsarnn.cu -ccbin "C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\bin" -include "C:\Program Files\boost\boost_1_69_0\boost"')
mex FlsaCudaRNN.cpp flsarnn.obj -lcudart -L"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\lib\x64"

