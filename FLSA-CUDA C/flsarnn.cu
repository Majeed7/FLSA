#include <iostream>
#include <cmath>
#include <utility>
#include <flsarnn.h>

#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>

#include <boost/numeric/odeint.hpp>

#include <boost/numeric/odeint/external/thrust/thrust.hpp>

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/random/variate_generator.hpp>

typedef double value_type;


using namespace std;
using namespace boost::numeric::odeint;

//change this to float if your device does not support double computation


//change this to host_vector< ... > of you want to run on CPU
typedef thrust::device_vector< value_type > state_type;
typedef thrust::device_vector< size_t > index_vector_type;
// typedef thrust::host_vector< value_type > state_type;
// typedef thrust::host_vector< size_t > index_vector_type;

void FLSARNN2(float *out, float *y, float *z, float lambda1, float lambda2, int size) {
	float temp;
	float *t = new float[size];

	//Compute t
	for (int i = 0; i < size; i++) {
		if (i == 0)
			temp = y[i] + z[i];
		else if (i == size - 1)
			temp = y[i] - z[i];
		else
			temp = y[i] - z[i - 1] + z[i];

		if (temp > lambda1)
			t[i] = lambda1 - temp;
		else if (temp < -lambda1)
			t[i] = -lambda1 - temp;
		else
			t[i] = 0;
	}

	// Compute z_next
	for (int i = 0; i < size - 1; ++i) {
		temp = z[i] + t[i] - t[i + 1];

		if (temp > lambda2)
			out[i] = lambda2 - z[i];
		else if (temp < -lambda2)
			out[i] = -lambda2 - z[i];
		else
			out[i] = temp - z[i];
	}
}

struct FLSARNN
{
	int N;
	value_type lam1, lam2;
	thrust::device_vector< int > idx;
	thrust::device_vector< value_type > y;

	FLSARNN(value_type lambda1, value_type lambda2, int size, state_type yy)
		: N(size), lam1(lambda1), lam2(lambda2)
	{
		// make new vector of 1, 2, 3 ...
		for (int i = 0; i < size; i++)
		{
			idx.push_back(i);
		}
		y = yy;
	}

	struct FLSARNN_functor1
	{
		value_type lam;
		int N;
		FLSARNN_functor1(value_type lambda, int size)
			: N(size), lam(lambda) {}
		template< class T >
		__host__ __device__
			void operator()(T t) const
		{

			value_type z = thrust::get< 0 >(t);
			value_type zminus1 = thrust::get< 1 >(t);
			value_type y = thrust::get< 2 >(t);
			int indx = thrust::get< 3 >(t);

			value_type temp;
			if (indx == 0)
				temp = y + z;
			else if (indx == N - 1)
				temp = y - z;
			else
				temp = y - zminus1 + z;

			if (temp > lam)
				thrust::get< 4 >(t) = lam - temp;
			else if (temp < -lam)
				thrust::get< 4 >(t) = -lam - temp;
			else
				thrust::get< 4 >(t) = 0.0;
		}
	};

	struct FLSARNN_functor2
	{
		value_type lam;
		FLSARNN_functor2(value_type lambda)
			: lam(lambda) {}
		template< class T >
		__host__ __device__
			void operator()(T t) const
		{
			// unpack the parameter we want to vary and the Lorenz variables
			value_type z = thrust::get< 0 >(t);
			value_type tt = thrust::get< 1 >(t);
			value_type tplus1 = thrust::get< 2 >(t);
			value_type temp = z + tt - tplus1;

			if (temp > lam)
				thrust::get< 3 >(t) = lam - z;
			else if (temp < -lam)
				thrust::get< 3 >(t) = -lam - z;
			else
				thrust::get< 3 >(t) = temp - z;
		}
	};

	template< class State, class Deriv >
	void operator()(const State &x, Deriv &dxdt, value_type t) const
	{
		thrust::device_vector< int > xminus1(N);
		thrust::device_vector< value_type > h(N);
		xminus1.insert(xminus1.begin() + 1, x.begin(), x.end() - 1);

		thrust::for_each(
			thrust::make_zip_iterator(
				thrust::make_tuple(
					x.begin(),
					xminus1.begin(),
					y.begin(),
					idx.begin(),
					h.begin()
				)),
			thrust::make_zip_iterator(
				thrust::make_tuple(
					x.end(),
					xminus1.end(),
					y.end(),
					idx.end(),
					h.end()
				)),
			FLSARNN_functor1(lam1, N)
		);

		thrust::device_vector< value_type > hplus1(N);
		hplus1.insert(hplus1.begin(), h.begin() + 1, h.end());

		thrust::for_each(
			thrust::make_zip_iterator(
				thrust::make_tuple(
					x.begin(),
					h.begin(),
					hplus1.begin(),
					dxdt.begin()
				)),
			thrust::make_zip_iterator(
				thrust::make_tuple(
					x.end(),
					h.end(),
					hplus1.end(),
					dxdt.end()
				)),
			FLSARNN_functor2(lam2)
		);
	}
};

int main(int arc, char* argv[])
{
	int driver_version, runtime_version;
	cudaDriverGetVersion(&driver_version);
	cudaRuntimeGetVersion(&runtime_version);
	cout << driver_version << "\t" << runtime_version << endl;


	state_type x(87);
	vector<value_type> vect{ -0.067885,-0.271394,-0.230591,0.000000,-0.237300,-0.242860,-0.158909,-0.151813,-0.155762,-0.218180,-0.070766,-0.217085,-0.221028,-0.142679,-0.115273,-0.235195,-0.172367,-0.229766,-0.268235,0.000000,-0.103669,-0.063601,-0.143264,-0.175859,-0.210658,-0.114841,-0.198376,-0.176402,-0.132816,-0.173373,0.047142,0.062702,0.145291,-0.051233,-0.032809,0.229869,-0.184480,2.257590,2.387689,0.756727,0.311959,0.131189,0.557582,-0.235412,0.099662,-0.146354,-0.270817,-0.219345,0.004307,0.001176,-0.128079,-0.135935,0.144436,-0.238093,0.517836,0.430683,0.585686,0.693343,0.690585,0.465043,0.237685,0.478356,0.422885,-0.265142,0.079919,0.155776,0.915878,0.085746,-0.050649,0.062327,0.234434,0.084003,-0.054938,0.177567,0.220642,0.130591,0.124412,0.005821,0.066331,0.231169,0.116183,0.236077,0.074800,0.137732,-0.005529,-0.008017,0.217077 };
	//value_type* yy = &vect[0];
	thrust::device_vector< value_type > y(87);// { -0.067885, -0.271394, -0.230591, 0.000000, -0.237300, -0.242860, -0.158909, -0.151813, -0.155762, -0.218180, -0.070766, -0.217085, -0.221028, -0.142679, -0.115273, -0.235195, -0.172367, -0.229766, -0.268235, 0.000000, -0.103669, -0.063601, -0.143264, -0.175859, -0.210658, -0.114841, -0.198376, -0.176402, -0.132816, -0.173373, 0.047142, 0.062702, 0.145291, -0.051233, -0.032809, 0.229869, -0.184480, 2.257590, 2.387689, 0.756727, 0.311959, 0.131189, 0.557582, -0.235412, 0.099662, -0.146354, -0.270817, -0.219345, 0.004307, 0.001176, -0.128079, -0.135935, 0.144436, -0.238093, 0.517836, 0.430683, 0.585686, 0.693343, 0.690585, 0.465043, 0.237685, 0.478356, 0.422885, -0.265142, 0.079919, 0.155776, 0.915878, 0.085746, -0.050649, 0.062327, 0.234434, 0.084003, -0.054938, 0.177567, 0.220642, 0.130591, 0.124412, 0.005821, 0.066331, 0.231169, 0.116183, 0.236077, 0.074800, 0.137732, -0.005529, -0.008017, 0.217077 };

	// initialize x
	thrust::fill(x.begin(), x.begin(), 0);
	y.insert(y.begin(), vect.begin(), vect.end());
	//y.insert(y.begin(), *yy, *(yy + 87));

	FLSARNN flsa(0.5, 0.2, 87, y);

	// create stepper
	runge_kutta4< state_type, value_type, state_type, value_type > stepper;

	int step = integrate(flsa, x, 0.0, 10.0, 0.1);
	std::cout << step << std::endl;
	thrust::copy(x.begin(), x.end(), std::ostream_iterator< value_type >(std::cout, "\n"));
	std::cout << std::endl;

	return 0;
}
void CopyData(state_type des, float *src, int N)
{
	for (int i = 0; i < N; i++)
	{
		des[i] = src[i];
	}
}
void flsarnn(float *out, float *y, float *z, float* lambda, int size)
{
	value_type lambda1 = lambda[0], lambda2 = lambda[1];
	state_type x(size), yy(size);
	CopyData(x, z, size);
	CopyData(yy, y, size);
	FLSARNN flsa(lambda1, lambda2, size, yy);
	integrate(flsa, x, 0.0, 10.0, 0.1);
	for (int i = 0; i < size; i++)
	{
		out[i] = x[i];
	}
}
