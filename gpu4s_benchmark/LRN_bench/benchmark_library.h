#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>


#ifdef INT
typedef int bench_t;
static const std::string type_kernel = "typedef int bench_t;\n";
#elif FLOAT
typedef float bench_t;
static const std::string type_kernel = "typedef float bench_t;\n";
const float K = 2;
const float ALPHA = 10e-4;
const float BETA = 0.75;
#elif DOUBLE
typedef double bench_t;
static const std::string type_kernel = "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\ntypedef double bench_t;\n";
const double K = 2;
const double ALPHA = 10e-4;
const double BETA = 0.75;
#endif

#ifdef CUDA
// CUDA lib
#include <cuda_runtime.h>
#elif OPENCL
// OpenCL lib
#include <CL/cl.hpp>
#elif OPENMP
// OpenMP lib
#include <omp.h>
#elif OPENACC 
// OpenACC lib 
#include <omp.h>
#include <openacc.h> 
#include <chrono>
#elif SYCL
// SYCL lib 
#include <CL/sycl.hpp>
#include <omp.h> 
#elif HIP
// HIP part
#include <hip/hip_runtime.h>
#else
// CPU part
#endif

#ifdef INT
	typedef int bench_t;
	#define __ptype "%d"
#elif FLOAT
	typedef float bench_t;
	#define __ptype "%f"
#elif DOUBLE 
	typedef double bench_t;
	#define __ptype "%f"
#else 
	// printf type helper, will resolve to %d or %f given the computed type
	#define __ptype "%f"
#endif

#ifndef BENCHMARK_H
#define BENCHMARK_H

struct GraficObject{
	#ifdef CUDA
	// CUDA PART
	bench_t* d_A;
	bench_t* d_B;
	cudaEvent_t *start_memory_copy_device;
	cudaEvent_t *stop_memory_copy_device;
	cudaEvent_t *start_memory_copy_host;
	cudaEvent_t *stop_memory_copy_host;
	cudaEvent_t *start;
	cudaEvent_t *stop;
   	#elif OPENCL
   	// OpenCL PART
	cl::Context *context;
	cl::CommandQueue *queue;
	cl::Device default_device;
	cl::Event *evt_copyA;
	cl::Event *evt_copyB;
	cl::Event *evt;
	cl::Buffer *d_A;
	cl::Buffer *d_B;
	#elif OPENMP
	// OpenMP part
	bench_t* d_A;
	bench_t* d_B;
	#elif OPENACC
	// OpenACC part 
	bench_t* d_A;
	bench_t* d_B;
	// float elapsed_time_HtD;
	// float elapsed_time_DtH;
	#elif SYCL
	bench_t* d_A;
	bench_t* d_B;
	#elif HIP
	// Hip part --
	bench_t* d_A;
	bench_t* d_B;
	hipEvent_t *start_memory_copy_device;
	hipEvent_t *stop_memory_copy_device;
	hipEvent_t *start_memory_copy_host;
	hipEvent_t *stop_memory_copy_host;
	hipEvent_t *start;
	hipEvent_t *stop;
	#else
	// CPU part
	bench_t* d_A;
	bench_t* d_B;
	#endif
	float elapsed_time;
	// provisional 
	double elapsed_time_HtD;
	double elapsed_time_DtH;
};
					
#ifdef SYCL 
class my_device_selector : public sycl::device_selector {
	public:
	int operator()(const sycl::device& dev) const override {
		#ifdef GPU
		if ( dev.has(sycl::aspect::gpu)) {
			return 1;
		}else {
			return -1;
		}
		#else
		if ( dev.has(sycl::aspect::cpu)) {
			return 1;
		}else {
			return -1; 
		}
		#endif
		
		return -1;	
	}
};

auto myQueue = sycl::queue{my_device_selector{}}; 
#endif

void init(GraficObject *device_object, char* device_name);
void init(GraficObject *device_object, int platform, int device, char* device_name);
bool device_memory_init(GraficObject *device_object, unsigned int size_a_matrix, unsigned int size_b_matrix);
void copy_memory_to_device(GraficObject *device_object, bench_t* h_A, unsigned int size_a);
void execute_kernel(GraficObject *device_object, unsigned int n, unsigned int m, unsigned int w);
void copy_memory_to_host(GraficObject *device_object, bench_t* h_C, int size);
float get_elapsed_time(GraficObject *device_object, bool csv_format, bool csv_format_timestamp, long int timestamp);
void clean(GraficObject *device_object);


#endif
