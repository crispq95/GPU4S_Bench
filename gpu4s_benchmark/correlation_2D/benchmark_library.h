#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>




#ifdef INT
typedef int bench_t;
typedef float result_bench_t;
static const char type_kernel[] = "typedef int bench_t;\ntypedef float result_bench_t;\n";
#elif FLOAT
typedef float bench_t;
typedef float result_bench_t;
static const char type_kernel[] = "typedef float bench_t;\ntypedef float result_bench_t;\n";
#elif DOUBLE
typedef double bench_t;
typedef double result_bench_t;
static const char type_kernel[] = "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\ntypedef double bench_t;\ntypedef double result_bench_t;\n";
#endif


#ifdef CUDA
// CUDA lib
#include <cuda_runtime.h>
#elif OPENCL
// OpenCL lib
//#include <CL/opencl.h>
#include <CL/cl.hpp>
#elif OPENMP
// OpenMP lib
#include <omp.h>
#elif OPENACC
// OpenACC lib
#include <omp.h>
#include <openacc.h>
#elif SYCL
// SYCL lib 
#if __has_include(<SYCL/sycl.hpp>)
#include <SYCL/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif
#include <omp.h> 
#elif HIP
// HIP part
#include <hip/hip_runtime.h>
// #else
// CPU LIB
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
	result_bench_t* d_R;
	result_bench_t* mean_A; // axuliar values for the mean of matrix A
	result_bench_t* mean_B; // axuliar values for the mean of matrix B
	result_bench_t* acumulate_value_a_b; // auxiliar values for the acumulation
	result_bench_t* acumulate_value_a_a; // auxiliar values for the acumulation
	result_bench_t* acumulate_value_b_b; // auxiliar values for the acumulation
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
	cl::Event *evt_copyAB;
	cl::Event *evt_copyAA;
	cl::Event *evt_copyBB;
	cl::Event *evt;
	cl::Event *evt_mean;
	cl::Buffer *d_A;
	cl::Buffer *d_B;
	cl::Buffer *d_R;
	cl::Buffer *mean_A; // axuliar values for the mean of matrix A
	cl::Buffer *mean_B; // axuliar values for the mean of matrix B
	cl::Buffer *acumulate_value_a_b; // auxiliar values for the acumulation
	cl::Buffer *acumulate_value_a_a; // auxiliar values for the acumulation
	cl::Buffer *acumulate_value_b_b; // auxiliar values for the acumulation

	#elif OPENMP
	// OpenMP part
	bench_t* d_A;
	bench_t* d_B;
	result_bench_t d_R;
	result_bench_t mean_A; // axuliar values for the mean of matrix A
	result_bench_t mean_B; // axuliar values for the mean of matrix B
	result_bench_t acumulate_value_a_b; // auxiliar values for the acumulation
	result_bench_t acumulate_value_a_a; // auxiliar values for the acumulation
	result_bench_t acumulate_value_b_b; // auxiliar values for the acumulation

	#elif SYCL
	// SYCL part
	bench_t* d_A;
	bench_t* d_B;
	result_bench_t d_R;
	result_bench_t *mean_A; // axuliar values for the mean of matrix A
	result_bench_t *mean_B; // axuliar values for the mean of matrix B
	result_bench_t *acumulate_value_a_b; // auxiliar values for the acumulation
	result_bench_t *acumulate_value_a_a; // auxiliar values for the acumulation
	result_bench_t *acumulate_value_b_b; // auxiliar values for the acumulation

	#elif HIP
	// Hip part --
	bench_t* d_A;
	bench_t* d_B;
	result_bench_t* d_R;
	result_bench_t* mean_A; // axuliar values for the mean of matrix A
	result_bench_t* mean_B; // axuliar values for the mean of matrix B
	result_bench_t* acumulate_value_a_b; // auxiliar values for the acumulation
	result_bench_t* acumulate_value_a_a; // auxiliar values for the acumulation
	result_bench_t* acumulate_value_b_b; // auxiliar values for the acumulation
	hipEvent_t *start_memory_copy_device;
	hipEvent_t *stop_memory_copy_device;
	hipEvent_t *start_memory_copy_host;
	hipEvent_t *stop_memory_copy_host;
	hipEvent_t *start;
	hipEvent_t *stop;
	#else
	bench_t* d_A;
	bench_t* d_B;
	result_bench_t d_R;
	result_bench_t mean_A; // axuliar values for the mean of matrix A
	result_bench_t mean_B; // axuliar values for the mean of matrix B
	result_bench_t acumulate_value_a_b; // auxiliar values for the acumulation
	result_bench_t acumulate_value_a_a; // auxiliar values for the acumulation
	result_bench_t acumulate_value_b_b; // auxiliar values for the acumulation
	#endif
	float elapsed_time;
	float elapsed_time2;
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
// auto myQueue = sycl::queue{my_device_selector{}};
#ifdef GPU
	auto myQueue = sycl::queue{my_device_selector{}};
#else
	auto myQueue = sycl::queue{sycl::host_selector()};
#endif
#endif

void init(GraficObject *device_object, char* device_name);
void init(GraficObject *device_object, int platform, int device, char* device_name);
bool device_memory_init(GraficObject *device_object, unsigned int size_a_matrix, unsigned int size_b_matrix);
void copy_memory_to_device(GraficObject *device_object, bench_t* h_A, unsigned int size_a, bench_t* h_B, unsigned int size_b);
void execute_kernel(GraficObject *device_object, unsigned int n);
void copy_memory_to_host(GraficObject *device_object, result_bench_t* h_R);
float get_elapsed_time(GraficObject *device_object, bool csv_format, bool csv_format_timestamp, long int timestamp);
void clean(GraficObject *device_object);


#endif