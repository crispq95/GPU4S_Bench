#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>


#define HIGHPASSFILTERSIZE 7
#define LOWPASSFILTERSIZE 9


#ifdef INT
typedef int bench_t;
static const std::string type_kernel = "typedef int bench_t;\n#define HIGHPASSFILTERSIZE 7\n#define LOWPASSFILTERSIZE 9\n";
static const bench_t lowpass_filter[LOWPASSFILTERSIZE] = {1,1,1,1,1,1,1,1,1};
static const bench_t highpass_filter[HIGHPASSFILTERSIZE] = {1,1,1,1,1,1,1};
#elif FLOAT
typedef float bench_t;
static const std::string type_kernel = "typedef float bench_t;\n#define HIGHPASSFILTERSIZE 7\n#define LOWPASSFILTERSIZE 9\n";
static const bench_t lowpass_filter[LOWPASSFILTERSIZE] = {0.037828455507,-0.023849465020,-0.110624404418,0.377402855613, 0.852698679009,0.377402855613, -0.110624404418,-0.023849465020, 0.037828455507};
static const bench_t highpass_filter[HIGHPASSFILTERSIZE] = {-0.064538882629, 0.040689417609, 0.418092273222,-0.788485616406,0.418092273222,0.040689417609,-0.064538882629};
#elif DOUBLE
typedef double bench_t;
static const std::string type_kernel = "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\ntypedef double bench_t;\n#define HIGHPASSFILTERSIZE 7\n#define LOWPASSFILTERSIZE 9\n";
static const bench_t lowpass_filter[LOWPASSFILTERSIZE] = {0.037828455507,-0.023849465020,-0.110624404418,0.377402855613, 0.852698679009,0.377402855613, -0.110624404418,-0.023849465020, 0.037828455507};
static const bench_t highpass_filter[HIGHPASSFILTERSIZE] = {-0.064538882629, 0.040689417609, 0.418092273222,-0.788485616406,0.418092273222,0.040689417609,-0.064538882629};
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
#include <openacc.h>
#include <omp.h>
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
#else
// CPU lib
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
	bench_t* low_filter;
	bench_t* high_filter;
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
	cl::Event *evt_copyC;
	cl::Event *evt;
	cl::Event *evt_int;
	cl::Buffer *d_A;
	cl::Buffer *d_B;
	cl::Buffer *low_filter;
	cl::Buffer *high_filter;
	#elif OPENMP
	// OpenMP part
	bench_t* d_A;
	bench_t* d_B;
	bench_t* low_filter;
	bench_t* high_filter;
	#elif OPENACC
	// OpenACC part
	bench_t* d_A;
	bench_t* d_B;
	bench_t* low_filter;
	bench_t* high_filter;
	#elif SYCL 
	// SYCL part --
	bench_t* d_A;
	bench_t* d_B;
	bench_t* low_filter;
	bench_t* high_filter;
	#elif HIP
	// Hip part --
	bench_t* d_A;
	bench_t* d_B;
	bench_t* low_filter;
	bench_t* high_filter;
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
	bench_t* low_filter;
	bench_t* high_filter;
	#endif
	float elapsed_time;
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
// auto myQueue = sycl::queue{my_device_selector{}, sycl::property::queue::in_order{}};
#ifdef GPU
	auto myQueue = sycl::queue{my_device_selector{}};
#else
	auto myQueue = sycl::queue{sycl::host_selector()};
#endif
#endif

void init(GraficObject *device_object, char* device_name);
void init(GraficObject *device_object, int platform, int device, char* device_name);
bool device_memory_init(GraficObject *device_object, unsigned int size_a_matrix, unsigned int size_b_matrix);
void copy_memory_to_device(GraficObject *device_object, bench_t* h_A, unsigned int size_a);
void execute_kernel(GraficObject *device_object, unsigned int n);
void copy_memory_to_host(GraficObject *device_object, bench_t* h_C, int size);
float get_elapsed_time(GraficObject *device_object, bool csv_format, bool csv_format_timestamp, long int timestamp);
void clean(GraficObject *device_object);


#endif