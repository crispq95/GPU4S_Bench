#include "../benchmark_library.h"
#include <cstring>

//namespace sycl = cl::sycl;

#if __has_include(<SYCL/sycl.hpp>)
#include <SYCL/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

class relu; 
//#include <vptr/virtual_ptr.hpp>

void init(GraficObject *device_object, char* device_name) 
{
	init(device_object, 0,0, device_name);
}


void init(GraficObject *device_object, int platform, int device, char* device_name)
{
	// TBD Feature: device name. -- Bulky generic platform implementation
	strcpy(device_name,"Generic device");
}


bool device_memory_init(GraficObject *device_object, unsigned int size_a_matrix, unsigned int size_b_matrix) 
{

	device_object->d_B = (bench_t*) malloc ( size_b_matrix * sizeof(bench_t*));
   	return true;
}


void copy_memory_to_device(GraficObject *device_object, bench_t* h_A, unsigned int size_a)
{
	device_object->d_A = h_A;
} 


void execute_kernel(GraficObject *device_object, unsigned int n, unsigned int m, unsigned int w)
{
	
	startTimeList.at(i) = wall_clock_t::now();

	sycl::queue myQueue; 
	//cl::sycl::codeplay::PointerMapper pMap;

	//Allocate matrices using SYCLmalloc. d_B and d_A are virutal pointers pointing to device buffers
	//bench_t* d_B = static_cast<bench_t*>(SYCLmalloc(n*n*sizeof(bench_t), pMap));
	//bench_t* d_A = static_cast<bench_t*>(SYCLmalloc(n*n*sizeof(bench_t), pMap));

	bench_t* d_B = sycl::malloc_device<bench_t>(n*n, myQueue);
	bench_t* d_A = sycl::malloc_device<bench_t>(n*n, myQueue);

	//SYCL memcpy HtD
	myQueue.memcpy(d_A, device_object->d_A, n*n*sizeof(bench_t));

	// Perform computation
	// Compute traditional relu approach 
	
	myQueue
	   .parallel_for<relu>(
			sycl::range{n*n}, 
			[=](sycl::id<1> idx){
				if (d_A[idx] > 0){
					d_B[idx] = d_A[idx];
				}else {
					d_B[idx] = 0;
				}
			}); 
	myQueue.wait();
	//memcpy to host (only needed for B!)
	myQueue.memcpy(device_object->d_B, d_B, n*n*sizeof(bench_t));
}


void copy_memory_to_host(GraficObject *device_object, bench_t* h_C, int size)
{	     
	memcpy(h_C, &device_object->d_B[0], sizeof(bench_t)*size);
}


float get_elapsed_time(GraficObject *device_object, bool csv_format, bool csv_format_timestamp, long int current_time)
{
	if (csv_format_timestamp){
        printf("%.10f;%.10f;%.10f;%ld;\n", (bench_t) 0, device_object->elapsed_time * 1000.f, (bench_t) 0,current_time);
    }
	else if (csv_format)
	{
        printf("%.10f;%.10f;%.10f;\n", (bench_t) 0, device_object->elapsed_time * 1000.f, (bench_t) 0);
    } 
	else
	{
		printf("Elapsed time Host->Device: %.10f milliseconds\n", (bench_t) 0);
		printf("Elapsed time kernel: %.10f milliseconds\n", device_object->elapsed_time * 1000.f);
		printf("Elapsed time Device->Host: %.10f milliseconds\n", (bench_t) 0);
    }
	return device_object->elapsed_time * 1000.f;
}


void clean(GraficObject *device_object)
{
	free(device_object->d_B);
}