#include "../benchmark_library.h"
#include <cstring>

#define max(a,b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a > _b ? _a : _b; })


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


void execute_kernel(GraficObject *restrict device_object, unsigned int n, unsigned int m, unsigned int w, unsigned int stride, unsigned int lateral_stride)
{
		// Start compute timer
	
	const unsigned int block_size = n/stride;
	#pragma acc enter data copyin(device_object[0:2])
	#pragma acc enter data copyin(device_object->d_A[:n*n]) create(device_object->d_B[0:block_size*block_size]) 
	const double start_wtime = omp_get_wtime();
	
	const unsigned int s_test = lateral_stride*lateral_stride;

	// printf("s_test : %d\n",s_test);
	#pragma acc parallel firstprivate(lateral_stride,n) 
	{
	// #pragma acc loop  
	#pragma acc loop independent
	for (unsigned int i = 0; i < s_test; ++i){
		bench_t max_value = device_object->d_A[(((i%lateral_stride) * stride )+ ((i/lateral_stride)*n * stride)) ];
		#pragma acc loop collapse(2) reduction(max:max_value) firstprivate(max_value) 
		for(unsigned int x = 0; x < stride; ++x){
			for(unsigned int y = 0; y < stride; ++y){
				max_value = max(max_value, device_object->d_A[((((i%lateral_stride) * stride )+ ((i/lateral_stride)*n * stride)) + x)  + ( y * n)]);
				}
			}
		device_object->d_B[i] = max_value;
	}
	}
	device_object->elapsed_time = omp_get_wtime() - start_wtime;

	#pragma acc exit data copyout(device_object->d_B[0:block_size*block_size])
	// End compute timer

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