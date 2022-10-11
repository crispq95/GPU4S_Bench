#include "../benchmark_library.h"
#include <cstring>
#include <cmath>

#include <omp.h>

class LRN_bench_kernel;

void init(GraficObject *device_object, char* device_name)
{
	init(device_object, 0,0, device_name);
}


void init(GraficObject *device_object, int platform ,int device, char* device_name)
{
	// TBD Feature: device name. -- Bulky generic platform implementation
	std :: cout << "Using device: " << myQueue.get_device().get_info<sycl::info::device::name>() << "\n";
}


bool device_memory_init(GraficObject *device_object, unsigned int size_a_matrix, unsigned int size_b_matrix)
{
	#ifdef USM
	device_object->d_B = sycl::malloc_device<bench_t>(size_b_matrix, myQueue);
	#else
	device_object->d_B = (bench_t*) malloc ( size_b_matrix * sizeof(bench_t*));
	#endif

	return true;
} 


void copy_memory_to_device(GraficObject *device_object, bench_t* h_A, unsigned int size_a)
{
	const double start_wtime = omp_get_wtime();
	
	
	#ifdef USM
	device_object->d_A = sycl::malloc_device<bench_t>(size_a, myQueue);
	myQueue.memcpy(device_object->d_A, h_A, (size_a)*sizeof(bench_t)).wait();
	#else
	device_object->d_A = h_A;
	#endif

	device_object->elapsed_time_HtD = omp_get_wtime() - start_wtime;
	
}


void execute_kernel(GraficObject *device_object, unsigned int n, unsigned int m, unsigned int w)
{
	// Start compute timer
	const double start_wtime = omp_get_wtime();
	
	#ifdef USM
	myQueue
	   .parallel_for<LRN_bench_kernel>(
			sycl::range{n}, 
			[=, d_A_local=device_object->d_A, d_B_local=device_object->d_B](sycl::id<1> idx){
				
				for(int j=0; j<n; j++)
					d_B_local[idx] = d_A_local[idx*n+j]/sycl::powr((K+ALPHA*powf(d_A_local[idx[0]*n+j],2)), BETA);
			}).wait();
	#else 
	try {
		// //create buffers 
		auto buffA = sycl::buffer{device_object->d_A, sycl::range{n*n}};
		auto buffB = sycl::buffer{device_object->d_B, sycl::range{n*n}};

		auto e = myQueue.submit([&](sycl::handler& cgh){
			//create accessors 
			auto accA = buffA.get_access<sycl::access::mode::read>(cgh);
			auto accB = buffB.get_access<sycl::access::mode::write>(cgh);
			
			cgh.parallel_for<LRN_bench_kernel>(
				sycl::range<1>{n}, [=](sycl::id<1> idx){  
				bench_t sum = 0.0;
				
				for(int j=0; j<n; j++)
					accB[idx*n+j] = accA[idx*n+j]/sycl::powr((K+ALPHA*powf(accA[idx*n+j],2)),BETA);

			});	//end parallel_for
		}); //end submit
 
		e.wait();
		}catch (const sycl::exception& e) {
        	std::cout << "Exception caught: " << e.what() << std::endl;
    	}

	#endif

	// End compute timer
	device_object->elapsed_time = omp_get_wtime() - start_wtime;
}


void copy_memory_to_host(GraficObject *device_object, bench_t* h_C, int size)
{	    

	const double start_wtime = omp_get_wtime();
	#ifdef USM
	myQueue.memcpy(h_C, device_object->d_B, (size)*sizeof(bench_t)).wait();
	#else
	memcpy(h_C, &device_object->d_B[0], sizeof(bench_t)*size);
	#endif

	device_object->elapsed_time_DtH = omp_get_wtime() - start_wtime;
	
}


float get_elapsed_time(GraficObject *device_object, bool csv_format, bool csv_format_timestamp, long int current_time)
{
	if (csv_format_timestamp){ 
        printf("%.10f;%.10f;%.10f;%ld;\n", (bench_t) 0, device_object->elapsed_time * 1000.f, (bench_t) 0, current_time);
    }
    else if (csv_format)
	{
        printf("%.10f;%.10f;%.10f;\n", (bench_t) 0, device_object->elapsed_time * 1000.f, (bench_t) 0);
    } 
	else
	{
		printf("Elapsed time Host->Device: %.10f milliseconds\n", device_object->elapsed_time_HtD* 1000.f);
		printf("Elapsed time kernel: %.10f milliseconds\n", device_object->elapsed_time * 1000.f);
		printf("Elapsed time Device->Host: %.10f milliseconds\n", device_object->elapsed_time_DtH * 1000.f);
    }
	return device_object->elapsed_time * 1000.f;
}


void clean(GraficObject *device_object)
{
	#ifdef USM
	sycl::free(device_object->d_A, myQueue);
    sycl::free(device_object->d_B, myQueue);
	#else
	free(device_object->d_B);
	#endif

}
