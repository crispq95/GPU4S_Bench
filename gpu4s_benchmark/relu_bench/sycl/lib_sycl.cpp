#include "../benchmark_library.h"
#include <cstring>
#include "math.h"


void init(GraficObject *device_object, char* device_name) 
{
	init(device_object, 0,0, device_name);
}


void init(GraficObject *device_object, int platform, int device, char* device_name)
{
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
	#ifdef USM
	device_object->d_A = sycl::malloc_device<bench_t>(size_a, myQueue);
	myQueue.memcpy(device_object->d_A, h_A, (size_a)*sizeof(bench_t)).wait();
	#else
	device_object->d_A = h_A;
	#endif
} 


void execute_kernel(GraficObject *device_object, unsigned int n, unsigned int m, unsigned int w)
{
	
	//startTimeList.at(i) = wall_clock_t::now();
	const double start_wtime = omp_get_wtime();
	
	#ifdef USM
	myQueue
	   .parallel_for<class relu>(
			sycl::range<2>{n,n}, 
			[=, d_A_local=device_object->d_A, d_B_local=device_object->d_B](sycl::id<2> idx){
				int row = idx[0], col = idx[1]; 
				if (d_A_local[row*n+col] > 0){
					d_B_local[row*n+col] = d_A_local[row*n+col];
				}else {
					d_B_local[row*n+col] = 0;
				}
			}); 

	myQueue.wait();
	#else 
	try {
	//create buffers 
	auto buffA = sycl::buffer{device_object->d_A, sycl::range{n*n}};
	auto buffB = sycl::buffer{device_object->d_B, sycl::range{n*n}};

	auto e = myQueue.submit([&](sycl::handler& cgh){
		//create accessors 
		auto accA = buffA.get_access<sycl::access::mode::read>(cgh);
		auto accB = buffB.get_access<sycl::access::mode::write>(cgh);
		
		cgh.parallel_for<class relu>(
			sycl::range<2>{n,n}, [=](sycl::id<2> idx){
			int row = idx[0], col = idx[1]; 

			if (accA[row*n+col] > 0){
				accB[row*n+col] = accA[row*n+col];
			}else {
				accB[row*n+col] = 0;
			}
		});	
	}); 
	e.wait();
	}catch (const sycl::exception& e) {
        	std::cout << "Exception caught: " << e.what() << std::endl;
	}
	#endif

	device_object->elapsed_time = omp_get_wtime() - start_wtime;
}


void copy_memory_to_host(GraficObject *device_object, bench_t* h_C, int size)
{	     
	#ifdef USM
	myQueue.memcpy(h_C, device_object->d_B, (size)*sizeof(bench_t)).wait();
	#else
	memcpy(h_C, &device_object->d_B[0], sizeof(bench_t)*size);
	#endif
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
	#ifdef USM
	sycl::free(device_object->d_A, myQueue);
    sycl::free(device_object->d_B, myQueue);
	#else
	free(device_object->d_B);
	#endif
}