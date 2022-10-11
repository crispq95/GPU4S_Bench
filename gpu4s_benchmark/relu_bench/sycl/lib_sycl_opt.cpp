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
	const double start_wtime = omp_get_wtime();
    int wgroup_size = 64; 
    //int n_wgroups = ceil((n*n)/wgroup_size);
    
	sycl::range<3> block(1, 1, BLOCK_SIZE);
	sycl::range<3> grid_row(1, 1, ceil(float((n*n))/(BLOCK_SIZE)));

	#ifdef USM
	myQueue
	   .parallel_for<class relu>(
			sycl::nd_range<3>(grid_row * block, block), [=, d_A_local=device_object->d_A, d_B_local=device_object->d_B]
			(sycl::nd_item<3> idx){
				int i = idx.get_local_range(2) * idx.get_group(2)+idx.get_local_id(2);
				bench_t threshold = 0;
				if(i < n*n)
				{
					#ifdef INT
					d_B_local[i] = sycl::max(threshold, d_A_local[i]); 
					#elif FLOAT
					d_B_local[i] = sycl::max(threshold, d_A_local[i]); 
					#else
					d_B_local[i] = fmaxf(threshold, d_A_local[i]); 
					#endif
				}
			}).wait();
	#else 
	try {
	//create buffers 
	auto buffA = sycl::buffer{device_object->d_A, sycl::range{n*n}};
	auto buffB = sycl::buffer{device_object->d_B, sycl::range{n*n}};

	myQueue.submit([&](sycl::handler& cgh){
		//create accessors 
		sycl::accessor accB(buffB, cgh, sycl::write_only, sycl::no_init);
		sycl::accessor accA(buffA, cgh, sycl::read_only);
		
		cgh.parallel_for<class relu>(
			sycl::nd_range<3>(grid_row * block, block), [=](sycl::nd_item<3> idx){

			int i = idx.get_local_range(2) * idx.get_group(2)+idx.get_local_id(2);
			bench_t threshold = 0;

			if(i < n*n)
			{
				#ifdef INT
				accB[i] = sycl::max(threshold, accA[i]); 
				#elif FLOAT
				accB[i] = sycl::max(threshold, accA[i]); 
				#else
				accB[i] = fmaxf(threshold, accA[i]); 
  				#endif

			}
		});	
	}).wait(); 

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
	printf("h_C[0] : %f\n", h_C[0]); 
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