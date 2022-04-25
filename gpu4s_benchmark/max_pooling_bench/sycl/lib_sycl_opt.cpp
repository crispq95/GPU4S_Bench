#include "../benchmark_library.h"
#include <cstring>
#include <math.h>

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
	std :: cout << "Using device: " << myQueue.get_device().get_info<sycl::info::device::name>() << "\n";
}


bool device_memory_init(GraficObject *device_object, unsigned int size_a_matrix, unsigned int size_b_matrix)
{
	printf("init\n");
	#ifdef USM
	device_object->d_B = sycl::malloc_device<bench_t>(size_b_matrix, myQueue);
	#else 
	device_object->d_B = (bench_t*) malloc ( size_b_matrix * sizeof(bench_t*));
	#endif
   	return true;
}


void copy_memory_to_device(GraficObject *device_object, bench_t* h_A, unsigned int size_a)
{
	printf("HtD\n");
	#ifdef USM
	device_object->d_A = sycl::malloc_device<bench_t>(size_a, myQueue);
	myQueue.memcpy(device_object->d_A, h_A, size_a*sizeof(bench_t));
	#else 
	device_object->d_A = h_A;
	#endif

}

#define BLOCK_SIZE_PLANE (BLOCK_SIZE * BLOCK_SIZE)

void execute_kernel(GraficObject *device_object, unsigned int n, unsigned int m, unsigned int w, unsigned int stride, unsigned int lateral_stride)
{
	// Start compute timer
	const double start_wtime = omp_get_wtime();

	const unsigned int block_size = n/stride;
	const unsigned int stride_squared = stride*stride;
	

	#ifdef USM
	const unsigned int s_test = lateral_stride*lateral_stride;
	myQueue
	   .parallel_for<class max_pooling_kernel>(
			sycl::range{s_test},
			 [=, d_A_local=device_object->d_A, d_B_local=device_object->d_B](sycl::id<1> idx)  {
				int i = idx[0]; 

				bench_t max_value = d_A_local[(((i%lateral_stride) * stride )+ ((i/lateral_stride)*n * stride)) ];
				for(unsigned int x = 0; x < stride; ++x)
					for(unsigned int y = 0; y < stride; ++y)
						max_value = max(max_value, d_A_local[((((i%lateral_stride) * stride )+ ((i/lateral_stride)*n * stride)) + x)  + ( y * n)]);
				d_B_local[idx] = max_value;
				
		}).wait();

	#else 
	
	try {
		// //create buffers 
		auto buffA = sycl::buffer{device_object->d_A, sycl::range{block_size*block_size}};
		auto buffB = sycl::buffer{device_object->d_B, sycl::range{block_size*block_size}};

		auto e = myQueue.submit([&](sycl::handler& cgh){
			//create accessors 
			auto accA = buffA.get_access<sycl::access::mode::read>(cgh);
			auto accB = buffB.get_access<sycl::access::mode::write>(cgh);
			
			cgh.parallel_for<class max_pooling_kernel>(
				sycl::range<1>{block_size*block_size}, [=](sycl::id<1> idx){
				int i = idx[0];
				
				bench_t max_value = accA[(((i%lateral_stride) * stride )+ ((i/lateral_stride)*n * stride)) ];
				for(unsigned int x = 0; x < stride; ++x)
					for(unsigned int y = 0; y < stride; ++y)
						max_value = max(max_value, accA[((((i%lateral_stride) * stride )+ ((i/lateral_stride)*n * stride)) + x)  + ( y * n)]);
				accB[idx] = max_value;	

			});	
		}); 

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
	// sycl::free(device_object->d_B, myQueue);
	// sycl::free(device_object->d_A, myQueue);
	#else 
	free(device_object->d_B);
	#endif
}