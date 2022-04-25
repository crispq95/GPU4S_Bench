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

void execute_kernel(GraficObject *device_object, unsigned int n, unsigned int m, unsigned int w, unsigned int stride, unsigned int lateral_stride)
{
	// Start compute timer
	const double start_wtime = omp_get_wtime();

	const unsigned int block_size = n/stride;
	const unsigned int stride_squared = stride*stride;
	

	#ifdef USM
	// myQueue
	//    .parallel_for<class max_pooling>(
	// 		sycl::range<2>{n,n},
	// 		 [=, d_A_local=device_object->d_A, d_B_local=device_object->d_B](sycl::id<2> idx)  {
	// 			int i = idx[0], j=idx[1];
	// 			if (i < n && j < n){
	// 			bench_t max_value = d_A_local[((i * stride)) * n + ((j*stride))];

	// 			for(unsigned int x = 0; x < stride; ++x)
    //        			for(unsigned int y = 0; y < stride; ++y){
	// 					max_value = max(max_value, d_A_local[((i * stride) + x) * n + ((j*stride) +y)]);
	// 				}
	// 			d_B_local[i * lateral_stride + j ] = max_value;
	// 			}
	// 	}).wait();
	// #else 

	myQueue
	   .parallel_for<class max_pooling_kernel>(
			sycl::range{block_size*block_size},
			 [=, d_A_local=device_object->d_A, d_B_local=device_object->d_B](sycl::id<1> idx)  {
				bench_t sum = 0.0;
				unsigned int blockx, blocky, block_zero, x, y = 0;
				
				blockx = idx%block_size;
				blocky = idx/block_size;
				block_zero = blockx*stride + blocky*stride*n;
				bench_t max_value = d_A_local[block_zero];

				for(unsigned int i = 0; i < stride_squared; ++i){  
					x = i%stride;
					y = i/stride; 
					max_value = max(max_value, d_A_local[(block_zero+x) + y*n]);
				}

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
				bench_t sum = 0.0;
				unsigned int blockx, blocky, block_zero, x, y = 0;

				blockx = i%block_size;
				blocky = i/block_size;
				block_zero = blockx*stride + blocky*stride*n;
				bench_t max_value = accA[block_zero];	
				
				for(unsigned int i = 0; i < stride_squared; ++i)
				{
					x = i%stride;
					y = i/stride; 
					max_value = max(max_value, accA[(block_zero+x) + y*n]);
				}
				accB[i] = max_value;	

			});	
		}); 

		e.wait();
	}catch (const sycl::exception& e) {
		std::cout << "Exception caught: " << e.what() << std::endl;
	}
	//TODO
	// for (unsigned int block = 0; block < block_size*block_size; ++block)
	// {
	// 	blockx = block%block_size;
	// 	blocky = block/block_size;
	// 	block_zero = blockx*stride + blocky*stride*n;
	// 	max_value = device_object->d_A[block_zero];	
		
	// 	for(unsigned int i = 0; i < stride_squared; ++i)
	// 	{
	// 		x = i%stride;
	// 		y = i/stride; 
	// 		max_value = max(max_value, device_object->d_A[(block_zero+x) + y*n]);
	// 	}
	// 	device_object->d_B[block] = max_value;	
	// }
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