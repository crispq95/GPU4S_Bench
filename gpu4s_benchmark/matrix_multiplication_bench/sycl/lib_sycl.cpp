#include "../benchmark_library.h"
#include <cstring>


void init(GraficObject *device_object, char* device_name){
	init(device_object, 0,0, device_name);
}


void init(GraficObject *device_object, int platform, int device, char* device_name)
{
	std :: cout << "Using device: " << myQueue.get_device().get_info<sycl::info::device::name>() << "\n";
}


bool device_memory_init(GraficObject *device_object, unsigned int size_a_matrix, unsigned int size_b_matrix, unsigned int size_c_matrix) 
{
	#ifdef USM
	device_object->d_A = sycl::malloc_device<bench_t>(size_a_matrix, myQueue);
	device_object->d_B = sycl::malloc_device<bench_t>(size_b_matrix, myQueue);
	device_object->d_C = sycl::malloc_device<bench_t>(size_c_matrix, myQueue);
	#else
	device_object->d_C = (bench_t*) malloc ( size_c_matrix * sizeof(bench_t*));
	#endif

   	return true;
}


void copy_memory_to_device(GraficObject *device_object, bench_t* h_A, bench_t* h_B, unsigned int size_a, unsigned int size_b)
{	
	#ifdef USM
	myQueue.memcpy(device_object->d_A, h_A, (size_a)*sizeof(bench_t)).wait();
	myQueue.memcpy(device_object->d_B, h_B, (size_b)*sizeof(bench_t)).wait();
	#else
	device_object->d_A = h_A;
	device_object->d_B = h_B;
	#endif
}


void execute_kernel(GraficObject *device_object, unsigned int n, unsigned int m, unsigned int w)
{
	// Start compute timer
	const double start_wtime = omp_get_wtime();
	
	// Compute traditional matrix multiplication approach 
	#ifdef USM
	myQueue
	   .parallel_for<class mat_mult>(
			sycl::range<2>{n,w}, 
			[=, d_A_local=device_object->d_A, d_B_local=device_object->d_B,	d_C_local=device_object->d_C]\
			(sycl::id<2> idx){
				int row = idx[0], col = idx[1]; 
				bench_t sum = 0.0;

				for (unsigned int k = 0; k < m; k++){  
					sum +=  d_A_local[row*n+k] * d_B_local[k*w+col];
				}
				d_C_local[row*n+col] = sum; 
		}).wait();

	#else 
	try {
		// //create buffers 
		auto buffA = sycl::buffer{device_object->d_A, sycl::range{n*m}};
		auto buffB = sycl::buffer{device_object->d_B, sycl::range{w*m}};
		auto buffC = sycl::buffer{device_object->d_C, sycl::range{n*w}};

		auto e = myQueue.submit([&](sycl::handler& cgh){
			//create accessors 
			auto accA = buffA.get_access<sycl::access::mode::read>(cgh);
			auto accB = buffB.get_access<sycl::access::mode::read>(cgh);
			auto accC = buffC.get_access<sycl::access::mode::write>(cgh);
			
			cgh.parallel_for<class mat_mult>(
				sycl::range<2>{n,w}, [=](sycl::id<2> idx){
				int row = idx[0], col = idx[1]; 
				bench_t sum = 0.0;

				for (unsigned int k = 0; k < m; k++){  
					sum +=  accA[row*n+k]* accB[k*w+col]; 
				}
				accC[row*n+col] = sum; 

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
	#ifdef USM  
	myQueue.memcpy(h_C, device_object->d_C, (size)*sizeof(bench_t)).wait();
	#else
	 // todo
	memcpy(h_C, &device_object->d_C[0], sizeof(bench_t)*size);
	#endif
}


float get_elapsed_time(GraficObject *device_object, bool csv_format, bool csv_format_timestamp, long int current_time)
{
	if (csv_format_timestamp){
        printf("%.10f;%.10f;%.10f;%ld;\n",(bench_t) 0, device_object->elapsed_time * 1000.f, (bench_t) 0, current_time);
    }
    else if (csv_format){
        printf("%.10f;%.10f;%.10f;\n", (bench_t) 0, device_object->elapsed_time * 1000.f, (bench_t) 0);
    } 
	else
	{
		//printf("Elapsed time Host->Device: %.10f milliseconds\n", (bench_t) 0);
		printf("Elapsed time kernel: %.10f milliseconds\n", device_object->elapsed_time * 1000.f);
		//printf("Elapsed time Device->Host: %.10f milliseconds\n", (bench_t) 0);
    }
    return device_object->elapsed_time * 1000.f;
}


void clean(GraficObject *device_object)
{
	#ifdef USM  
	sycl::free(device_object->d_A, myQueue);
    sycl::free(device_object->d_B, myQueue);
	sycl::free(device_object->d_C, myQueue);
	#else
	 // todo
	 free(device_object->d_C);
	#endif
}