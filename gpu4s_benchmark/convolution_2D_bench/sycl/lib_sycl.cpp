#include "../benchmark_library.h"
#include <cstring>
#include <omp.h>

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
	device_object->kernel = sycl::malloc_device<bench_t>(size_c_matrix, myQueue);
	#else 
	device_object->d_B = (bench_t*) malloc (size_b_matrix*sizeof(bench_t*));
	#endif
   	return true;
}


void copy_memory_to_device(GraficObject *device_object, bench_t* h_A, bench_t* kernel, unsigned int size_a, unsigned int size_b)
{
	#ifdef USM
	myQueue.memcpy(device_object->d_A, h_A, (size_a)*sizeof(bench_t));
	myQueue.memcpy(device_object->kernel, kernel, (size_b)*sizeof(bench_t));
	#else
	device_object->d_A = h_A;
	device_object->kernel = kernel;
	#endif
}


void execute_kernel(GraficObject *device_object, unsigned int n, unsigned int m,unsigned int w, unsigned int kernel_size)
{
	// Start compute timer
	const double start_wtime = omp_get_wtime();

	#ifdef USM 
	myQueue
	   .parallel_for<class covolution_kernel>(
			sycl::range<2>{n,n}, 
			[=,  d_A_local=device_object->d_A, d_B_local=device_object->d_B, kernel_local=device_object->kernel]\
			(sycl::id<2> idx)  {
			int x = idx[0], y = idx[1]; 

			int size = n;
			int kernel_rad = kernel_size / 2;

			bench_t sum = 0;

			for(int i = -kernel_rad; i <= kernel_rad; ++i) 
            {
                for(int j = -kernel_rad; j <= kernel_rad; ++j){
                    bench_t value = 0;
                    
                    if (i + x < 0 || j + y < 0)
                        value = 0;
                    else if ( i + x > size - 1 || j + y > size - 1)
                        value = 0;
                    else
                        value = d_A_local[(x + i)*size+(y + j)];
                    sum += value * kernel_local[(i+kernel_rad)* kernel_size + (j+kernel_rad)];
                }
            }
    		d_B_local[x*size+y] = sum;
		}).wait(); 
	#else 
	try {
		// //create buffers 
		sycl::buffer<bench_t> buffA(device_object->d_A, (n * n));
		sycl::buffer<bench_t> buffB(device_object->d_B, (n * n));
		sycl::buffer<bench_t> buffKernel(device_object->kernel, (kernel_size));

		auto e = myQueue.submit([&](sycl::handler& cgh){
			//create accessors 
			auto accA = buffA.get_access<sycl::access::mode::read>(cgh);
			auto accB = buffB.get_access<sycl::access::mode::write>(cgh);
			auto accKernel = buffKernel.get_access<sycl::access::mode::read>(cgh);
			
			cgh.parallel_for<class mat_mult>(
				sycl::range<2>{n,n}, [=](sycl::id<2> idx){
				int x = idx[0], y = idx[1]; 

				int size = n;
				int kernel_rad = kernel_size / 2;

				bench_t sum = 0;

				for(int i = -kernel_rad; i <= kernel_rad; ++i) 
				{
					for(int j = -kernel_rad; j <= kernel_rad; ++j){
						bench_t value = 0;
						
						if (i + x < 0 || j + y < 0)
							value = 0;
						else if ( i + x > size - 1 || j + y > size - 1)
							value = 0;
						else
							value = accA[(x + i)*size+(y + j)];
						sum += value * accKernel[(i+kernel_rad)* kernel_size + (j+kernel_rad)];
					}
				}
				accB[x*size+y] = sum;

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
	myQueue.memcpy(h_C, device_object->d_B, size*sizeof(bench_t)).wait();
	#else
	memcpy(h_C, &device_object->d_B[0], sizeof(bench_t)*size);
	#endif
}


float get_elapsed_time(GraficObject *device_object, bool csv_format,bool csv_format_timestamp, long int current_time)
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
	sycl::free(device_object->kernel, myQueue);
	#else
	free(device_object->d_B);
	#endif
}