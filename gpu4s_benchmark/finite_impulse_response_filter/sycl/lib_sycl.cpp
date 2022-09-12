#include "../benchmark_library.h"
#include <cstring>

void FIR_kernel(const bench_t *A, bench_t *B, const bench_t *kernel,const int output_size, const int size, 
	const int w, const int kernel_size, sycl::id<1> idx)
{
    int i = idx[0];
    bench_t sum = 0;

    if (i < output_size)
{
        for (int j = 0; j< kernel_size; ++j){             
            if (i +(j - kernel_size + 1) >= 0 && i +(j - kernel_size +1)<  size)
                sum += kernel[kernel_size - j - 1] * A[i +(j - kernel_size + 1)];
            else
                sum += 0;
        }
        B[i] = sum;
    }
}

void init(GraficObject *device_object, char* device_name){
	init(device_object, 0,0, device_name);
}


void init(GraficObject *device_object, int platform, int device, char* device_name)
{
	std :: cout << "Using device: " << myQueue.get_device().get_info<sycl::info::device::name>() << "\n";;
}
 

bool device_memory_init(GraficObject *device_object, unsigned int size_a_matrix, unsigned int size_b_matrix, unsigned int size_c_matrix)
{
	#ifdef USM
	device_object->d_A = sycl::malloc_device<bench_t>(size_a_matrix, myQueue);
	device_object->d_B = sycl::malloc_device<bench_t>(size_b_matrix, myQueue);
	device_object->kernel = sycl::malloc_device<bench_t>(size_b_matrix, myQueue);
	#else 
	device_object->d_B = (bench_t*) malloc ( size_b_matrix * sizeof(bench_t*));
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

void execute_kernel(GraficObject * device_object, unsigned int n, unsigned int m,unsigned int w, unsigned int kernel_size)
{
	// Start compute timer
	const double start_wtime = omp_get_wtime();
	const unsigned int kernel_rad = kernel_size / 2;
	const unsigned int output_size = n + kernel_size - 1;

	#ifdef USM
	myQueue
	   .parallel_for<class fir_kernel>(
			sycl::range<1>{output_size}, 
			[=,  d_A_local=device_object->d_A, d_B_local=device_object->d_B, kernel_local=device_object->kernel]\
			(sycl::id<1> idx)  {
				FIR_kernel(d_A_local, d_B_local, kernel_local, n, m, w, kernel_size, idx); 
		}).wait(); 
	#else
	try {
		// //create buffers 
		sycl::buffer<bench_t> buffA(device_object->d_A, (m));
		sycl::buffer<bench_t> buffB(device_object->d_B, (output_size));
		sycl::buffer<bench_t> buffKernel(device_object->kernel, (kernel_size));

			myQueue.submit([&](sycl::handler& cgh){
			//create accessors 
			auto accA = buffA.get_access<sycl::access::mode::read>(cgh);
			auto accB = buffB.get_access<sycl::access::mode::write>(cgh);
			auto accKernel = buffKernel.get_access<sycl::access::mode::read>(cgh);
			
			cgh.parallel_for<class fir_kernel>(
				sycl::range<1>{output_size}, [=](sycl::id<1> idx){
				FIR_kernel(accA.get_pointer(), accB.get_pointer(), accKernel.get_pointer(), n, m, w, kernel_size, idx); 
			});	//end parallel_for
		}).wait(); //end submit
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


float get_elapsed_time(GraficObject *device_object, bool csv_format, bool csv_format_timestamp, long int current_time)
{
	if (csv_format_timestamp){
        printf("%.10f;%.10f;%.10f;%ld;\n",(bench_t) 0, device_object->elapsed_time , (bench_t) 0, current_time);
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
	// sycl::free(device_object->d_B, myQueue); // ??
	// sycl::free(device_object->d_A, myQueue);
	// sycl::free(device_object->kernel, myQueue);
	#else
	free(device_object->d_B);
	#endif 
}