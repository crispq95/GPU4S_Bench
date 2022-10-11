#include "../benchmark_library.h"
#include <cmath>
#include <cstring>

void init(GraficObject *device_object, char* device_name){
	init(device_object, 0,0, device_name);
}


void init(GraficObject *device_object, int platform, int device, char* device_name)
{
	std :: cout << "Using device: " << myQueue.get_device().get_info<sycl::info::device::name>() << "\n";
}


bool device_memory_init(GraficObject *device_object, unsigned int size_a_matrix, unsigned int size_b_matrix)
{
	#ifdef USM
	device_object->d_A = sycl::malloc_device<bench_t>(size_a_matrix, myQueue);
	device_object->d_B = sycl::malloc_device<bench_t>(size_b_matrix, myQueue);
	#else
	device_object->d_B = (bench_t*) malloc ( size_b_matrix * sizeof(bench_t*));
	#endif
   	return true;
}


void copy_memory_to_device(GraficObject *device_object, bench_t* h_A, unsigned int size_a)
{
	#ifdef USM
	myQueue.memcpy(device_object->d_A, h_A, (size_a)*sizeof(bench_t)).wait();
	#else
	device_object->d_A = h_A;
	#endif
	
}


void execute_kernel(GraficObject * device_object, unsigned int n, unsigned int m, unsigned int w)
{
	// Start compute timer
	const double start_wtime = omp_get_wtime();
	
	#ifdef USM
   
    bench_t *sum_values = sycl::malloc_device<bench_t>(1, myQueue);

	myQueue
	   .parallel_for<class mm_reduction_kernel>(
			sycl::range<1>{n}, 
			[=,  d_A_local=device_object->d_A, d_B_local=device_object->d_B]\
			(sycl::id<1> idx)  {
				int i = idx[0];
					
				for (unsigned int j = 0; j < n; ++j){			
					d_B_local[i*n+j] = sycl::exp(d_A_local[i*n+j]);
					sycl::atomic_ref<bench_t, sycl::memory_order::relaxed, sycl::memory_scope::device, \ 
					sycl::access::address_space::global_space> atomic_data (sum_values[0]);

					atomic_data += d_B_local[i*n+j]; 
				}
				
		}).wait();

	myQueue
	   .parallel_for<class mm_kernel>(
			sycl::range<1>{n}, 
			[=,  d_A_local=device_object->d_A, d_B_local=device_object->d_B]\
			(sycl::id<1> idx)  {
				int i = idx[0];
				
				for (unsigned int j = 0; j < n; ++j){			
					d_B_local[i*n+j] = (d_B_local[i*n+j]/(*sum_values));
				}
				
		}).wait();
	
	#else 
	
    bench_t add = 0;

	{
	sycl::buffer<bench_t> counter_buf(&add, 1);
	sycl::buffer<bench_t> buffA(device_object->d_A, (n * n));
	sycl::buffer<bench_t> buffB(device_object->d_B, (n * n));

    myQueue.submit([&](sycl::handler& cgh) {
		auto atomic_buf = counter_buf.get_access<sycl::access::mode::read_write>(cgh);
		auto a_bA = buffA.get_access<sycl::access::mode::read>(cgh);
		auto a_bB = buffB.get_access<sycl::access::mode::read_write>(cgh);

		cgh.parallel_for<class reduction_kernel>(sycl::range<1>{n}, 
		[=](sycl::id<1> idx){
			// sycl::atomic_ref<bench_t, sycl::memory_order::relaxed, sycl::memory_scope::device,
			// sycl::access::address_space::global_space> ao (atomic_buf[0]);

			// a_bB[idx] = sycl::exp(a_bA[idx]);
			// ao += a_bB[idx];

			int i = idx[0];
					
				for (unsigned int j = 0; j < n; ++j){			
					a_bB[i*n+j] = sycl::exp(a_bB[i*n+j]);
					sycl::atomic_ref<bench_t, sycl::memory_order::relaxed, sycl::memory_scope::device, \ 
					sycl::access::address_space::global_space> ao (atomic_buf[0]);

					ao += a_bB[i*n+j]; 
				}
		});
    }).wait();

	myQueue.submit([&](sycl::handler& cgh) {
		// sycl::accessor<bench_t> a_bB(buffB, cgh);
		auto a_bB = buffB.get_access<sycl::access::mode::write>(cgh);
		auto atomic_buf = counter_buf.get_access<sycl::access::mode::read>(cgh);

		cgh.parallel_for<class sm_kernel>(sycl::range<1>{n}, 
		[=](sycl::id<1> idx){
			// a_bB[idx] = a_bB[idx]/atomic_buf[0];
			int i = idx[0];
				
			for (unsigned int j = 0; j < n; ++j){			
				a_bB[i*n+j] = (a_bB[i*n+j]/(atomic_buf[0]));
			}
		});
	}).wait();
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
	sycl::free(device_object->d_A, myQueue);
    sycl::free(device_object->d_B, myQueue);
	#else
	free(device_object->d_B);
	#endif
}
