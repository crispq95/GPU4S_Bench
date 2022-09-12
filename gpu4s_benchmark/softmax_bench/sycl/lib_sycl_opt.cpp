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
    
    auto wgroup_size = 32;
	auto part_size = wgroup_size * 2;
	auto n_wgroups = ((n*n)+part_size-1)/part_size; 

    
    myQueue.submit([&](sycl::handler& cgh) {
		sycl::accessor <int32_t, 1, sycl::access::mode::read_write, sycl::access::target::local> local_mem(sycl::range<1>(wgroup_size), cgh);

        
        cgh.parallel_for<class reduction_kernel>(sycl::nd_range<1>{n_wgroups*wgroup_size, wgroup_size}, 
		[=,  d_A_local=device_object->d_A, d_B_local=device_object->d_B] (sycl::nd_item<1> idx){
			size_t local_id = idx.get_local_linear_id();
            size_t global_id = idx.get_global_linear_id();

			local_mem[local_id] = 0;

            if ((2 * global_id) < (n*n)) {
				d_B_local[2*global_id] = sycl::exp(d_A_local[2*global_id]);
				d_B_local[2*global_id+1] = sycl::exp(d_A_local[2*global_id+1]);

               local_mem[local_id] = d_B_local[2 * global_id] + d_B_local[2 * global_id + 1];
            }
            idx.barrier(sycl::access::fence_space::local_space);
            
            
			for (size_t stride = 1; stride < wgroup_size; stride *= 2) {
               auto i = 2 * stride * local_id;
               if (i < wgroup_size) {
                  local_mem[i] = local_mem[i] + local_mem[i + stride];
               }

               idx.barrier(sycl::access::fence_space::local_space);
            }
            
            if (local_id == 0) {
                sycl::atomic_ref<bench_t, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space> (*sum_values) += local_mem[0];
            }
		});
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
	auto wgroup_size = 32;
	auto part_size = wgroup_size * 2;
	auto n_wgroups = ((n*n)+part_size-1)/part_size; 
	
	sycl::buffer<bench_t> buffA(device_object->d_A, (n * n));
	sycl::buffer<bench_t> buffB(device_object->d_B, (n * n));

    myQueue.submit([&](sycl::handler& cgh) {
		sycl::accessor <int32_t, 1, sycl::access::mode::read_write, sycl::access::target::local>
                         local_mem(sycl::range<1>(wgroup_size), cgh);

		auto atomic_buf = counter_buf.get_access<sycl::access::mode::read_write>(cgh);
		auto a_bA = buffA.get_access<sycl::access::mode::read>(cgh);
		auto a_bB = buffB.get_access<sycl::access::mode::read_write>(cgh);

		cgh.parallel_for<class reduction_kernel>(sycl::nd_range<1>{n_wgroups*wgroup_size, wgroup_size}, 
		[=](sycl::nd_item<1> idx){
			size_t local_id = idx.get_local_linear_id();
            size_t global_id = idx.get_global_linear_id();

			sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space> ao (atomic_buf[0]);
			local_mem[local_id] = 0;

            if ((2 * global_id) < (n*n)) {
				a_bB[2*global_id] = sycl::exp(a_bA[2*global_id]);
				a_bB[2*global_id+1] = sycl::exp(a_bA[2*global_id+1]);

               local_mem[local_id] = a_bB[2 * global_id] + a_bB[2 * global_id + 1];
            }
            idx.barrier(sycl::access::fence_space::local_space);
            
			for (size_t stride = 1; stride < wgroup_size; stride *= 2) {
               auto i = 2 * stride * local_id;
               if (i < wgroup_size) {
                  local_mem[i] = local_mem[i] + local_mem[i + stride];
               }

               idx.barrier(sycl::access::fence_space::local_space);
            }
            if (local_id == 0) {
			 	ao += local_mem[0];
            }
		});
    }).wait();

	myQueue.submit([&](sycl::handler& cgh) {
		auto a_bB = buffB.get_access<sycl::access::mode::read_write>(cgh);
		auto atomic_buf = counter_buf.get_access<sycl::access::mode::read>(cgh);

		cgh.parallel_for<class sm_kernel>(sycl::range<1>(n*n), 
		[=](sycl::id<1> idx){
			a_bB[idx] = a_bB[idx]/atomic_buf[0];
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
