#include "../benchmark_library.h"
#include <cstring>
#include <cmath>

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

 	device_object->mean_A = sycl::malloc_device<result_bench_t>(1, myQueue);
    device_object->mean_B = sycl::malloc_device<result_bench_t>(1, myQueue);
    
    device_object->acumulate_value_a_b = sycl::malloc_device<result_bench_t>(1, myQueue);
    device_object->acumulate_value_b_b = sycl::malloc_device<result_bench_t>(1, myQueue);
    device_object->acumulate_value_a_a = sycl::malloc_device<result_bench_t>(1, myQueue);
	#endif
	return true;
}


void copy_memory_to_device(GraficObject *device_object, bench_t* h_A, unsigned int size_a, bench_t* h_B, unsigned int size_b)
{
	#ifdef USM
	myQueue.memcpy(device_object->d_A, h_A, (size_a)*sizeof(bench_t));
	myQueue.memcpy(device_object->d_B, h_B, (size_b)*sizeof(bench_t));

    bench_t zero = 0;
    myQueue.memcpy(device_object->mean_A, &zero, (1)*sizeof(result_bench_t)).wait();
    myQueue.memcpy(device_object->mean_B, &zero, (1)*sizeof(result_bench_t)).wait();

    myQueue.memcpy(device_object->acumulate_value_a_b, &zero, (1)*sizeof(result_bench_t)).wait();
    myQueue.memcpy(device_object->acumulate_value_b_b, &zero, (1)*sizeof(result_bench_t)).wait();
    myQueue.memcpy(device_object->acumulate_value_a_a, &zero, (1)*sizeof(result_bench_t)).wait();

	myQueue.wait();
	#else
	device_object->d_A = h_A;
	device_object->d_B = h_B;
	#endif
}



result_bench_t get_mean_matrix(const bench_t* A,const int size){
	
	bench_t sum_val = 0;

	#ifdef USM

	#else 
	for (int i=0; i<size; i++){
		for (int j=0; j<size; j++){
			sum_val += A[i*size+j];
		}
	}
	#endif

	return result_bench_t(sum_val) / result_bench_t(size*size);
}


void execute_kernel(GraficObject *device_object, unsigned int size)
{
	// Start compute timer
	const double start_wtime = omp_get_wtime();

	#ifdef USM
    int n=size;
    bench_t *sum_values_a = sycl::malloc_device<result_bench_t>(1, myQueue);
    bench_t *sum_values_b = sycl::malloc_device<result_bench_t>(1, myQueue);
    
    bench_t *acumulate_value_a_a = sycl::malloc_shared<result_bench_t>(1, myQueue);
    bench_t *acumulate_value_a_b = sycl::malloc_shared<result_bench_t>(1, myQueue);
    bench_t *acumulate_value_b_b = sycl::malloc_shared<result_bench_t>(1, myQueue);
    
    
    bench_t *aux = 0; 
    
    myQueue.memcpy(sum_values_a, &aux, 1*sizeof(result_bench_t)).wait(); 
    myQueue.memcpy(sum_values_b, &aux, 1*sizeof(result_bench_t)).wait(); 
    
    
    *acumulate_value_a_a = 0.0, *acumulate_value_a_b = 0.0, *acumulate_value_b_b = 0.0; 
    
	//correlation 2D kernel 
	myQueue
	   .parallel_for<class reduction_kernel>(
			sycl::range<2>{size,size}, [=,  d_A_local=device_object->d_A, d_B_local=device_object->d_B] (sycl::id<2> idx)  {
			int i = idx[0], j = idx[1];
			
			sycl::atomic_ref<bench_t, sycl::memory_order::relaxed, sycl::memory_scope::device, 
            sycl::access::address_space::global_space> (*sum_values_a) += d_A_local[i*size+j];

			sycl::atomic_ref<bench_t, sycl::memory_order::relaxed, sycl::memory_scope::device, 
            sycl::access::address_space::global_space> (*sum_values_b) += d_B_local[i*size+j];
		}).wait();

	myQueue
		.parallel_for<class correlation_2D>(
		sycl::range<2>{size,size}, 
			[=,  d_A_local=device_object->d_A, d_B_local=device_object->d_B]\
			(sycl::id<2> idx)  {
            int i = idx[0], j = idx[1];
                
            result_bench_t local_sum_values_a = *(sum_values_a)/(size*size);
            result_bench_t local_sum_values_b = *(sum_values_b)/(size*size);
            
            result_bench_t result_mean_a = d_A_local[i*size+j] - local_sum_values_a;
			result_bench_t result_mean_b = d_B_local[i*size+j] - local_sum_values_b;
                
			sycl::atomic_ref<bench_t, sycl::memory_order::relaxed, sycl::memory_scope::device, 
            sycl::access::address_space::global_space> (*acumulate_value_a_b) += result_mean_a * result_mean_b;
                
			sycl::atomic_ref<bench_t, sycl::memory_order::relaxed, sycl::memory_scope::device, 
            sycl::access::address_space::global_space> (*acumulate_value_a_a) += result_mean_a * result_mean_a;
			
            sycl::atomic_ref<bench_t, sycl::memory_order::relaxed, sycl::memory_scope::device, 
            sycl::access::address_space::global_space> (*acumulate_value_b_b) += result_mean_b * result_mean_b;
		}).wait();
	
    *device_object->acumulate_value_a_a = *(acumulate_value_a_a); 
    *device_object->acumulate_value_a_b = *(acumulate_value_a_b);
    *device_object->acumulate_value_b_b = *(acumulate_value_b_b);

    sycl::free(sum_values_a, myQueue);
    sycl::free(sum_values_b, myQueue);
    
    sycl::free(acumulate_value_a_a, myQueue);
    sycl::free(acumulate_value_a_b, myQueue);
    sycl::free(acumulate_value_b_b, myQueue);

	#else 

	{
		// printf("EXEC\n"); 
		result_bench_t mean_a_matrix=0; 
		result_bench_t mean_b_matrix=0;
		
		result_bench_t acumulate_value_a_b = 0;
		result_bench_t acumulate_value_a_a = 0;
		result_bench_t acumulate_value_b_b = 0;
		
		result_bench_t result_mean_a = 0;
		result_bench_t result_mean_b = 0;
	
		sycl::buffer<result_bench_t> buff_mean_a(&mean_a_matrix, 1);
		sycl::buffer<result_bench_t> buff_mean_b(&mean_b_matrix, 1);

		sycl::buffer<result_bench_t> buff_acumulate_a_a(&acumulate_value_a_a, 1);
		sycl::buffer<result_bench_t> buff_acumulate_a_b(&acumulate_value_a_b, 1);
		sycl::buffer<result_bench_t> buff_acumulate_b_b(&acumulate_value_b_b, 1);

		sycl::buffer<bench_t> buffA(device_object->d_A, (size*size));
		sycl::buffer<bench_t> buffB(device_object->d_B, (size*size));

		// // printf("k1\n"); 
		myQueue.submit([&](sycl::handler& cgh) {
			auto atomic_acc_a = buff_mean_a.get_access<sycl::access::mode::read_write>(cgh);
			auto atomic_acc_b = buff_mean_b.get_access<sycl::access::mode::read_write>(cgh);

			sycl::accessor<bench_t> a_bA(buffA, cgh);
			sycl::accessor<bench_t> a_bB(buffB, cgh);

			cgh.parallel_for<class reduction_kernel>(sycl::range<1>{size*size}, 
			[=](sycl::id<1> idx){
				sycl::atomic_ref<result_bench_t, sycl::memory_order::relaxed, sycl::memory_scope::device,
				sycl::access::address_space::global_space> atomic_a (atomic_acc_a[0]);
				sycl::atomic_ref<result_bench_t, sycl::memory_order::relaxed, sycl::memory_scope::device,
				sycl::access::address_space::global_space> atomic_b (atomic_acc_b[0]);

				atomic_a += a_bA[idx[0]];
				atomic_b += a_bB[idx[0]];
			});
		}).wait();

		// // printf("k2\n"); 	
        myQueue.submit([&](sycl::handler& cgh) {
			auto atomic_acc_a = buff_mean_a.get_access<sycl::access::mode::read_write>(cgh);
			auto atomic_acc_b = buff_mean_b.get_access<sycl::access::mode::read_write>(cgh);

			auto acc_val_a_a = buff_acumulate_a_a.get_access<sycl::access::mode::read_write>(cgh);
			auto acc_val_a_b = buff_acumulate_a_b.get_access<sycl::access::mode::read_write>(cgh);
			auto acc_val_b_b = buff_acumulate_b_b.get_access<sycl::access::mode::read_write>(cgh);

			sycl::accessor<bench_t> a_bA(buffA, cgh);
			sycl::accessor<bench_t> a_bB(buffB, cgh);

			cgh.parallel_for<class corr>(sycl::range<1>{size*size}, 
			[=](sycl::id<1> idx){
                
                sycl::atomic_ref<result_bench_t, sycl::memory_order::relaxed, sycl::memory_scope::device,
				sycl::access::address_space::global_space> atomic_a (atomic_acc_a[0]);
				sycl::atomic_ref<result_bench_t, sycl::memory_order::relaxed, sycl::memory_scope::device,
				sycl::access::address_space::global_space> atomic_b (atomic_acc_b[0]);
                
                sycl::atomic_ref<result_bench_t, sycl::memory_order::relaxed, sycl::memory_scope::device,
				sycl::access::address_space::global_space> atomic_acc_val_a_a (acc_val_a_a[0]);
				sycl::atomic_ref<result_bench_t, sycl::memory_order::relaxed, sycl::memory_scope::device,
				sycl::access::address_space::global_space> atomic_acc_val_a_b (acc_val_a_b[0]);
                sycl::atomic_ref<result_bench_t, sycl::memory_order::relaxed, sycl::memory_scope::device,
				sycl::access::address_space::global_space> atomic_acc_val_b_b (acc_val_b_b[0]);
                
				result_bench_t mean_a_matrix = atomic_acc_a[0]/(size*size);
				result_bench_t mean_b_matrix = atomic_acc_b[0]/(size*size);

				result_bench_t result_mean_a = a_bA[idx[0]] - mean_a_matrix;
				result_bench_t result_mean_b = a_bB[idx[0]] - mean_b_matrix;

				atomic_acc_val_a_a += (result_mean_a*result_mean_a);
				atomic_acc_val_a_b += (result_mean_a*result_mean_b); 
				atomic_acc_val_b_b += (result_mean_b*result_mean_b);
			});

		}).wait();

		// // printf("mem 1 \n"); 
		(device_object->acumulate_value_a_b) = &acumulate_value_a_b;
		(device_object->acumulate_value_a_a) = &acumulate_value_a_a;
		(device_object->acumulate_value_b_b) = &acumulate_value_b_b;
		// printf("mem 2 dep \n"); 
    }

	
	#endif

	// End compute timer
	device_object->elapsed_time = omp_get_wtime() - start_wtime;
}


void copy_memory_to_host(GraficObject *device_object, result_bench_t* h_R)
{	     
	printf("MEMCPY\n"); 
	#ifdef USM  
	*h_R = (result_bench_t)((*device_object->acumulate_value_a_b )/ (result_bench_t)(sqrt((*device_object->acumulate_value_a_a) * (*device_object->acumulate_value_b_b))));
	#else
	*h_R = (result_bench_t)((*(device_object->acumulate_value_a_b)) / (result_bench_t)(sqrt((*(device_object->acumulate_value_a_a)) * (*(device_object->acumulate_value_b_b)))));
	#endif
	printf("h_R = %f\n", *h_R); 
    
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
// 	sycl::free(device_object->acumulate_value_a_b, myQueue);
//     sycl::free(device_object->acumulate_value_a_a, myQueue);
// 	sycl::free(device_object->acumulate_value_b_b, myQueue);

// 	sycl::free(device_object->mean_A, myQueue);
// 	sycl::free(device_object->mean_B, myQueue);

	sycl::free(device_object->d_B, myQueue);
	sycl::free(device_object->d_A, myQueue);
	#else 
	// free(device_object->d_A);
	// free(device_object->d_B);
	// free(device_object->mean_A);
	// free(device_object->mean_B);
	// free(device_object->acumulate_value_a_b);
	// free(device_object->acumulate_value_a_a);
	// free(device_object->acumulate_value_b_b);
	
	#endif
	return;
}
