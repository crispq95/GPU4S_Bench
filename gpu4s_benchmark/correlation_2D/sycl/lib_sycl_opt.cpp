#include "../benchmark_library.h"
#include <cstring>
#include <cmath>


void mean_matrices(bench_t *d_A_local, bench_t *d_B_local, result_bench_t *mean_A, result_bench_t *mean_B, int size, bench_t *shared_data_A, bench_t *shared_data_B, sycl::nd_item<3> idx)
{
    sycl::atomic_ref<result_bench_t, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space> atomic_ma (mean_A[0]);
    sycl::atomic_ref<result_bench_t, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space> atomic_mb (mean_B[0]);

    unsigned int i = idx.get_group(2) * idx.get_local_range(2) + idx.get_local_id(2);
    unsigned int j = idx.get_group(1) * idx.get_local_range(1) + idx.get_local_id(1);
    unsigned int tid_x = idx.get_local_id(2);
    unsigned int tid_y = idx.get_local_id(1);

    if (i < size && j < size){
        shared_data_A[tid_x * idx.get_local_range(1) + tid_y] = d_A_local[i * size + j];
        shared_data_B[tid_x * idx.get_local_range(1) + tid_y] = d_B_local[i * size + j];

        idx.barrier();

        for (unsigned int s_y = idx.get_local_range(1) / 2; s_y > 0; s_y >>= 1){
            if (tid_y < s_y){
                shared_data_A[tid_x * idx.get_local_range(1) + tid_y] += shared_data_A[tid_x * idx.get_local_range(1) + tid_y +s_y];
                shared_data_B[tid_x * idx.get_local_range(1) + tid_y] += shared_data_B[tid_x * idx.get_local_range(1) + tid_y +s_y];
            }
            idx.barrier();
        }
        for (unsigned int s_x = idx.get_local_range(2) / 2; s_x > 0; s_x >>= 1)
        {
            if(tid_x < s_x)
            {
                shared_data_A[tid_x * idx.get_local_range(1)] += shared_data_A[(tid_x + s_x) * idx.get_local_range(1)];
                shared_data_B[tid_x * idx.get_local_range(1)] += shared_data_B[(tid_x + s_x) * idx.get_local_range(1)];
            }
            idx.barrier();
        }

        if( tid_x == 0 && tid_y == 0){
            atomic_ma += shared_data_A[0]; 
            atomic_mb += shared_data_B[0]; 
        }
    }
}

void correlation_2D(bench_t *d_A_local, bench_t *d_B_local, result_bench_t *mean_A_local, result_bench_t *mean_B_local, result_bench_t* acumulate_value_a_b_local,
    result_bench_t* acumulate_value_b_b_local, result_bench_t* acumulate_value_a_a_local, int size, bench_t *shared_data_A_A, 
    bench_t *shared_data_A_B, bench_t *shared_data_B_B, sycl::nd_item<3> idx)
{
    unsigned int i = idx.get_group(2) * idx.get_local_range(2) + idx.get_local_id(2);
    unsigned int j = idx.get_group(1) * idx.get_local_range(1) + idx.get_local_id(1);

    unsigned int tid_x = idx.get_local_id(2);
    unsigned int tid_y = idx.get_local_id(1);

    result_bench_t mean_a_matrix = (*mean_A_local) / (size*size);
    result_bench_t mean_b_matrix = (*mean_B_local) / (size*size);

    sycl::atomic_ref<result_bench_t, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space> atomic_a_a (acumulate_value_a_a_local[0]);
    sycl::atomic_ref<result_bench_t, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space> atomic_a_b (acumulate_value_a_b_local[0]);
    sycl::atomic_ref<result_bench_t, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space> atomic_b_b (acumulate_value_b_b_local[0]);


    if (i < size && j < size){
        result_bench_t result_mean_a = 0;
        result_bench_t result_mean_b = 0;
        result_mean_a = d_A_local[i*size+j] - mean_a_matrix;
        result_mean_b = d_B_local[i*size+j] - mean_b_matrix;
        shared_data_A_B[tid_x * idx.get_local_range(1) + tid_y] = result_mean_a * result_mean_b;
        shared_data_A_A[tid_x * idx.get_local_range(1) + tid_y] = result_mean_a * result_mean_a;
        shared_data_B_B[tid_x * idx.get_local_range(1) + tid_y] = result_mean_b * result_mean_b;

        // first get the final value  in A (A - mean(a)) and in B (B - mean(b))
        idx.barrier();

        for (unsigned int s_y = idx.get_local_range(1) / 2; s_y > 0; s_y >>= 1)
        {
            if (tid_y < s_y)
            {
                shared_data_A_B[tid_x * idx.get_local_range(1) + tid_y] += shared_data_A_B[tid_x * idx.get_local_range(1) + tid_y + s_y];
                shared_data_A_A[tid_x * idx.get_local_range(1) + tid_y] += shared_data_A_A[tid_x * idx.get_local_range(1) + tid_y + s_y];
                shared_data_B_B[tid_x * idx.get_local_range(1) + tid_y] += shared_data_B_B[tid_x * idx.get_local_range(1) + tid_y + s_y];
            }
            idx.barrier();
        }
        for (unsigned int s_x = idx.get_local_range(2) / 2; s_x > 0; s_x >>= 1){
            if(tid_x < s_x){
                shared_data_A_B[tid_x * idx.get_local_range(1)] += shared_data_A_B[(tid_x + s_x) * idx.get_local_range(1)];
                shared_data_A_A[tid_x * idx.get_local_range(1)] += shared_data_A_A[(tid_x + s_x) * idx.get_local_range(1)];
                shared_data_B_B[tid_x * idx.get_local_range(1)] += shared_data_B_B[(tid_x + s_x) * idx.get_local_range(1)];
            }
            idx.barrier();
        }

        if( tid_x == 0 && tid_y == 0){
            atomic_a_a += shared_data_A_A[0];
            atomic_a_b += shared_data_A_B[0];
            atomic_b_b += shared_data_B_B[0];
        }
    }
}

void init(GraficObject *device_object, char* device_name){
	init(device_object, 0,0, device_name);
} 


void init(GraficObject *device_object, int platform, int device, char* device_name)
{
	// TBD Feature: device name. -- Bulky generic platform implementation
	strcpy(device_name,"Generic device");
}


bool device_memory_init(GraficObject *device_object, unsigned int size_a_matrix, unsigned int size_b_matrix)
{
    device_object->d_A = sycl::malloc_device<bench_t>(size_a_matrix, myQueue);
	device_object->d_B = sycl::malloc_device<bench_t>(size_b_matrix, myQueue);

    device_object->mean_A = sycl::malloc_device<result_bench_t>(1, myQueue);
    device_object->mean_B = sycl::malloc_device<result_bench_t>(1, myQueue);
    
    device_object->acumulate_value_a_b = sycl::malloc_device<result_bench_t>(1, myQueue);
    device_object->acumulate_value_b_b = sycl::malloc_device<result_bench_t>(1, myQueue);
    device_object->acumulate_value_a_a = sycl::malloc_device<result_bench_t>(1, myQueue);
    
	return true;
}


void copy_memory_to_device(GraficObject *device_object, bench_t* h_A, unsigned int size_a, bench_t* h_B, unsigned int size_b)
{
    myQueue.memcpy(device_object->d_A, h_A, (size_a)*sizeof(bench_t)).wait();
	myQueue.memcpy(device_object->d_B, h_B, (size_b)*sizeof(bench_t)).wait();

    bench_t zero = 0;
    myQueue.memcpy(device_object->mean_A, &zero, (1)*sizeof(result_bench_t)).wait();
    myQueue.memcpy(device_object->mean_B, &zero, (1)*sizeof(result_bench_t)).wait();

    myQueue.memcpy(device_object->acumulate_value_a_b, &zero, (1)*sizeof(result_bench_t)).wait();
    myQueue.memcpy(device_object->acumulate_value_b_b, &zero, (1)*sizeof(result_bench_t)).wait();
    myQueue.memcpy(device_object->acumulate_value_a_a, &zero, (1)*sizeof(result_bench_t)).wait();
    
	// device_object->d_A = h_A;
	// device_object->d_B = h_B;
}


void execute_kernel(GraficObject *device_object, unsigned int size){
    struct timespec start, end;
    // clock_gettime(CLOCK_MONOTONIC_RAW, &start);
    const double start_wtime = omp_get_wtime();

    sycl::range<3> dimBlock(1, BLOCK_SIZE, BLOCK_SIZE);
    sycl::range<3> dimGrid(1, ceil(float(size) / dimBlock[1]), ceil(float(size) / dimBlock[2]));
    
    myQueue.submit([&](sycl::handler& cgh) {
        sycl::accessor<bench_t, 1, sycl::access_mode::read_write, sycl::access::target::local> shared_data_A(sycl::range<1>(BLOCK_SIZE * BLOCK_SIZE), cgh);
        sycl::accessor<bench_t, 1, sycl::access_mode::read_write, sycl::access::target::local> shared_data_B(sycl::range<1>(BLOCK_SIZE * BLOCK_SIZE), cgh);

        cgh.parallel_for<class reduction_kernel>(sycl::nd_range<3>(dimGrid * dimBlock, dimBlock), 
            [=,  d_A_local=device_object->d_A, d_B_local=device_object->d_B, mean_A=device_object->mean_A, mean_B=device_object->mean_B](sycl::nd_item<3> idx){
                mean_matrices(d_A_local, d_B_local, mean_A, mean_B, size, shared_data_A.get_pointer(), shared_data_B.get_pointer(), idx);
            });        
        }).wait();

    myQueue.submit([&](sycl::handler& cgh) {
        sycl::accessor<bench_t, 1, sycl::access_mode::read_write,sycl::access::target::local> shared_data_A_B(sycl::range<1>(BLOCK_SIZE * BLOCK_SIZE), cgh);
        sycl::accessor<bench_t, 1, sycl::access_mode::read_write,sycl::access::target::local> shared_data_A_A(sycl::range<1>(BLOCK_SIZE * BLOCK_SIZE), cgh);
        sycl::accessor<bench_t, 1, sycl::access_mode::read_write,sycl::access::target::local> shared_data_B_B(sycl::range<1>(BLOCK_SIZE * BLOCK_SIZE), cgh);

        cgh.parallel_for<class correlation_kernel>(sycl::nd_range<3>(dimGrid * dimBlock, dimBlock), 
            [=,  d_A_local=device_object->d_A, d_B_local=device_object->d_B, mean_A_local=device_object->mean_A, mean_B_local=device_object->mean_B, \
            acumulate_value_a_b_local=device_object->acumulate_value_a_b, acumulate_value_b_b_local=device_object->acumulate_value_b_b, \
            acumulate_value_a_a_local=device_object->acumulate_value_a_a ](sycl::nd_item<3> idx){
                correlation_2D(d_A_local, d_B_local, mean_A_local, mean_B_local, acumulate_value_a_b_local, acumulate_value_b_b_local, 
                    acumulate_value_a_a_local, size, shared_data_A_A.get_pointer(), shared_data_A_B.get_pointer(), shared_data_B_B.get_pointer(), idx);
        }); 
      }).wait(); 

  	// clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    // device_object->elapsed_time = (end.tv_sec - start.tv_sec) * 1000 + (end.tv_nsec - start.tv_nsec) / 1000000;
    device_object->elapsed_time2 = omp_get_wtime() - start_wtime;
}


void copy_memory_to_host(GraficObject *device_object, result_bench_t* h_R)
{	     
    result_bench_t a_a, a_b, b_b; 

    myQueue.memcpy(&a_b, device_object->acumulate_value_a_b, (1)*sizeof(result_bench_t)).wait();
    myQueue.memcpy(&b_b, device_object->acumulate_value_b_b, (1)*sizeof(result_bench_t)).wait();
    myQueue.memcpy(&a_a, device_object->acumulate_value_a_a, (1)*sizeof(result_bench_t)).wait();

    *h_R = (result_bench_t)(a_b/(result_bench_t)(sqrt(a_a*b_b)));
     printf("h_R = %f\n", *h_R); 
}


float get_elapsed_time(GraficObject *device_object, bool csv_format,bool csv_format_timestamp, long int current_time)
{
	if (csv_format_timestamp){
        printf("%.10f;%.10f;%.10f;%ld;\n", (bench_t) 0, device_object->elapsed_time, (bench_t) 0, current_time);
    }
    else if (csv_format)
	{
        printf("%.10f;%.10f;%.10f;\n", (bench_t) 0, device_object->elapsed_time, (bench_t) 0);
    } 
	// else
	{
		printf("Elapsed time Host->Device: %.10f milliseconds\n", (bench_t) 0);
		printf("Elapsed time kernel: %.10f milliseconds\n", device_object->elapsed_time2 * 1000.f);
		printf("Elapsed time Device->Host: %.10f milliseconds\n", (bench_t) 0);
    }
	return device_object->elapsed_time * 1000.f;
}


void clean(GraficObject *device_object)
{
    sycl::free(device_object->mean_A, myQueue); 
    sycl::free(device_object->mean_B, myQueue); 

    sycl::free(device_object->acumulate_value_a_b, myQueue); 
    sycl::free(device_object->acumulate_value_b_b, myQueue); 
    sycl::free(device_object->acumulate_value_a_a, myQueue); 
	return;
}
