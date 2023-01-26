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
	const double start_wtime = omp_get_wtime();
	#ifdef USM
	myQueue.memcpy(device_object->d_A, h_A, (size_a)*sizeof(bench_t)).wait();
	myQueue.memcpy(device_object->d_B, h_B, (size_b)*sizeof(bench_t)).wait();
	#else
	device_object->d_A = h_A;
	device_object->d_B = h_B;
	#endif
	device_object->elapsed_time_HtD = omp_get_wtime() - start_wtime;
}


void transpose(bench_t *A, bench_t *B, int n) {
    int i,j;
    for(i=0; i<n; i++) {
        for(j=0; j<n; j++) {
            B[j*n+i] = A[i*n+j];
        }
    }
}

void matrix_multiplication_kernel(const bench_t *A,const bench_t *B,  bench_t *C, const int n, const int m, const int w,
                             sycl::nd_item<3> idx, bench_t *A_tile, bench_t *B_tile)
{

    unsigned int i = idx.get_group(2) * BLOCK_SIZE + idx.get_local_id(2);
    unsigned int j = idx.get_group(1) * BLOCK_SIZE + idx.get_local_id(1);

    bench_t acumulated = 0;
    unsigned int idx = 0;

    // load memory
    for (unsigned int sub = 0; sub < idx.get_group_range(2); ++sub)
    {

        idx = i * n + sub * BLOCK_SIZE + idx.get_local_id(1);

        if(idx >= m*n)
        {
            A_tile[idx.get_local_id(2) * BLOCK_SIZE + idx.get_local_id(1)] = 0;
        }
        else
        {
            A_tile[idx.get_local_id(2) * BLOCK_SIZE +
                   idx.get_local_id(1)] = A[idx];
        }
        idx = (sub * BLOCK_SIZE + idx.get_local_id(2)) * w + j;

        if (idx >= m*w)
        {
            B_tile[idx.get_local_id(2) * BLOCK_SIZE + idx.get_local_id(1)] = 0;
        }
        else
        {
            B_tile[idx.get_local_id(2) * BLOCK_SIZE +
                   idx.get_local_id(1)] = B[idx];
        }
        idx.barrier();
        for (unsigned int k = 0; k < BLOCK_SIZE; ++k)
        {
            acumulated += A_tile[idx.get_local_id(2) * BLOCK_SIZE + k] *
                          B_tile[k * BLOCK_SIZE + idx.get_local_id(1)];
        }
        idx.barrier();
    }
    if (i < n && j < w)
    {
        C[i *n + j] = acumulated;
    }
}


void execute_kernel(GraficObject * device_object, unsigned int n, unsigned int m, unsigned int w)
{
	// Start compute timer
	const double start_wtime = omp_get_wtime();
	
	// Transpose B to then compute matrix multiply
	unsigned int i, j, k;

	sycl::range<3> dimBlock(1, BLOCK_SIZE, BLOCK_SIZE);
    sycl::range<3> dimGrid(1, ceil(float(n) / dimBlock[1]), ceil(float(m) / dimBlock[2]));
	
	#ifdef USM 
	myQueue.submit([&](sycl::handler &cgh) {
        sycl::accessor<bench_t, 1, sycl::access_mode::read_write, sycl::access::target::local> A_tile(sycl::range<1>(BLOCK_SIZE*BLOCK_SIZE), cgh);
        sycl::accessor<bench_t, 1, sycl::access_mode::read_write, sycl::access::target::local> B_tile(sycl::range<1>(BLOCK_SIZE*BLOCK_SIZE), cgh);

        cgh.parallel_for(sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
			[=, d_A_local=device_object->d_A, d_B_local=device_object->d_B, d_C_local=device_object->d_C](sycl::nd_item<3> idx) {
				matrix_multiplication_kernel( d_A_local, d_B_local, d_C_local, n, m, w, idx, A_tile.get_pointer(), B_tile.get_pointer());
			}); 
	}).wait(); 
	#else 

	sycl::buffer<bench_t> buffA(device_object->d_A, (n * n));
	sycl::buffer<bench_t> buffB(device_object->d_B, (n * n));
	sycl::buffer<bench_t> buffC(device_object->d_C, (n * n));
	
	int blockSize = 4;

	myQueue.submit([&](sycl::handler &cgh) {
        sycl::accessor<bench_t, 1, sycl::access_mode::read_write, sycl::access::target::local> A_tile(sycl::range<1>(16), cgh);
        sycl::accessor<bench_t, 1, sycl::access_mode::read_write, sycl::access::target::local> B_tile(sycl::range<1>(16), cgh);

		auto accA = buffA.get_access<sycl::access::mode::read>(cgh);
		auto accB = buffB.get_access<sycl::access::mode::read>(cgh);
		auto accC = buffC.get_access<sycl::access::mode::write>(cgh);

        cgh.parallel_for(sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
			[=](sycl::nd_item<3> idx) {
				matrix_multiplication_kernel( accA.get_pointer(), accB.get_pointer(), accC.get_pointer(), 
					n, m, w, idx, A_tile.get_pointer(), B_tile.get_pointer());
			}); 
	}).wait(); 

	#endif


	// End compute timer
	device_object->elapsed_time = omp_get_wtime() - start_wtime;
}


void copy_memory_to_host(GraficObject *device_object, bench_t* h_C, int size)
{	     
	const double start_wtime = omp_get_wtime();
	#ifdef USM  
	myQueue.memcpy(h_C, device_object->d_C, (size)*sizeof(bench_t)).wait();
	#else
	 // todo
	memcpy(h_C, &device_object->d_C[0], sizeof(bench_t)*size);
	#endif
	device_object->elapsed_time_DtH = omp_get_wtime() - start_wtime;
	printf("h_C[0]= %f\n", h_C[0]); 
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
		printf("Elapsed time Host->Device: %.10f milliseconds\n", device_object->elapsed_time_HtD * 1000.f);
		printf("Elapsed time kernel: %.10f milliseconds\n", device_object->elapsed_time * 1000.f);
		printf("Elapsed time Device->Host: %.10f milliseconds\n", device_object->elapsed_time_DtH * 1000.f);
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
