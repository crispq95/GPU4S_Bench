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


void transpose(bench_t *A, bench_t *B, int n) {
    int i,j;
    for(i=0; i<n; i++) {
        for(j=0; j<n; j++) {
            B[j*n+i] = A[i*n+j];
        }
    }
}


void execute_kernel(GraficObject * device_object, unsigned int n, unsigned int m, unsigned int w)
{
	// Start compute timer
	const double start_wtime = omp_get_wtime();
	
	// Transpose B to then compute matrix multiply
	bench_t *B_transposed;
    B_transposed = (bench_t*)malloc( sizeof(bench_t) * n * n);
    transpose(device_object->d_B, B_transposed, n);
	unsigned int i, j, k;
	
	#ifdef USM 
	printf("USM Model\n");
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
	printf("Accessor-Buffer Model\n");

	sycl::buffer<bench_t> buffA(device_object->d_A, (n * n));
	sycl::buffer<bench_t> buffB(device_object->d_B, (n * n));
	sycl::buffer<bench_t> buffC(device_object->d_C, (n * n));
	
	int blockSize = 4;

	myQueue.submit([&](sycl::handler& cgh) {
		auto accA = buffA.get_access<sycl::access::mode::read>(cgh);
		auto accB = buffB.get_access<sycl::access::mode::read>(cgh);
		auto accC = buffC.get_access<sycl::access::mode::write>(cgh);

		sycl::accessor<bench_t, 1, sycl::access::mode::read_write, sycl::access::target::local> A(sycl::range<1>(blockSize*blockSize), cgh);
		sycl::accessor<bench_t, 1, sycl::access::mode::read_write, sycl::access::target::local> B(sycl::range<1>(blockSize*blockSize), cgh);

		cgh.parallel_for<class mat_mult>(
          sycl::nd_range<2>{sycl::range<2>(n, n), sycl::range<2>(blockSize, blockSize)},
          [=](sycl::nd_item<2> idx) {
            // Local item
			int col = idx.get_local_id(1), blockX = idx.get_group(1);	//localX
			int row = idx.get_local_id(0), blockY = idx.get_group(0);	//localY

			int gRow = blockSize * idx.get_group().get_id(0) + row;
			int gCol = blockSize * idx.get_group().get_id(1) + col;

            int a_start= n*blockSize*blockY, b_start= blockSize*blockX;
            int a_end = a_start + n - 1;

            bench_t sum = 0.0f;
			for (int a = a_start, b = b_start; a <= a_end;  a += blockSize, b += (blockSize * n)) {
				A[row*blockSize+col] = accA[a+n*row+col];
				B[col*blockSize+row] = accB[b+n*row+col];

				idx.barrier(sycl::access::fence_space::local_space);
				for (int k = 0; k < blockSize; k++) {
					sum += A[row * blockSize + k] * B[col * blockSize + k];
				}
				idx.barrier(sycl::access::fence_space::local_space);
			}
			auto index = idx.get_global_id(0) * idx.get_global_range()[1] + idx.get_global_id(1);
			accC[index] = sum;
		  });
	});
	

	#endif

    free(B_transposed);

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
	sycl::free(device_object->d_C, myQueue);
	#else
	 // todo
	 free(device_object->d_C);
	#endif
}