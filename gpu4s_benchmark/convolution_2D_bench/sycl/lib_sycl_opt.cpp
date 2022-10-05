#include "../benchmark_library.h"
#include <cstring>
#include <omp.h>


void covolution_kernel(const bench_t *A, bench_t *B, const bench_t *kernel,const int n, const int m, const int w, const int kernel_size, const int shared_size, const int kernel_rad,
                  uint8_t *local_data, sycl::nd_item<3> idx)
{
    unsigned int size = n;
    unsigned int x = idx.get_group(2) * idx.get_local_range(2) +idx.get_local_id(2);
    unsigned int y = idx.get_group(1) * idx.get_local_range(1) +idx.get_local_id(1);
    int x0, y0;

    auto data = (bench_t *)local_data;
    if (x < size && y < size)
    {
        // each thread load 4 values ,the corners
        //TOP right corner
        x0 = x - kernel_rad;
        y0 = y - kernel_rad;
        if ( x0 < 0 || y0 < 0 )
        {
            data[idx.get_local_id(2) * shared_size + idx.get_local_id(1)] = 0;
        }
        else
        {
            data[idx.get_local_id(2) * shared_size + idx.get_local_id(1)] = A[x0 * size + y0];
        }

        //BOTTOM right corner
        x0 = x + kernel_rad;
        y0 = y - kernel_rad;
        if ( x0 > size-1  || y0 < 0 )
        {
            data[(idx.get_local_id(2) + kernel_rad * 2) * shared_size + idx.get_local_id(1)] = 0;
        }
        else
        {
            data[(idx.get_local_id(2) + kernel_rad * 2) * shared_size + idx.get_local_id(1)] = A[x0 * size + y0];
        }

        //TOP left corner
        x0 = x - kernel_rad;
        y0 = y + kernel_rad;
        if ( x0 < 0  || y0 > size-1 )
        {
            data[idx.get_local_id(2) * shared_size + (idx.get_local_id(1) + kernel_rad * 2)] = 0;
        }
        else
        {
            data[idx.get_local_id(2) * shared_size + (idx.get_local_id(1) + kernel_rad * 2)] = A[x0 * size + y0];
        }

        //BOTTOM left corner
        x0 = x + kernel_rad;
        y0 = y + kernel_rad;
        if ( x0 > size-1  || y0 > size-1 )
        {
            data[(idx.get_local_id(2) + kernel_rad * 2) * shared_size +
                 (idx.get_local_id(1) + kernel_rad * 2)] = 0;
        }
        else
        {
            data[(idx.get_local_id(2) + kernel_rad * 2) * shared_size +
                 (idx.get_local_id(1) + kernel_rad * 2)] =
                A[x0 * size + y0];
        }

        idx.barrier();
        bench_t sum = 0;
        unsigned int xa = kernel_rad + idx.get_local_id(2);
        unsigned int ya = kernel_rad + idx.get_local_id(1);

		#pragma unroll
        for(int i = -kernel_rad; i <= kernel_rad; ++i) // loop over kernel_rad  -1 to 1 in kernel_size 3
            {
                #pragma unroll
                for(int j = -kernel_rad; j <= kernel_rad; ++j)
                {
                    sum += data[(xa + i) * shared_size +  (ya + j)] * kernel[(i+kernel_rad)* kernel_size + (j+kernel_rad)];
                }
            }

        B[x*size+y ] = sum;
    }

}

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
	time_t frame_start=clock();
		
	#ifdef USM 
		#ifdef CPU
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
				sycl::range<3> dimBlock(1, BLOCK_SIZE, BLOCK_SIZE);
				sycl::range<3> dimGrid(1, ceil(float(m) / dimBlock[1]),ceil(float(n) / dimBlock[2]));

				unsigned int kernel_rad =  kernel_size / 2;
				unsigned int size_shared = (BLOCK_SIZE + kernel_rad * 2) * sizeof(bench_t) * (BLOCK_SIZE + kernel_rad * 2) * sizeof(bench_t);
				unsigned int size_shared_position = (BLOCK_SIZE + kernel_rad *2);
			
				myQueue.submit([&](sycl::handler &cgh) {
					sycl::accessor<uint8_t, 1, sycl::access_mode::read_write, sycl::access::target::local> local_data(sycl::range<1>(size_shared), cgh);

					cgh.parallel_for(sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
						[=, d_A_local=device_object->d_A, d_B_local=device_object->d_B, d_k_local=device_object->kernel](sycl::nd_item<3> idx) {
							covolution_kernel( d_A_local, d_B_local, d_k_local, n, m, w, kernel_size, size_shared_position, kernel_rad, local_data.get_pointer(), idx);
					});
				}).wait();
			#endif
	#else 
	try {
		#ifdef CPU
			// //create buffers 
			sycl::buffer<bench_t> buffA(device_object->d_A, (n * n));
			sycl::buffer<bench_t> buffB(device_object->d_B, (n * n));
			sycl::buffer<bench_t> buffKernel(device_object->kernel, (kernel_size*kernel_size));

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
			#else 
				sycl::range<3> dimBlock(1, BLOCK_SIZE, BLOCK_SIZE);
				sycl::range<3> dimGrid(1, ceil(float(m) / dimBlock[1]),ceil(float(n) / dimBlock[2]));

				sycl::buffer<bench_t> buffA(device_object->d_A, (n * n));
				sycl::buffer<bench_t> buffB(device_object->d_B, (n * n));
				sycl::buffer<bench_t> buffKernel(device_object->kernel, (kernel_size*kernel_size));

				unsigned int kernel_rad =  kernel_size / 2;
				unsigned int size_shared = (BLOCK_SIZE + kernel_rad * 2) * sizeof(bench_t) * (BLOCK_SIZE + kernel_rad * 2) * sizeof(bench_t);
				unsigned int size_shared_position = (BLOCK_SIZE + kernel_rad *2);

				myQueue.submit([&](sycl::handler& cgh){
					//create accessors 
					auto accA = buffA.get_access<sycl::access::mode::read>(cgh);
					auto accB = buffB.get_access<sycl::access::mode::write>(cgh);
					auto accKernel = buffKernel.get_access<sycl::access::mode::read>(cgh);
					sycl::accessor<uint8_t, 1, sycl::access_mode::read_write, sycl::access::target::local> local_data(sycl::range<1>(size_shared), cgh);
				
					cgh.parallel_for(sycl::nd_range<3>(dimGrid * dimBlock, dimBlock),
						[=](sycl::nd_item<3> idx) {
							covolution_kernel( accA.get_pointer(), accB.get_pointer(), accKernel.get_pointer(), n, m, w, kernel_size, size_shared_position, kernel_rad, local_data.get_pointer(), idx);
					});
				}).wait();
			#endif 

	}catch (const sycl::exception& e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
    }
	#endif 

	time_t frame_end = clock();
	const float frame_time = (double)(frame_end-frame_start)/CLOCKS_PER_SEC * 1000;
    device_object->elapsed_time = frame_time;
	printf("time : %f ms \n", frame_time); 
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