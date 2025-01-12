#include "../benchmark_library.h"
#include <cstring>
#include <cmath>

#define max(a,b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a > _b ? _a : _b; })


void convolution_kernel(const bench_t *A, bench_t *B, const bench_t *kernel,const int n, const int m, const int w, const int kernel_size)
{
	const unsigned int squared_kernel_size = kernel_size * kernel_size;
	const int kernel_rad = kernel_size / 2;
	const unsigned k_size_n = n; 
	
	#ifdef USM
	const unsigned k_size = n*n; 
	myQueue
	   .parallel_for<class mat_mult_USM>(sycl::range<1>{k_size}, [=] (sycl::id<1> idx){
				int block = idx[0]; 
				int x, y, kx, ky = 0;
				bench_t sum = 0;

				x = block/n;
				y = block%n;
				for(unsigned int k = 0; k < squared_kernel_size; ++k)
				{
					bench_t value = 0;
					kx = (k/kernel_size) - kernel_rad; 
					ky = (k%kernel_size) - kernel_rad;
					if(!(kx + x < 0 || ky + y < 0) && !( kx + x > n - 1 || ky + y > n - 1))
					{
						value = A[(x + kx)*n+(y + ky)];
					}
					sum += value * kernel[(kx+kernel_rad)* kernel_size + (ky+kernel_rad)];
				}
				B[x*n+y] = sum;
				
		}).wait();
	#else
	{
	sycl::buffer<bench_t> buffA(A, (n*n));
	sycl::buffer<bench_t> buffB(B, (n*n));
	sycl::buffer<bench_t> buffKernel(kernel, (kernel_size*kernel_size));

	auto e = myQueue.submit([&](sycl::handler& cgh){
		//create accessors 
		auto accA = buffA.get_access<sycl::access::mode::read>(cgh);
		auto accB = buffB.get_access<sycl::access::mode::write>(cgh);
		auto accKernel = buffKernel.get_access<sycl::access::mode::read>(cgh);

		const unsigned k_size = n*n; 
		
		cgh.parallel_for<class convolution_kernel_AB>(
			sycl::range<1>{k_size}, [=](sycl::id<1> idx){

			int block = idx[0]; 
			int x, y, kx, ky = 0;
			bench_t sum = 0.0;
			
			x = block/n;
			y = block%n;
			
			for(unsigned int k = 0; k < squared_kernel_size; ++k){
				bench_t value = 0;
				kx = (k/kernel_size) - kernel_rad; 
				ky = (k%kernel_size) - kernel_rad;
				if(!(kx + x < 0 || ky + y < 0) && !( kx + x > n - 1 || ky + y > n - 1))
					value = accA[(x + kx)*n+(y + ky)];
				sum += value * accKernel[(kx+kernel_rad)* kernel_size + (ky+kernel_rad)];
			}
			accB[x*n+y] = sum;
    
			});
	});
	e.wait(); 
    }
	
	#endif 
}


void relu_kernel(const bench_t *A, bench_t *B, const int size)
{
	#ifdef USM
	const unsigned int s = size; 
	myQueue
	   .parallel_for<class relu_kernel_USM>(
			sycl::range<2>{s,s}, 
			[=]	(sycl::id<2> idx){
				int i = idx[0], j = idx[1]; 

				if (A[i*size+j] > 0)
					B[i*size+j] = A[i*size+j];
				else 
					B[i*size+j] = 0;
		}).wait();

    #else
	{
	sycl::buffer<bench_t> buffA(A, (size*size));
	sycl::buffer<bench_t> buffB(B, (size*size));

	auto e = myQueue.submit([&](sycl::handler& cgh){
		//create accessors 
		auto accA = buffA.get_access<sycl::access::mode::read>(cgh);
		auto accB = buffB.get_access<sycl::access::mode::write>(cgh);
		
		const unsigned int s = size; 

		cgh.parallel_for<class relu_kernel_AB>(
			sycl::range<2>{s, s}, [=](sycl::id<2> idx){
				int i = idx[0], j = idx[0]; 

				for (unsigned int j = 0; j < size; ++j)
					if (accA[i*size+j] > 0)
						accB[i*size+j] = accA[i*size+j];
					else 
						accB[i*size+j] = 0;
			});
	});
	e.wait(); 
	}
	#endif 
}


void relu_linear_kernel(const bench_t *A, bench_t *B, const int size)
{
	#ifdef USM
	const unsigned int s = size; 
	myQueue
	   .parallel_for<class relu_linear_kerne_USM>(
			sycl::range<1>{s}, 
			[=]	(sycl::id<1> idx){
			int i = idx[0];

			if (A[i] > 0)
				B[i] = A[i];
			else 
				B[i] = 0;
	}).wait();
	
	#else
	{
	sycl::buffer<bench_t> buffA(A, (size));
	sycl::buffer<bench_t> buffB(B, (size));

	auto e = myQueue.submit([&](sycl::handler& cgh){
		//create accessors 
		auto accA = buffA.get_access<sycl::access::mode::read>(cgh);
		auto accB = buffB.get_access<sycl::access::mode::write>(cgh);
		
		const unsigned int s = size; 

		cgh.parallel_for<class relu_linear_kernel_AB>(
			sycl::range<1>{s}, [=](sycl::id<1> idx){
				int i = idx[0]; 

				if (accA[i] > 0)
					accB[i] = accA[i];
				else 
					accB[i] = 0;
			});
	});
	e.wait(); 
	}
	#endif
}


void max_pooling_kernel(const bench_t *A, bench_t *B, const int size, const unsigned int stride,  const unsigned int lateral_stride)
{	
	
	const unsigned int block_size = size/stride;
	const unsigned int stride_squared = stride*stride;

	#ifdef USM
	const unsigned int s = block_size*block_size; 

	myQueue
	   .parallel_for<class max_pooling_kernel_USM>(
			sycl::range<1>{s}, 
			[=]	(sycl::id<1> idx){
			int block = idx[0];
			unsigned int blockx, blocky, block_zero, x, y = 0;
			
			blockx = block%block_size;
			blocky = block/block_size;
			block_zero = blockx*stride + blocky*stride*size;
			bench_t max_value = A[block_zero];		
			for(unsigned int i = 0; i < stride_squared; ++i)
			{
				x = i%stride;
				y = i/stride; 
				max_value = max(max_value, A[(block_zero+x) + y*size]);
			}
			B[block] = max_value;	
	}).wait();


	#else
	{
	sycl::buffer<bench_t> buffA(A, (size*size));
	sycl::buffer<bench_t> buffB(B, (block_size*block_size));

	myQueue.submit([&](sycl::handler& cgh){
		//create accessors 
		auto accA = buffA.get_access<sycl::access::mode::read>(cgh);
		auto accB = buffB.get_access<sycl::access::mode::write>(cgh);
		
		const unsigned int s = block_size*block_size;  // REVISAR !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

		cgh.parallel_for<class max_pooling_kernel_AB>(
			sycl::range<1>{s}, [=](sycl::id<1> idx){
				int block = idx[0]; 
				bench_t max_value = 0;
				unsigned int blockx, blocky, block_zero, x, y = 0;

				blockx = block%block_size;
				blocky = block/block_size;
				block_zero = blockx*stride + blocky*stride*size;
				max_value = accA[block_zero];		

				for(unsigned int i = 0; i < stride_squared; ++i)
				{
					x = i%stride;
					y = i/stride; 
					max_value = max(max_value, accA[(block_zero+x) + y*size]);
				}
				accB[block] = max_value;
			});
	}).wait(); 
	
	}
	#endif
}

void lrn_kernel(const bench_t *A, bench_t *B, const int size)
{
	#ifdef USM
	const unsigned int s = size;
	myQueue
	   .parallel_for<class lrn_kernel_USM>(
			sycl::range<2>{s,s}, 
			[=]	(sycl::id<2> idx){
			int i = idx[0], j = idx[1];

			B[i*size+j] = A[i*size+j]/sycl::powr((K+ALPHA*powf(A[i*size+j],2)),BETA);
	}).wait();

	#else
	{
	sycl::buffer<bench_t> buffA(A, (size*size));
	sycl::buffer<bench_t> buffB(B, (size*size));

	auto e = myQueue.submit([&](sycl::handler& cgh){
		//create accessors 
		auto accA = buffA.get_access<sycl::access::mode::read>(cgh);
		auto accB = buffB.get_access<sycl::access::mode::write>(cgh);
		
		const unsigned int s = size; 

		cgh.parallel_for<class lrn_kernel_AB>(
			sycl::range<2>{s, s}, [=](sycl::id<2> idx){
				int i = idx[0], j = idx[1]; 

				accB[i*size+j] = accA[i*size+j]/sycl::powr((K+ALPHA*powf(accA[i*size+j],2)),BETA);
			});
	});
	e.wait(); 
	}
	#endif
}

void matrix_multiplication_kernel(const bench_t *A,const bench_t *B,  bench_t *C, unsigned int n, unsigned int m, unsigned int w)
{

	#ifdef USM
	const unsigned int c_n = n, c_m = m; 

	myQueue
	   .parallel_for<class matrix_multiplication_kernel_USM>(
			sycl::range<2>{c_n,c_m}, 
			[=]	(sycl::id<2> idx){
			int i = idx[0], j = idx[1];

			bench_t acumulated = 0;
			for (unsigned int k = 0; k < w; k++)
			{   
				acumulated += A[i*w+k] * B[k*m+j];
			}
			C[i*m+j] = acumulated;
	}).wait();

	#else
	{

	auto buffA = sycl::buffer{A, sycl::range{n*w}};
	auto buffB = sycl::buffer{B, sycl::range{w*m}};
	auto buffC = sycl::buffer{C, sycl::range{n*m}};

	auto e = myQueue.submit([&](sycl::handler& cgh){
		//create accessors 
		auto accA = buffA.get_access<sycl::access::mode::read>(cgh);
		auto accB = buffB.get_access<sycl::access::mode::read>(cgh);
		auto accC = buffC.get_access<sycl::access::mode::write>(cgh);

		const unsigned int c_n = n, c_m = m; 

		cgh.parallel_for<class matrix_multiplication_kernel_AB>(
			sycl::range<2>{c_n,c_m}, [=](sycl::id<2> idx){
			int row = idx[0], col = idx[1]; 
			bench_t sum = 0.0;

			if (row < c_n && col < c_m)
			{
			for (unsigned int k = 0; k < w; k++){  
				sum +=  accA[row*w+k]* accB[k*m+col]; 
			}
		    accC[row*m+col] = sum; 
			}
		});	//end parallel_for
	}); //end submit
	e.wait();

	
	}
	#endif
}


void softmax_kernel(const bench_t *A, bench_t *B, const int size)
{	
    
	#ifdef USM
    unsigned long s = size;
	bench_t *sum_values = sycl::malloc_device<bench_t>(1, myQueue);

    bench_t a = 0; 
    
    myQueue.memcpy(sum_values, &a, 1*sizeof(bench_t)).wait(); 
    
    myQueue
	   .parallel_for<class reduction_kernel>(
			sycl::range<1>{s}, 
			[=]	(sycl::id<1> idx){
                
            B[idx] = sycl::exp(A[idx]);
            sycl::atomic_ref<bench_t, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space> (*sum_values) += B[idx];
        }).wait();
    
     myQueue
	   .parallel_for<class sm_kernel>(
			sycl::range<1>{s}, 
			[=]	(sycl::id<1> idx){
            
            B[idx] = (B[idx]/ *sum_values);
        }).wait(); 
    
    sycl::free(sum_values, myQueue);

	#else

    unsigned int n = size; 
    bench_t sum_values = 0;
	{
	sycl::buffer<bench_t> counter_buf(&sum_values, 1);
	sycl::buffer<bench_t> buffA(A, (n));
	sycl::buffer<bench_t> buffB(B, (n));

    myQueue.submit([&](sycl::handler& cgh) {
		auto atomic_buf = counter_buf.get_access<sycl::access::mode::read_write>(cgh);
		sycl::accessor<bench_t> a_bA(buffA, cgh);
		sycl::accessor<bench_t> a_bB(buffB, cgh);

		cgh.parallel_for<class reduction_kernel>(sycl::range<1>{n}, 
		[=](sycl::id<1> idx){
			sycl::atomic_ref<bench_t, sycl::memory_order::relaxed, sycl::memory_scope::device,
			sycl::access::address_space::global_space> ao (atomic_buf[0]);

			a_bB[idx] = sycl::exp(a_bA[idx]);
			ao += a_bB[idx];
		});
    }).wait();

	myQueue.submit([&](sycl::handler& cgh) {
		sycl::accessor<bench_t> a_bB(buffB, cgh);
		auto atomic_buf = counter_buf.get_access<sycl::access::mode::read_write>(cgh);

		cgh.parallel_for<class sm_kernel>(sycl::range<1>{n}, 
		[=](sycl::id<1> idx){
			a_bB[idx] = a_bB[idx]/atomic_buf[0];

		});
	}).wait();
    }
    
	#endif
}


void init(GraficObject *device_object, char* device_name)
{
	init(device_object, 0,0, device_name);
}


void init(GraficObject *device_object, int platform ,int device, char* device_name)
{
	std :: cout << "Using device: " << myQueue.get_device().get_info<sycl::info::device::name>() << "\n";
}


bool device_memory_init(GraficObject *device_object, unsigned int input_data, unsigned int output_data, unsigned int kernel_1, unsigned int kernel_2, unsigned int stride_1, unsigned int stride_2, unsigned int neurons_dense_1, unsigned int neurons_dense_2, unsigned int number_of_images)
{
	const unsigned int size_pooling_1 = input_data / stride_1;
    const unsigned int size_pooling_2 = size_pooling_1 / stride_2;
    const unsigned int weights_layer_1 = size_pooling_2 * size_pooling_2 * neurons_dense_1;
    const unsigned int weights_layer_2 = neurons_dense_1 * neurons_dense_2; 

	#ifdef USM
	device_object->conv_1_output = sycl::malloc_device<bench_t>(input_data*input_data, myQueue);
	device_object->pooling_1_output = sycl::malloc_device<bench_t>(size_pooling_1*size_pooling_1, myQueue);
	device_object->conv_2_output = sycl::malloc_device<bench_t>(size_pooling_1*size_pooling_1, myQueue);
	device_object->pooling_2_output = sycl::malloc_device<bench_t>(size_pooling_2*size_pooling_2, myQueue);
	device_object->dense_layer_1_output = sycl::malloc_device<bench_t>(neurons_dense_1, myQueue);
	device_object->dense_layer_2_output = sycl::malloc_device<bench_t>(neurons_dense_2, myQueue);
	device_object->output_data = sycl::malloc_device<bench_t>(number_of_images*neurons_dense_2, myQueue);
	#else 
	// Convolution 1
   	device_object->conv_1_output = (bench_t*) malloc ( input_data * input_data * sizeof(bench_t*));
	// Pooling 1
	device_object->pooling_1_output = (bench_t*) malloc ( size_pooling_1 * size_pooling_1 * sizeof(bench_t));
	// Convolution 2
   	device_object->conv_2_output = (bench_t*) malloc ( size_pooling_1 * size_pooling_1 * sizeof(bench_t*));
	// Pooling 2
	device_object->pooling_2_output = (bench_t*) malloc ( size_pooling_2 * size_pooling_2 * sizeof(bench_t));
	// Dense 1
   	device_object->dense_layer_1_output = (bench_t*) malloc ( neurons_dense_1 * sizeof(bench_t));
   	// Dense 2
   	device_object->dense_layer_2_output = (bench_t*) malloc ( neurons_dense_2 * sizeof(bench_t));
   	// Output data
   	device_object->output_data = (bench_t*) malloc ( number_of_images * neurons_dense_2 * sizeof(bench_t));
	#endif 
	return true;
}


void copy_memory_to_device(GraficObject *device_object, bench_t* input_data, bench_t* kernel_1_data, bench_t* kernel_2_data, bench_t* weights_1 ,bench_t* weights_2,unsigned int input , unsigned int kernel_size_1, unsigned int kernel_size_2, unsigned int weights_1_size, unsigned int weights_2_size, unsigned int number_of_images)
{
	#ifdef USM
	device_object->input_data = sycl::malloc_device<bench_t>(input*input, myQueue);
	device_object->kernel_1 = sycl::malloc_device<bench_t>(kernel_size_1*kernel_size_1, myQueue);
	device_object->kernel_2 = sycl::malloc_device<bench_t>(kernel_size_2*kernel_size_2, myQueue);
	device_object->dense_layer_1_weights = sycl::malloc_device<bench_t>(weights_1_size, myQueue);
	device_object->dense_layer_2_weights = sycl::malloc_device<bench_t>(weights_2_size, myQueue);

	myQueue.memcpy(device_object->input_data, input_data, input*input*sizeof(bench_t)); 
	myQueue.memcpy(device_object->kernel_1, kernel_1_data, kernel_size_1*kernel_size_1*sizeof(bench_t));
	myQueue.memcpy(device_object->kernel_2, kernel_2_data, kernel_size_2*kernel_size_2*sizeof(bench_t));
	myQueue.memcpy(device_object->dense_layer_1_weights, weights_1, weights_1_size*sizeof(bench_t));
	myQueue.memcpy(device_object->dense_layer_2_weights, weights_2, weights_2_size*sizeof(bench_t));
	#else 
	device_object->input_data = input_data;
	device_object->kernel_1 = kernel_1_data;
	device_object->kernel_2 = kernel_2_data;
	device_object->dense_layer_1_weights = weights_1;
	device_object->dense_layer_2_weights = weights_2;
	#endif
}


void execute_kernel(GraficObject *device_object, unsigned int input_data, unsigned int output_data, unsigned int kernel_1, unsigned int kernel_2, unsigned int stride_1, unsigned int stride_2, unsigned int neurons_dense_1, unsigned int neurons_dense_2, unsigned int number_of_images)
{
	// Start compute timer
	const double start_wtime = omp_get_wtime();

	bench_t* aux_output_data = device_object->output_data;
    bench_t* aux_input_data = device_object->input_data;

	for(unsigned int position = 0; position < number_of_images; ++position)
    {
		aux_input_data = device_object->input_data + position * input_data * input_data;
        aux_output_data = device_object->output_data + position * output_data;
	
		// 1-1 Step convolution
		convolution_kernel(aux_input_data, device_object->conv_1_output, device_object->kernel_1, input_data, input_data, input_data, kernel_1);

		// 1-2 Step activation
		relu_kernel(device_object->conv_1_output, device_object->conv_1_output, input_data);
		
		// 1-3 Step pooling
		const unsigned int size_lateral_1 = input_data / stride_1;
		max_pooling_kernel(device_object->conv_1_output, device_object->pooling_1_output, input_data, stride_1, size_lateral_1);

		// 1-4 Normalization
		lrn_kernel(device_object->pooling_1_output, device_object->pooling_1_output, size_lateral_1);
		
		// 2-1 Step convolution
		convolution_kernel(device_object->pooling_1_output, device_object->conv_2_output, device_object->kernel_2, size_lateral_1, size_lateral_1, size_lateral_1, kernel_2);

		// 2-2 Step activation
		relu_kernel(device_object->conv_2_output, device_object->conv_2_output, size_lateral_1);

		// 2-3 Normalization
		lrn_kernel(device_object->conv_2_output, device_object->conv_2_output, size_lateral_1);

		// 2-4 Step pooling
		const unsigned int size_lateral_2 = size_lateral_1 / stride_2;
		max_pooling_kernel(device_object->conv_2_output, device_object->pooling_2_output, size_lateral_1, stride_2, size_lateral_2);

		// Dense layer 1
		matrix_multiplication_kernel(device_object->dense_layer_1_weights, device_object->pooling_2_output,device_object->dense_layer_1_output,neurons_dense_1, 1, size_lateral_2*size_lateral_2);

		// Activation layer dense 1
		relu_linear_kernel(device_object->dense_layer_1_output, device_object->dense_layer_1_output, neurons_dense_1);
		
		// Dense layer 2
		matrix_multiplication_kernel(device_object->dense_layer_2_weights, device_object->dense_layer_1_output, device_object->dense_layer_2_output, neurons_dense_2, 1, neurons_dense_1);

		// Activation layer dense 2
		relu_linear_kernel(device_object->dense_layer_2_output, device_object->dense_layer_2_output, neurons_dense_2);

		// Softmax - Output
		softmax_kernel(device_object->dense_layer_2_output, aux_output_data, neurons_dense_2);
	}

	// End compute timer
	device_object->elapsed_time = omp_get_wtime() - start_wtime;
}


void copy_memory_to_host(GraficObject *device_object, bench_t* h_C, int size, unsigned int number_of_images)
{	     
	#ifdef USM
	myQueue.memcpy(h_C, device_object->output_data, (size*number_of_images)*sizeof(bench_t)).wait();
	#else 
	memcpy(h_C, &device_object->output_data[0], sizeof(bench_t)*size*number_of_images);
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
	sycl::free(device_object->conv_1_output, myQueue);
	sycl::free(device_object->pooling_1_output, myQueue);
	sycl::free(device_object->conv_2_output, myQueue);
	sycl::free(device_object->pooling_2_output, myQueue);
	sycl::free(device_object->dense_layer_1_output, myQueue);
	sycl::free(device_object->dense_layer_2_output, myQueue);
	sycl::free(device_object->output_data, myQueue);
	#else 
	free(device_object->conv_1_output);
	free(device_object->pooling_1_output);
	free(device_object->conv_2_output);
	free(device_object->pooling_2_output);
	free(device_object->dense_layer_1_output);
	free(device_object->dense_layer_2_output);
	free(device_object->output_data);
	#endif
}