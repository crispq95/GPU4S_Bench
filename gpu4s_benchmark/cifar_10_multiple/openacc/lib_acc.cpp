#include "../benchmark_library.h"
#include <cstring>
#include <cmath>

#define max(a,b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a > _b ? _a : _b; })


void convolution_kernel(const bench_t *A, bench_t *B, const bench_t *kernel,const int n, const int m, const int w, const int kernel_size)
{
	int kernel_rad = kernel_size / 2;

	#pragma acc parallel loop present(A, B, kernel)
	for (unsigned int block = 0; block < n*n; ++block)
	{
		const unsigned int x = block/n;
		const unsigned int y = block%n;
		bench_t sum = 0;
		for(int i = -kernel_rad; i <= kernel_rad; ++i)
		{
			for(int j = -kernel_rad; j <= kernel_rad; ++j){
				bench_t value = 0;
				if (i + x < 0 || j + y < 0)
				{
					value = 0;
				}
				else if ( i + x > n - 1 || j + y > n - 1)
				{
					value = 0;
				}
				else
				{
					value = A[(x + i)*n+(y + j)];
				}
				sum += value * kernel[(i+kernel_rad)* kernel_size + (j+kernel_rad)];
			}
		}			
		B[x * n + y] = sum;
	}

	
}


void relu_kernel(const bench_t *A, bench_t *B, const int size)
{
	// Compute traditional relu approach 
	#pragma acc parallel loop present(A, B)
	for (unsigned int i = 0; i < size; ++i)
	{
		for (unsigned int j = 0; j < size; ++j)
		{
			if (A[i*size+j] > 0)
			{
				B[i*size+j] = A[i*size+j];
			}
			else 
			{
				B[i*size+j] = 0;
			}
		}
	}
}


void relu_linear_kernel(const bench_t *A, bench_t *B, const int size)
{
	// Compute traditional relu approach 
	#pragma acc parallel loop present(A,B)
	for (unsigned int i = 0; i < size; ++i)
	{
		if (A[i] > 0)
		{
			B[i] = A[i];
		}
		else 
		{
			B[i] = 0;
		}
	}
}

// REVISAR
void max_pooling_kernel(const bench_t *A, bench_t *B, const int size, const unsigned int stride,  const unsigned int lateral_stride)
{	
	bench_t max_value = 0;
	
	#pragma acc parallel loop collapse(2) firstprivate(max_value) present(B, A)
	for (unsigned int i = 0; i < size; i+= stride)
	{
		for (unsigned int j = 0; j < size; j+= stride)
		{
			max_value = A[i*size+j];
			//#pragma acc parallel for loop reduction(max:max_value) collapse(2)
			for(unsigned int x = 0; x < stride; ++x)
			{
				for(unsigned int y = 0; y < stride; ++y)
				{
					max_value = max(max_value, A[(i + x) * size + (j +y)]);
				}
			}
			B[(i / stride)* lateral_stride + (j/stride)] = max_value;
		}
	}
}


void lrn_kernel(const bench_t *A, bench_t *B, const int size)
{
	#pragma acc parallel loop collapse(2) present(A,B)
	for (unsigned int i = 0; i < size; ++i)
	{
		for (unsigned int j = 0; j < size; ++j)
		{
			B[i*size+j] = A[i*size+j]/pow((K+ALPHA*pow(A[i*size+j],2)),BETA);
		}
	}
}


void matrix_multiplication_kernel(const bench_t *A,const bench_t *B,  bench_t *C, const int n, const int m, const int w)
{
	#pragma acc parallel loop present(A,B,C)
	for (unsigned int i = 0; i < n; i++)
	{
		for (unsigned int j = 0; j < m; j++)
		{
			bench_t acumulated = 0;
			for (unsigned int k = 0; k < w; k++)
			{   
				acumulated += A[i*w+k] * B[k*m+j];
			}
			C[i*m+j] = acumulated;
		}
	}
}


void softmax_kernel(const bench_t *A, bench_t *B, const int size)
{	
	bench_t sum_values = 0;
	bench_t value = 0;

	
	#pragma acc parallel loop reduction(+:sum_values) present(A, B)
	for (unsigned int i = 0; i < size; i++)
	{
		value = expf (A[i]);
		sum_values += value;
		B[i] = value;
	}

	#pragma acc parallel loop  present(B)
	for (unsigned int i = 0; i < size; i++)
	{
		B[i] = (B[i]/sum_values);
	}

}


void init(GraficObject *device_object, char* device_name)
{
	init(device_object, 0,0, device_name);
}


void init(GraficObject *device_object, int platform ,int device, char* device_name)
{
	// TBD Feature: device name. -- Bulky generic platform implementation
	strcpy(device_name,"Generic device");
}


bool device_memory_init(GraficObject *device_object, unsigned int input_data, unsigned int output_data, unsigned int kernel_1, unsigned int kernel_2, unsigned int stride_1, unsigned int stride_2, unsigned int neurons_dense_1, unsigned int neurons_dense_2, unsigned int number_of_images)
{
	const unsigned int size_pooling_1 = input_data / stride_1;
    const unsigned int size_pooling_2 = size_pooling_1 / stride_2;
    const unsigned int weights_layer_1 = size_pooling_2 * size_pooling_2 * neurons_dense_1;
    const unsigned int weights_layer_2 = neurons_dense_1 * neurons_dense_2; 

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
	return true;
}


void copy_memory_to_device(GraficObject *device_object, bench_t* input_data, bench_t* kernel_1_data, bench_t* kernel_2_data, bench_t* weights_1 ,bench_t* weights_2,unsigned int input , unsigned int kernel_size_1, unsigned int kernel_size_2, unsigned int weights_1_size, unsigned int weights_2_size, unsigned int number_of_images)
{
	device_object->input_data = input_data;
	device_object->kernel_1 = kernel_1_data;
	device_object->kernel_2 = kernel_2_data;
	device_object->dense_layer_1_weights = weights_1;
	device_object->dense_layer_2_weights = weights_2;
}


void execute_kernel(GraficObject *device_object, unsigned int input_data, unsigned int output_data, unsigned int kernel_1, unsigned int kernel_2, unsigned int stride_1, unsigned int stride_2, unsigned int neurons_dense_1, unsigned int neurons_dense_2, unsigned int number_of_images)
{
	// Start compute timer
	const double start_wtime = omp_get_wtime();

	const unsigned int size_lateral_1 = input_data / stride_1;
	const unsigned int size_lateral_2 = size_lateral_1 / stride_2;

	bench_t* aux_output_data = device_object->output_data;
    bench_t* aux_input_data = device_object->input_data;

	#pragma acc enter data copyin(device_object[0:11])
	{
	#pragma acc enter data copyin(device_object->input_data[0:input_data*input_data], device_object->kernel_1[0:kernel_1]) \
	create(device_object->conv_1_output[0:input_data*input_data])
	{

	for(unsigned int position = 0; position < number_of_images; ++position)
    {
		aux_input_data = device_object->input_data + position * input_data * input_data;
        aux_output_data = device_object->output_data + position * output_data;
	
		// 1-1 Step convolution
		convolution_kernel(aux_input_data, device_object->conv_1_output, device_object->kernel_1, input_data, input_data, input_data, kernel_1);

		// 1-2 Step activation
		relu_kernel(device_object->conv_1_output, device_object->conv_1_output, input_data);
		
		// 1-3 Step pooling
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

	#pragma acc exit data copyout(device_object->output_data[0:neurons_dense_2]) 
	}
	}

	// End compute timer
	device_object->elapsed_time = omp_get_wtime() - start_wtime;
}


void copy_memory_to_host(GraficObject *device_object, bench_t* h_C, int size, unsigned int number_of_images)
{	     
	memcpy(h_C, &device_object->output_data[0], sizeof(bench_t)*size*number_of_images);
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
	free(device_object->conv_1_output);
	free(device_object->pooling_1_output);
	free(device_object->conv_2_output);
	free(device_object->pooling_2_output);
	free(device_object->dense_layer_1_output);
	free(device_object->dense_layer_2_output);
	free(device_object->output_data);
}