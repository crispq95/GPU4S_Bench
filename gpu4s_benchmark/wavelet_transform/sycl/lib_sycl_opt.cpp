#include "../benchmark_library.h"
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
	device_object->d_B = sycl::malloc_device<bench_t>(size_b_matrix, myQueue);
	// printf("SIZE B : %d\n", size_b_matrix); 
	#else
	device_object->d_B = (bench_t*) malloc ( size_b_matrix * sizeof(bench_t*));
	#endif

	#ifdef FLOAT
	#ifdef USM
	device_object->low_filter = sycl::malloc_device<bench_t>(LOWPASSFILTERSIZE, myQueue);
	device_object->high_filter = sycl::malloc_device<bench_t>(HIGHPASSFILTERSIZE, myQueue);
	#else
	device_object->low_filter = (bench_t*) malloc (LOWPASSFILTERSIZE * sizeof(bench_t));
	device_object->high_filter = (bench_t*) malloc (HIGHPASSFILTERSIZE * sizeof(bench_t));
	#endif
	#endif
	
   	return true;
}


void copy_memory_to_device(GraficObject *device_object, bench_t* h_A, unsigned int size_a)
{
	#ifdef USM
	device_object->d_A = sycl::malloc_device<bench_t>(size_a, myQueue);
	myQueue.memcpy(device_object->d_A, h_A, (size_a)*sizeof(bench_t)).wait();
	#else
	printf("SIZE A %d\n", size_a); 

	device_object->d_A = h_A;
	#endif
	
	#ifdef FLOAT
	#ifdef USM
	myQueue.memcpy(device_object->low_filter, lowpass_filter, (LOWPASSFILTERSIZE)*sizeof(bench_t)).wait();
	myQueue.memcpy(device_object->high_filter, highpass_filter, (HIGHPASSFILTERSIZE)*sizeof(bench_t)).wait();
	#else
	memcpy(&device_object->low_filter[0], lowpass_filter, sizeof(bench_t)*LOWPASSFILTERSIZE);
	memcpy(&device_object->high_filter[0], highpass_filter, sizeof(bench_t)*HIGHPASSFILTERSIZE);
	#endif
	#endif
}


void execute_kernel(GraficObject *device_object, unsigned int size)
{
	// Start compute timer
	const double start_wtime = omp_get_wtime();

	// the output will be in the B array the lower half will be the lowpass filter and the half_up will be the high pass filter
	#ifdef INT
	printf("Working with ints\n");

	#ifdef USM	//USM INT
	printf("INT USM ______________________\n");
	unsigned int full_size = size * 2;

	myQueue
		.parallel_for<class wavelet_transform_high>(
			sycl::range<1>{size}, 
			[=, d_A_local=device_object->d_A, d_B_local=device_object->d_B]\
			(sycl::id<1> idx){
				int i = idx[0];
				bench_t sum_value_high = 0;
				// specific cases
				if(i == 0){
					sum_value_high = d_A_local[1] - (int)( ((9.0/16.0) * (d_A_local[0] + d_A_local[2])) - ((1.0/16.0) * (d_A_local[2] + d_A_local[4])) + (1.0/2.0));
				}
				else if(i == size -2){
					sum_value_high = d_A_local[2*size - 3] - (int)( ((9.0/16.0) * (d_A_local[2*size -4] + d_A_local[2*size -2])) - ((1.0/16.0) * (d_A_local[2*size - 6] + d_A_local[2*size - 2])) + (1.0/2.0));
				}
				else if(i == size - 1){
					sum_value_high = d_A_local[2*size - 1] - (int)( ((9.0/8.0) * (d_A_local[2*size -2])) -  ((1.0/8.0) * (d_A_local[2*size - 4])) + (1.0/2.0));
				}
				else{
					// generic case
					sum_value_high = d_A_local[2*i+1] - (int)( ((9.0/16.0) * (d_A_local[2*i] + d_A_local[2*i+2])) - ((1.0/16.0) * (d_A_local[2*i - 2] + d_A_local[2*i + 4])) + (1.0/2.0));
				}
			
			//store
			d_B_local[i+size] = sum_value_high;

			}).wait();

	myQueue
		.parallel_for<class wavelet_transform_low>(
			sycl::range<1>{size}, 
			[=, d_A_local=device_object->d_A, d_B_local=device_object->d_B]\
			(sycl::id<1> idx){
				int i = idx[0];
				bench_t sum_value_low = 0;
				
				if(i == 0){
					sum_value_low = d_A_local[0] - (int)(- (d_B_local[size]/2.0) + (1.0/2.0));
				}
				else
				{
					sum_value_low = d_A_local[2*i] - (int)( - (( d_B_local[i + size -1] +  d_B_local[i + size])/ 4.0) + (1.0/2.0) );
				}
				
				d_B_local[i] = sum_value_low;
			}).wait();
	

	#else	//AB INT
	printf ("INT AB_________\n");
	unsigned int full_size = size * 2;
	//create buffers 
	sycl::buffer<bench_t> buffA(device_object->d_A, (size*2));
	sycl::buffer<bench_t> buffB(device_object->d_B, (size*size));

	myQueue.submit([&](sycl::handler& cgh){
		//create accessors 
		auto accA = buffA.get_access<sycl::access::mode::read>(cgh);
		auto accB = buffB.get_access<sycl::access::mode::write>(cgh);

		cgh.parallel_for<class wavelet_transform_high>(
			sycl::range<1>{size}, 
			[=](sycl::id<1> i){
				bench_t sum_value_high = 0;

				if(i == 0){
					sum_value_high = accA[1]- (int)( ((9.0/16.0) * (accA[0] + accA[2])) - ((1.0/16.0) * (accA[2] + accA[4])) + (1.0/2.0));
				}
				else if(i == size -2){
					sum_value_high = accA[2*size - 3] - (int)( ((9.0/16.0) * (accA[2*size -4] + accA[2*size -2])) - ((1.0/16.0) * (accA[2*size - 6] + accA[2*size - 2])) + (1.0/2.0));
				}
				else if(i == size - 1){
					sum_value_high = accA[2*size - 1] - (int)( ((9.0/8.0) * (accA[2*size -2])) -  ((1.0/8.0) * (accA[2*size - 4])) + (1.0/2.0));
				}
				else{
					// generic case
					sum_value_high = accA[2*i+1] - (int)( ((9.0/16.0) * (accA[2*i] + accA[2*i+2])) - ((1.0/16.0) * (accA[2*i - 2] + accA[2*i + 4])) + (1.0/2.0));
				}

				accB[i+size] = sum_value_high;
			});
	});

	myQueue.submit([&](sycl::handler& cgh){
		auto accA = buffA.get_access<sycl::access::mode::read>(cgh);
		auto accB = buffB.get_access<sycl::access::mode::read_write>(cgh);

		cgh.parallel_for<class wavelet_transform_low>(
			sycl::range<1>{size}, 
			[=](sycl::id<1> idx){
				int i = idx[0]; 
				bench_t sum_value_low = 0;

				if(i == 0){
					sum_value_low = accA[0]  -  (int)(- (accB[size]/2.0) + (1.0/2.0));
				}
				else{
					sum_value_low = accA[2*i] - (int)( - (( accB[i + size -1] +  accB[i + size])/ 4.0) + (1.0/2.0) );
				}
				
				accB[i] = sum_value_low;
			});
	});
	
	myQueue.wait();

	#endif 
	
	#else //ENDIF INT
	printf("Working with floats\n");

	// flotating part
	// unsigned int full_size = size * 2;
	// const int hi_end = LOWPASSFILTERSIZE / 2;
	unsigned int full_size = size * 2;
	int hi_start = -(LOWPASSFILTERSIZE / 2);
	int hi_end = LOWPASSFILTERSIZE / 2;
	int gi_start = -(HIGHPASSFILTERSIZE / 2 );
	int gi_end = HIGHPASSFILTERSIZE / 2;

	#ifdef USM	//USM FLOAT

	myQueue
		.parallel_for<class wavelet_transform_low>(
			sycl::range<1>{size}, 
			[=, d_A_local=device_object->d_A, d_B_local=device_object->d_B,\
			 low_filter_local=device_object->low_filter, high_filter_local=device_object->high_filter]\
			(sycl::id<1> idx){
				int i = idx[0]; 
				// loop over N elements of the input vector.
				bench_t sum_value_low = 0;

				//Lowpass filter
				for (int hi = hi_start; hi < hi_end + 1; ++hi){
					int x_position = (2 * i) + hi;
					if (x_position < 0) {
						// turn negative to positive
						x_position = x_position * -1;
					}
					else if (x_position > full_size - 1)
					{
						x_position = full_size - 1 - (x_position - (full_size -1 ));
					}
				// Restore the hi value to work with the array
				sum_value_low += low_filter_local[hi + hi_end] * d_A_local[x_position];
				}	//end low filter 

				d_B_local[i] = sum_value_low;
				bench_t sum_value_high =  0;

				//Highpass filter
				for (int gi = gi_start; gi < gi_end + 1; ++gi){
					int x_position = (2 * i) + gi + 1;
					if (x_position < 0) {
						// turn negative to positive
						x_position = x_position * -1;
					}
					else if (x_position >  full_size - 1)
					{
						x_position = full_size - 1 - (x_position - (full_size -1 ));
					}
					sum_value_high += high_filter_local[gi + gi_end] * d_A_local[x_position];
				}
				// store the value
				d_B_local[i+size] = sum_value_high;
	}).wait();
	// myQueue
	// 	.parallel_for<class wavelet_transform>(
	// 		sycl::range<1>{size}, 
	// 		[=, d_A_local=device_object->d_A, d_B_local=device_object->d_B,\
	// 		 low_filter_local=device_object->low_filter, high_filter_local=device_object->high_filter]\
	// 		(sycl::id<1> idx){
	// 			int i = idx[0]; 
	// 			// loop over N elements of the input vector.
	// 			bench_t sum_value_low = 0;

	// 			int x_position = (2 * i);
	// 			sum_value_low = (low_filter_local[-4 + hi_end] * d_A_local[(x_position -4) < 0 ? (x_position-4) * -1 : (x_position-4) > full_size - 1 ?\
	// 			 full_size - 1 - ((x_position-4) - (full_size -1 )) : (x_position-4)]) + (low_filter_local[-3 + hi_end] * d_A_local[(x_position -3) < 0 ? (x_position-3) * -1 :\
	// 			  (x_position-3) > full_size - 1 ? full_size - 1 - ((x_position-3) - (full_size -1 )) : (x_position-3)]) + (low_filter_local[-2 + hi_end] * d_A_local[(x_position -2) < 0 ?\
	// 			   (x_position-2) * -1 : (x_position-2) > full_size - 1 ? full_size - 1 - ((x_position-2) - (full_size -1 )) : (x_position-2)]) + (low_filter_local[-1 + hi_end]\
	// 			    * d_A_local[(x_position -1) < 0 ? (x_position-1) * -1 : (x_position-1) > full_size - 1 ? full_size - 1 - ((x_position-1) - (full_size -1 )) : (x_position-1)]) +\
	// 				 (low_filter_local[hi_end] * d_A_local[(x_position) < 0 ? (x_position) * -1 : (x_position) > full_size - 1 ? full_size - 1 - ((x_position) - (full_size -1 )) : (x_position)])\
	// 				  + (low_filter_local[1 + hi_end] * d_A_local[(x_position + 1) < 0 ? (x_position + 1) * -1 : (x_position + 1) > full_size - 1 ? full_size - 1 - ((x_position + 1)\
	// 				   - (full_size -1 )) : (x_position+1)]) + (low_filter_local[+2 + hi_end] * d_A_local[(x_position +2) < 0 ? (x_position+2) * -1 : (x_position+2) > full_size - 1 ?\
	// 				    full_size - 1 - ((x_position+2) - (full_size -1 )) : (x_position+2)]) + (low_filter_local[3 + hi_end] * d_A_local[(x_position +3) < 0 ? (x_position + 3) * -1 :\
	// 					 (x_position + 3) > full_size - 1 ? full_size - 1 - ((x_position+3) - (full_size -1 )) : (x_position+3)]) + (low_filter_local[4 + hi_end] * d_A_local[(x_position +4) < 0 ?\
	// 					  (x_position+4) * -1 : (x_position+4) > full_size - 1 ? full_size - 1 - ((x_position+4) - (full_size -1 )) : (x_position+4)]);
	// 			d_B_local[i] = sum_value_low;

	// 		}).wait();


    // const int gi_end = HIGHPASSFILTERSIZE / 2;

	// myQueue
	// 	.parallel_for<class wavelet_transform_high>(
	// 		sycl::range<1>{size}, 
	// 		[=, d_A_local=device_object->d_A, d_B_local=device_object->d_B, high_filter_local=device_object->high_filter]\
	// 		(sycl::id<1> idx){
	// 		int i = idx[0]; 
	// 		bench_t sum_value_high = 0;

    //         int x_position = (2 * i) + 1;
    //         x_position = x_position < 0 ? x_position * -1 : x_position > full_size - 1 ? full_size - 1 - (x_position - (full_size -1 )) : x_position;

    //         sum_value_high = (high_filter_local[-3 + gi_end] * d_A_local[ (x_position - 3) < 0 ? (x_position - 3) * -1 : (x_position - 3) > full_size - 1 ? full_size - 1 - ((x_position- 3) - (full_size -1 )) : (x_position- 3)]) + \
	// 		(high_filter_local[-2 + gi_end] * d_A_local[ (x_position - 2) < 0 ? (x_position - 2) * -1 : (x_position - 2) > full_size - 1 ? full_size - 1 - ((x_position- 2) - \
	// 		(full_size -1 )) : (x_position- 2)]) + (high_filter_local[-1 + gi_end] * d_A_local[ (x_position - 1) < 0 ? (x_position - 1) \
	// 		* -1 : (x_position - 1) > full_size - 1 ? full_size - 1 - ((x_position- 1) - (full_size -1 )) : (x_position- 1)]) + \
	// 		(high_filter_local[gi_end] * d_A_local[ (x_position) < 0 ? (x_position) * -1 : (x_position) > full_size - 1 ? full_size - 1 - ((x_position) - (full_size -1 )) : (x_position)])\
	// 		 + (high_filter_local[1 + gi_end] * d_A_local[ (x_position  + 1) < 0 ? (x_position + 1) * -1 : (x_position + 1) > full_size - 1 ? full_size - 1 - ((x_position + 1) - (full_size -1 ))\
	// 		  : (x_position + 1)]) + (high_filter_local[2 + gi_end] * d_A_local[ (x_position + 2) < 0 ? \
	// 		(x_position + 2) * -1 : (x_position + 2) > full_size - 1 ? full_size - 1 - ((x_position + 2) - (full_size -1 )) : (x_position + 2)]) + \
	// 		(high_filter_local[3 + gi_end] * d_A_local[ (x_position + 3) < 0 ? (x_position + 3) * -1 : \
	// 		(x_position + 3) > full_size - 1 ? full_size - 1 - ((x_position + 3) - (full_size -1 )) : (x_position + 3)]);

	// 		d_B_local[i+size] = sum_value_high;

	// 		}).wait();
   
	#else 

	// //create buffers 
	auto buffA = sycl::buffer{device_object->d_A, sycl::range{size*size}};
	auto buffB = sycl::buffer{device_object->d_B, sycl::range{size*size}};
	auto buffLF = sycl::buffer{device_object->low_filter, sycl::range{LOWPASSFILTERSIZE}};
	auto buffHF = sycl::buffer{device_object->high_filter, sycl::range{HIGHPASSFILTERSIZE}};

	auto e = myQueue.submit([&](sycl::handler& cgh){
		//create accessors 
		auto accA = buffA.get_access<sycl::access::mode::read>(cgh);
		auto accB = buffB.get_access<sycl::access::mode::write>(cgh);
		auto accLF = buffLF.get_access<sycl::access::mode::read>(cgh);
		auto accHF = buffHF.get_access<sycl::access::mode::read>(cgh);
		
		cgh.parallel_for<class wavelet_transform>(
			sycl::range<1>{size}, 
			[=](sycl::id<1> idx){
				int i = idx[0]; 
				bench_t sum_value_low = 0;

				//Lowpass filter
				for (int hi = hi_start; hi < hi_end + 1; ++hi){
					signed int x_position = (2 * i) + hi;
					if (x_position < 0) {
						// turn negative to positive
						x_position = x_position * -1;
					}
					else if (x_position > full_size - 1)
					{
						x_position = full_size - 1 - (x_position - (full_size -1 ));
					}
					// Restore the hi value to work with the array
					sum_value_low += accLF[hi + hi_end] * accA[x_position];
				}	
				accB[i] = sum_value_low;

				bench_t sum_value_high =  0;
				//Highpass filter
				for (int gi = gi_start; gi < gi_end + 1; ++gi){
					int x_position = (2 * i) + gi + 1;
					if (x_position < 0) {
						// turn negative to positive
						x_position = x_position * -1;
					}
					else if (x_position >  full_size - 1){
						x_position = full_size - 1 - (x_position - (full_size -1 ));
					}
					sum_value_high += accHF[gi + gi_end] * accA[x_position];
				}
				// store the value
				accB[i+size] = sum_value_high;

			});
		}); //end submit

		e.wait();

	#endif //ENDIF AB
	#endif //ENDIF FLOAT

	// End compute timer
	device_object->elapsed_time = omp_get_wtime() - start_wtime;
}


void copy_memory_to_host(GraficObject *device_object, bench_t* h_C, int size)
{	     
	printf("COPY DtH\n");
	#ifdef USM  
	myQueue.memcpy(h_C, device_object->d_B, (size)*sizeof(bench_t)).wait();
	#else
	memcpy(h_C, &device_object->d_B[0], sizeof(bench_t)*size);
	#endif
}


float get_elapsed_time(GraficObject *device_object, bool csv_format, bool csv_format_timestamp, long int current_time)
{
	printf("Time\n");
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
		setvbuf(stdout, NULL, _IONBF, 0); 
		printf("Elapsed time Device->Host: %.10f milliseconds\n", (bench_t) 0);
    }
	return device_object->elapsed_time * 1000.f;
}


void clean(GraficObject *device_object)
{
	printf("Free\n");
	#ifdef USM  
	sycl::free(device_object->d_B ,myQueue);
	sycl::free(device_object->d_A ,myQueue);

	#ifdef FLOAT
	sycl::free(device_object->low_filter ,myQueue);
	sycl::free(device_object->high_filter ,myQueue);
	#endif
	#else 
	free(device_object->d_B);
	#ifdef FLOAT
	free(device_object->low_filter);
	free(device_object->high_filter);
	#endif
	#endif
}