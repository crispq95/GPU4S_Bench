#include "../benchmark_library.h"
#include <cmath>
#include <cstring>

void init(GraficObject *device_object, char* device_name)
{
	init(device_object, 0,0, device_name);
}


void init(GraficObject *device_object, int platform ,int device, char* device_name)
{
	std :: cout << "Using device: " << myQueue.get_device().get_info<sycl::info::device::name>() << "\n";
}


bool device_memory_init(GraficObject *device_object, int64_t size)
{
    #ifdef USM
    device_object->d_B = sycl::malloc_device<bench_t>(size, myQueue);
	device_object->d_Br = sycl::malloc_device<bench_t>(size, myQueue);
	#endif
	return true;
}


void copy_memory_to_device(GraficObject *device_object, bench_t* h_B,int64_t size)
{
    #ifdef USM
	myQueue.memcpy(device_object->d_B, h_B, (size)*sizeof(bench_t)).wait();
	#else
	device_object->d_Br = h_B;
	#endif
}


void execute_kernel(GraficObject *device_object,  int64_t size)
{
	// Start compute timer
	const double start_wtime = omp_get_wtime();

	// int64_t loop_w = 0, loop_for_1 = 0, loop_for_2 = 0; 
	// int64_t  mmax,   istep, i;
    // bench_t wtemp, wr, wpr, wpi, wi, theta;
    // bench_t tempr, tempi;
 
    // reverse-binary reindexing
    

    #ifdef USM
    // myQueue
	//    .parallel_for<class binary_reverse_kernel>(
	// 		sycl::range<1>{n}, 
	// 		[=, d_Br_local=device_object->d_Br, d_B_local=device_object->d_B]\
	// 		(sycl::id<1> idx){
            
    //         int i = idx[0]; 
    //         unsigned int position = 0;
    //         if (i < size){   
    //             //position = (__rbit(i) >> (32 - group)) * 2;
    //             position = ((((i * 0x0802LU & 0x22110LU) | (i * 0x8020LU & 0x88440LU)) * 0x10101LU >> 16) >> (32 - (int64_t)log2(size))) * 2; 
    //             d_Br_local[position] = d_B_local[i *2];
    //             d_Br_local[position + 1] = d_B_local[i *2 + 1];
    //         }
    //         }).wait();

	const unsigned int mode = (unsigned int)log2(size);
	unsigned int z, l, istep, mmax, m = 0;
    unsigned int n = (unsigned int)(size); 

    myQueue
	   .parallel_for<class binary_reverse_kernel>(
			sycl::range<1>{n}, 
			[=, d_Br_local=device_object->d_Br, d_B_local=device_object->d_B]\
			(sycl::id<1> idx){
                int i = idx[0]; 
                unsigned int j; 
	            unsigned int position = 0;

                // j = i;                                                                                                    
                // j = (j & 0x55555555) << 1 | (j & 0xAAAAAAAA) >> 1;                                                                      
                // j = (j & 0x33333333) << 2 | (j & 0xCCCCCCCC) >> 2;                                                                      
                // j = (j & 0x0F0F0F0F) << 4 | (j & 0xF0F0F0F0) >> 4;                                                                      
                // j = (j & 0x00FF00FF) << 8 | (j & 0xFF00FF00) >> 8;                                                                      
                // j = (j & 0x0000FFFF) << 16 | (j & 0xFFFF0000) >> 16;                                                                    
                // j >>= (32-mode);                      

                position = ((((i * 0x0802LU & 0x22110LU) | (i * 0x8020LU & 0x88440LU)) * 0x10101LU >> 16) >> (32 - (int64_t)log2(size))) * 2; 
                // position = j * 2;   

                d_Br_local[position] = d_B_local[i *2];                                                                                                
                d_Br_local[position + 1] = d_B_local[i *2 + 1];  

        }).wait();

    bench_t wpr, wpi, theta, wi, tempr, tempi, wtemp, wr = 0.f;
    mmax=2;
	n = size << 1;

    bench_t *h_B; 
    h_B = (bench_t*)malloc(sizeof(bench_t)*size);
    myQueue.memcpy(h_B, device_object->d_Br, (size)*sizeof(bench_t)).wait();
	bench_t* a = h_B;

    while (n>mmax) {
        istep = mmax<<1;
        theta = -(2*M_PI/mmax);
        wtemp = sin(0.5*theta);
        wpr = -2.0*wtemp*wtemp;
        wpi = sin(theta);
        wr = 1.0;
        wi = 0.0;
		#pragma omp task untied
        for (int m=1; m < mmax; m += 2) {
            for (int i=m; i <= n; i += istep) {
				int j=i+mmax;
				tempr = wr*a[j-1] - wi*a[j];
				tempi = wr * a[j] + wi*a[j-1];
                a[j-1] = a[i-1] - tempr;
                a[j] = a[i] - tempi;
                a[i-1] += tempr;
                a[i] += tempi;
            }
            wtemp=wr;
            wr += wr*wpr - wi*wpi;
            wi += wi*wpr + wtemp*wpi;
        }
        mmax=istep;
    }


	// n = (unsigned int)(size << 1);

    // while (n>mmax) {
    //     istep = mmax<<1;
    //     theta = -(2*M_PI/mmax);
    //     wtemp = sin(0.5*theta);
    //     wpr = -2.0*wtemp*wtemp;
    //     wpi = sin(theta);
    //     wr = 1.0;
    //     wi = 0.0;

    //     myQueue
	//    .parallel_for<class binary_reverse_kernel>(
	// 		sycl::range<1>{mmax}, 
	// 		[=, d_Br_local=device_object->d_Br, d_B_local=device_object->d_B]\
	// 		(sycl::id<1> idx){
    //             int m = idx[0]; 

    //             bench_t* a = d_Br_local;

    //             for (int i=m; i <= n; i += istep) {
    //                 int j=i+mmax;
    //                 tempr = wr*a[j-1] - wi*a[j];
    //                 tempi = wr * a[j] + wi*a[j-1];
    //                 a[j-1] = a[i-1] - tempr;
    //                 a[j] = a[i] - tempi;
    //                 a[i-1] += tempr;
    //                 a[i] += tempi;
    //             }
    //             wtemp=wr;
    //             wr += wr*wpr - wi*wpi;
    //             wi += wi*wpr + wtemp*wpi;
    //         }); 
	
    //     mmax=istep;
    // }

        
    #else 
    //TODO
      for (i=1; i<n; i+=2) {
        if (j>i) {
            std::swap(device_object->d_Br[j-1], device_object->d_Br[i-1]);
            std::swap(device_object->d_Br[j], device_object->d_Br[i]);
        }
        m = size;
        while (m>=2 && j>m) {
            j -= m;
            m >>= 1;
        }
        j += m;
    }


    // here begins the Danielson-Lanczos section
    mmax=2;
    while (n>mmax) {
        istep = mmax<<1;
        theta = -(2*M_PI/mmax);
        wtemp = sin(0.5*theta);
        wpr = -2.0*wtemp*wtemp;
        wpi = sin(theta);
        wr = 1.0;
        wi = 0.0;

       
        for (m=1; m < mmax; m += 2) {
            for (i=m; i <= n; i += istep) {
                j=i+mmax;
                tempr = wr*device_object->d_Br[j-1] - wi*device_object->d_Br[j];
                tempi = wr * device_object->d_Br[j] + wi*device_object->d_Br[j-1];
 				
                device_object->d_Br[j-1] = device_object->d_Br[i-1] - tempr;
                device_object->d_Br[j] = device_object->d_Br[i] - tempi;
                device_object->d_Br[i-1] += tempr;
                device_object->d_Br[i] += tempi;
                ++loop_for_1;
            }
            loop_for_1 = 0;
            
            wtemp=wr;
            wr += wr*wpr - wi*wpi;
            wi += wi*wpr + wtemp*wpi;
            ++loop_for_2;

        }
        loop_for_2 = 0;
        mmax=istep;
    	++loop_w;    
    }
    #endif
	
	// End compute timer
	device_object->elapsed_time = omp_get_wtime() - start_wtime;
}


void copy_memory_to_host(GraficObject *device_object, bench_t* h_B, int64_t size)
{	     
    #ifdef USM  
    myQueue.memcpy(h_B, device_object->d_Br, (size)*sizeof(bench_t)).wait();
    #endif
    return;
}


float get_elapsed_time(GraficObject *device_object, bool csv_format, bool csv_format_timestamp, long int current_time)
{
	if (csv_format_timestamp){
        printf("%.10f;%.10f;%.10f;%ld;\n",(bench_t) 0, device_object->elapsed_time , (bench_t) 0, current_time);
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
    sycl::free(device_object->d_Br, myQueue);
	#endif
    return;
}