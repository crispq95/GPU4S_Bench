#include "../benchmark_library.h"
#include <cmath>
#include <cstring>

void binary_reverse_kernel(bench_t *Br, unsigned int mode, sycl::id<1> idx)
{
    int64_t i=idx, j; 

    j = i; 
    j = (j & 0x55555555) << 1 | (j & 0xAAAAAAAA) >> 1;                                                                      
    j = (j & 0x33333333) << 2 | (j & 0xCCCCCCCC) >> 2;                                                                      
    j = (j & 0x0F0F0F0F) << 4 | (j & 0xF0F0F0F0) >> 4;                                                                      
    j = (j & 0x00FF00FF) << 8 | (j & 0xFF00FF00) >> 8;                                                                      
    j = (j & 0x0000FFFF) << 16 | (j & 0xFFFF0000) >> 16;                                                                    
    j >>= (32-mode);


    unsigned int position = j * 2;  

    Br[position] = Br[i *2];                                                                                                
    Br[position + 1] = Br[i *2 + 1]; 
}


void fft_kernel( bench_t *Br, const int64_t mmax, const bench_t wr, const bench_t wi, sycl::id<1> idx)
{
    int i = idx[0]; 
    int j=i+mmax;

    bench_t tempr = wr*Br[j-1] - wi*Br[j];
    bench_t tempi = wr*Br[j] + wi*Br[j-1];
    
    Br[j-1] = Br[i-1] - tempr;
    Br[j] = Br[i] - tempi;
    Br[i-1] += tempr;
    Br[i] += tempi;
}

void init(GraficObject *device_object, char* device_name)
{
	init(device_object, 0,0, device_name);
}


void init(GraficObject *device_object, int platform ,int device, char* device_name)
{
	std::cout << "Using device: " << myQueue.get_device().get_info<sycl::info::device::name>() << "\n";
}


bool device_memory_init(GraficObject *device_object, int64_t size)
{
    #ifdef USM
	device_object->d_Br = sycl::malloc_device<bench_t>(size, myQueue);
    device_object->d_B = sycl::malloc_device<bench_t>(size, myQueue);
    #else 
    device_object->d_Br = (bench_t*) malloc ( size * sizeof(bench_t*));
	#endif
	return true;
}

void copy_memory_to_device(GraficObject *device_object, bench_t* h_B,int64_t size)
{
    #ifdef USM
    myQueue.memcpy(device_object->d_Br, h_B, (size)*sizeof(bench_t)).wait();
    #else 
    // device_object->d_Br = h_B;
    memcpy(device_object->d_Br, &h_B[0], sizeof(bench_t)*size);
    #endif
}

void execute_kernel(GraficObject *device_object, int64_t size)
{
	// Start compute timer
	const double start_wtime = omp_get_wtime();

	int64_t loop_w = 0, loop_for_1 = 0, loop_for_2 = 0; 
	int64_t mmax; 
    int64_t n, m, istep, i,  j;
    bench_t wtemp, wr, wpr, wpi, wi, theta;
    bench_t tempr, tempi;
 
    // reverse-binary reindexing
    const unsigned int mode = (unsigned int)log2(size);
    const unsigned int s = size; 

    n = size<<1;
    j=1;
    #ifdef USM 
    myQueue.parallel_for(sycl::range<1>{s}, 
        [=, d_Br_local=device_object->d_Br](sycl::id<1> idx){
            binary_reverse_kernel(d_Br_local, mode, idx); 
    }).wait();
    
    #else 
    // // NOT WORKING
    // sycl::buffer<bench_t> buffBr(device_object->d_Br, (s*2));
    
    // myQueue.submit([&](sycl::handler& cgh) {
    //     auto accBr = buffBr.get_access<sycl::access::mode::read_write>(cgh);
    //     cgh.parallel_for<class reduction_kernel_AB>(sycl::range<1>{s}, 
    //     [=](sycl::id<1> idx){
    //         binary_reverse_kernel(accBr.get_pointer(), mode, idx); 
    //     });

    // }).wait();
    #endif

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
            #ifdef USM 
            myQueue.parallel_for(sycl::range<1>{s}, 
            [=, d_Br_local=device_object->d_Br](sycl::id<1> idx){
                fft_kernel( d_Br_local, mmax, wr, wi, idx);
            }).wait();
            #else 
            // sycl::buffer<bench_t> buffBr(device_object->d_Br, (s*2));

            // myQueue.submit([&](sycl::handler& cgh) {
            //     auto accBr = buffBr.get_access<sycl::access::mode::read_write>(cgh);
                
            //     cgh.parallel_for(sycl::range<1>{s}, [=](sycl::id<1> idx){
            //         fft_kernel(accBr.get_pointer(), mmax, wr, wi, idx);
            //     });
            // }).wait();
            #endif
            wtemp=wr;
            wr += wr*wpr - wi*wpi;
            wi += wi*wpr + wtemp*wpi;
            ++loop_for_2;

        }
        loop_for_2 = 0;
        mmax=istep;
    	++loop_w;    
    }

    // printf("device_object->d_Br[0]=%f\n", device_object->d_B[0]); 
	// End compute timer
	device_object->elapsed_time = omp_get_wtime() - start_wtime;
}


void copy_memory_to_host(GraficObject *device_object, bench_t* h_B, int64_t size)
{	     
    #ifdef USM
    myQueue.memcpy(h_B, device_object->d_Br, (size)*sizeof(bench_t)).wait();
    #else 
    memcpy(h_B, &device_object->d_Br[0], sizeof(bench_t)*size);
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
        sycl::free(device_object->d_B, myQueue); 
    #else 
        free(device_object->d_Br); 
    #endif

    return;
}