# CONFIGURATION DIRECTIVES
# Compilers 
CC = g++
NVCC = /usr/local/cuda/bin/nvcc
NVC = /opt/nvidia/hpc_sdk/Linux_x86_64/22.2/compilers/bin/nvc++ 
SYCL = /home/cperalta/sycl_workspace/llvm/build/bin/clang++ 
HIP = /opt/rocm/hip/bin/hipcc
# the build target executable:
TARGET = relu
# FLAGS 
# CC compiler flags:
CFLAGS  = -g
# NVCC compiler flags
NVCCFLAGS = -arch compute_72 -code sm_72
# CUDA FLAGS
CUFLAGS = -I/usr/local/cuda/include/ -L/usr/local/cuda/lib64 -lcuda -lcudart 
# OPENCL FLAGS
OPFLAGS = -I/usr/local/cuda/include/ -L/oldroot/root/usr/lib/x86_64-linux-gnu/ -lOpenCL 
# OPENMP FLAGS
OMPFLAGS = -fopenmp -lm
OMPGPUFLAGS = -mp=gpu -gpu=cc61,managed -Minfo=mp 
# SYCL FLAGS 
#SYCLFLAGS =  -fsycl -fsycl-targets=nvptx64-nvidia-cuda -arch=sm_71 -g -Wall  -L. -O3 -lm -std=c++17  -fopenmp 
# SYCLFLAGS =  -fsycl  -g -Wall  -L. -O3 -lm -std=c++17  -fopenmp 
# SYCLFLAGS =  -fsycl -g -Wall  -L. -O3 -std=c++17  -fopenmp 
# OPENACC FLAGS
ACCFLAGS = -acc=gpu -ta=tesla:cc61 -Minfo=accel -lm  #-ta=tesla:cc61,managed #-acc=gpu
# HIP FLAGS
HIPFLAGS = -I/opt/rocm/hip/include -L/opt/rocm/hip/lib

# Littelendian and Bigendian flags, by default if value is not set is Littelendian if value is set to -DBIGENDIAN is Bigendian
# -DBIGENDIAN
ENDIANFLAGS = 

# Data type can be INT FLOAT and DOUBLE
DATATYPE =

# By default BLOCKSIZE equals 4.
BLOCKSIZE = 4
BLOCKSIZESQUARED = $(shell echo $(BLOCKSIZE)\*$(BLOCKSIZE) | bc)

# SYCL default device is CPU
DEVICETARGET = CPU
MODEL = USM

CPU = CPU

ifeq ($(DEVICETARGET),$(CPU))
   SYCLFLAGS = -fsycl  -g -Wall  -L. -O3 -lm -std=c++17  -fopenmp
else
   SYCLFLAGS = -fsycl -fsycl-targets=nvptx64-nvidia-cuda -arch=sm_71 -g -Wall  -L. -O3 -lm -std=c++17  -fopenmp 
endif

# FOLDERS
# CUDA FOLDER
CUFOLDER = ./cuda/
# OPENCL FOLDER
OPFOLDER = ./opencl/
# OPENMP FOLDER
OMPFOLDER = ./openmp/
# OPENACC FOLDER 
ACCFOLDER = ./openacc/
# SYCL FOLDER 
SYCLFOLDER = ./sycl/
# HIP FOLDER
HIPFOLDER = ./hip/
# CPU FOLDER
CPUFOLDER = ./cpu/
# CPU FUNCTIONS FOLDER
CPUFUNCTIONFOLDER = ./cpu_functions/
# OUTPUT FOLDER
OUTPUTFOLDER = ./bin/

# COMPILER MAIN
all:
	@echo "YOU NEED TO SELECT CUDA OR OPENCL -> make CUDA / make OpenCL / make CUDA-opt / make OpenCL-opt / make CUDA-lib / make OpenCL-lib"
# End Main
# Shortcuts
.PHONY: all-bin
all-bin: cuda cuda-opt cuda-lib opencl opencl-opt opencl-lib openmp openmp-opt openmp-lib openacc openacc-opt openacc-lib sycl hip hip-opt
.PHONY: all-cuda
all-cuda: cuda cuda-opt cuda-lib
.PHONY: all-opencl
all-opencl: opencl opencl-opt opencl-lib
.PHONY: all-openmp
all-openmp: openmp openmp-opt openmp-lib
.PHONY: all-openmpgpu
all-openmpgpu: openmpgpu openmpgpu-opt openmpgpu-lib
.PHONY: all-openacc
all-openacc: openacc openacc-opt openacc-lib
.PHONY: all-sycl 
all-sycl: sycl sycl-opt sycl-lib
.PHONY: all-hip
all-hip: hip hip-opt
.PHONY: CUDA
CUDA: cuda
.PHONY: OpenCL
OpenCL: opencl
.PHONY: OpenMP
OpenMP: openmp
.PHONY: OpenMPGPU
OpenMPGPU: openmpgpu
.PHONY: OpenACC
OpenAcc: openacc
.PHONY: SYCL
SYCL: sycl 
.PHONY: Hip
Hip: hip
.PHONY: CUDA-opt
CUDA-opt: cuda-opt
.PHONY: OpenCL-opt
OpenCL-opt: opencl-opt
.PHONY: OpenMP-opt
OpenMP-opt: openmp-opt
.PHONY: OpenMPGPU-opt
OpenMPGPU-opt: openmpgpu-opt
.PHONY: OpenACC-opt
OpenACC-opt: openacc-opt
.PHONY: SYCL-opt
SYCL-opt: sycl-opt
.PHONY: Hip-opt
Hip-opt: hip-opt
.PHONY: CUDA-lib
CUDA-lib: cuda-lib
.PHONY: OpenCL-lib
OpenCL-lib: opencl-lib
.PHONY: OpenMP-lib
OpenMP-lib: openmp-lib
.PHONY: OpenMPGPU-lib
OpenMPGPU-lib: openmpgpu-lib
.PHONY: OpenACC-lib
OpenACC-lib: openacc-lib
.PHONY: SYCL-lib
SYCL-lib: sycl-lib
# End Shortcuts
# CPU part
cpu_functions.o: $(CPUFUNCTIONFOLDER)cpu_functions.cpp
	$(CC) $(ENDIANFLAGS) -D$(DATATYPE) -c $(CPUFUNCTIONFOLDER)cpu_functions.cpp -o $(CPUFUNCTIONFOLDER)cpu_functions.o $(CFLAGS) 
# End CPU 

# CUDA part
.PHONY: cuda
cuda: main_cuda

lib_cuda.o: $(CUFOLDER)lib_cuda.cu
	$(NVCC) -DCUDA -D$(DATATYPE) -DBLOCK_SIZE=$(BLOCKSIZE) -c $(CUFOLDER)lib_cuda.cu -o $(CUFOLDER)lib_cuda.o $(NVCCFLAGS)


main_cuda: main.cpp lib_cuda.o cpu_functions.o
	mkdir -p $(OUTPUTFOLDER)
	$(CC) -D$(DATATYPE) -DCUDA main.cpp $(CUFOLDER)lib_cuda.o $(CPUFUNCTIONFOLDER)cpu_functions.o -o $(OUTPUTFOLDER)$(TARGET)_cuda_$(shell echo $(DATATYPE) | tr A-Z a-z)_$(BLOCKSIZESQUARED) $(CUFLAGS) $(CFLAGS) -lstdc++
# End CUDA

# OpenCL Part
opencl:  main_opencl

lib_opencl.o: $(OPFOLDER)lib_opencl.cpp 
	$(CC) -D$(DATATYPE) -DBLOCK_SIZE=$(BLOCKSIZE) -DOPENCL -c $(OPFOLDER)lib_opencl.cpp -o $(OPFOLDER)lib_opencl.o $(CFLAGS) $(OPFLAGS)

main_opencl: main.cpp lib_opencl.o cpu_functions.o
	mkdir -p $(OUTPUTFOLDER) 
	$(CC) -D$(DATATYPE) -DOPENCL main.cpp $(OPFOLDER)lib_opencl.o  $(CPUFUNCTIONFOLDER)cpu_functions.o -o $(OUTPUTFOLDER)$(TARGET)_opencl_$(shell echo $(DATATYPE) | tr A-Z a-z)_$(BLOCKSIZESQUARED) $(CFLAGS) $(OPFLAGS) 

# End OpenCL

# OpenMP Part
openmp:  main_openmp

lib_omp.o: $(OMPFOLDER)lib_omp.cpp 
	export OMP_NUM_THREADS=8
	$(CC) -D$(DATATYPE) -DBLOCK_SIZE=$(BLOCKSIZE) -DOPENMP -c $(OMPFOLDER)lib_omp.cpp -o $(OMPFOLDER)lib_omp.o $(CFLAGS) $(OMPFLAGS)

main_openmp: main.cpp lib_omp.o cpu_functions.o
	mkdir -p $(OUTPUTFOLDER) 
	$(CC) -D$(DATATYPE) -DOPENMP main.cpp $(OMPFOLDER)lib_omp.o  $(CPUFUNCTIONFOLDER)cpu_functions.o -o $(OUTPUTFOLDER)$(TARGET)_omp_$(shell echo $(DATATYPE) | tr A-Z a-z) $(CFLAGS) $(OMPFLAGS)

# End OpenMP

# OpenMP GPU Part
openmpgpu:  main_openmp_gpu

lib_omp_gpu.o: $(OMPFOLDER)lib_omp.cpp 
	$(NVC) -D$(DATATYPE) -DBLOCK_SIZE=$(BLOCKSIZE) -DTARGET_GPU -DOPENMP -c $(OMPFOLDER)lib_omp.cpp -o $(OMPFOLDER)lib_omp_gpu.o $(CFLAGS) $(OMPGPUFLAGS)

main_openmp_gpu: main.cpp lib_omp_gpu.o cpu_functions.o
	mkdir -p $(OUTPUTFOLDER) 
	$(NVC) -D$(DATATYPE) -DOPENMP main.cpp $(OMPFOLDER)lib_omp_gpu.o  $(CPUFUNCTIONFOLDER)cpu_functions.o -o $(OUTPUTFOLDER)$(TARGET)_omp_gpu_$(shell echo $(DATATYPE) | tr A-Z a-z) $(CFLAGS) $(OMPGPUFLAGS)
# End OpenMP GPU Part

# OpenACC Part 
openacc: main_openacc

lib_acc.o: $(ACCFOLDER)lib_acc.cpp
	$(NVC) -D$(DATATYPE) -DBLOCK_SIZE=$(BLOCKSIZE) -DOPENACC -c $(ACCFOLDER)lib_acc.cpp -o $(ACCFOLDER)lib_acc.o $(CFLAGS) $(ACCFLAGS)

main_openacc: main.cpp lib_acc.o cpu_functions.o 
	mkdir -p $(OUTPUTFOLDER)
	$(NVC) -D$(DATATYPE) -DOPENACC main.cpp $(ACCFOLDER)lib_acc.o $(CPUFUNCTIONFOLDER)cpu_functions.o -o $(OUTPUTFOLDER)$(TARGET)_acc_$(shell echo $(DATATYPE) | tr A-Z a-z) $(CFLAGS) $(ACCFLAGS)
# End OpenACC

# SYCL Part 
sycl: main_sycl

lib_sycl.o: $(SYCLFOLDER)lib_sycl.cpp
	$(SYCL) -D$(DATATYPE) -D$(MODEL) -D$(DEVICETARGET) -DBLOCK_SIZE=$(BLOCKSIZE) -DSYCL -c $(SYCLFOLDER)lib_sycl.cpp -o $(SYCLFOLDER)lib_sycl.o $(CFLAGS) $(SYCLFLAGS)

main_sycl: main.cpp lib_sycl.o cpu_functions.o 
	mkdir -p $(OUTPUTFOLDER)
	$(SYCL) -D$(DATATYPE) -D$(MODEL) -D$(DEVICETARGET) main.cpp $(SYCLFOLDER)lib_sycl.o $(CPUFUNCTIONFOLDER)cpu_functions.o -o $(OUTPUTFOLDER)$(TARGET)_sycl_$(shell echo $(DATATYPE) | tr A-Z a-z) $(CFLAGS) $(SYCLFLAGS)
# End SYCL

# Hip part
hip: main_hip

lib_hip.o: $(HIPFOLDER)lib_hip.cpp
	$(HIP) -D$(DATATYPE) -DBLOCK_SIZE=$(BLOCKSIZE) -DHIP -c $(HIPFOLDER)lib_hip.cpp -o $(HIPFOLDER)lib_hip.o $(CFLAGS) $(HIPFLAGS)

main_hip: main.cpp lib_hip.o cpu_functions.o
	mkdir -p $(OUTPUTFOLDER) 
	$(HIP) -D$(DATATYPE) -DHIP main.cpp -x none $(HIPFOLDER)lib_hip.o  $(CPUFUNCTIONFOLDER)cpu_functions.o -o $(OUTPUTFOLDER)$(TARGET)_hip_$(shell echo $(DATATYPE) | tr A-Z a-z)_$(BLOCKSIZESQUARED) $(CFLAGS) $(HIPFLAGS)
# End Hip 

# CPU part
.PHONY: cpu
cpu: main_cpu
lib_cpu.o: $(CPUFOLDER)lib_cpu.cpp
	$(CC) -D$(DATATYPE) -DBLOCK_SIZE=$(BLOCKSIZE) -c $(CPUFOLDER)lib_cpu.cpp -o $(CPUFOLDER)lib_cpu.o $(CPUFLAGS)


main_cpu: main.cpp lib_cpu.o cpu_functions.o
	mkdir -p $(OUTPUTFOLDER)
	$(CC) -D$(DATATYPE) main.cpp $(CPUFOLDER)lib_cpu.o $(CPUFUNCTIONFOLDER)cpu_functions.o -o $(OUTPUTFOLDER)$(TARGET)_cpu_$(shell echo $(DATATYPE) | tr A-Z a-z)_$(BLOCKSIZESQUARED) $(CPUFLAGS) $(CFLAGS)

# End CPU 

# CUDA part optimized
.PHONY: cuda
cuda-opt: main_cuda_opt

lib_cuda_opt.o: $(CUFOLDER)lib_cuda_opt.cu
	$(NVCC) -DCUDA -D$(DATATYPE) -DBLOCK_SIZE=$(BLOCKSIZE) -c $(CUFOLDER)lib_cuda_opt.cu -o $(CUFOLDER)lib_cuda_opt.o $(NVCCFLAGS)


main_cuda_opt: main.cpp lib_cuda_opt.o cpu_functions.o
	mkdir -p $(OUTPUTFOLDER)
	$(CC) -D$(DATATYPE) -DCUDA main.cpp $(CUFOLDER)lib_cuda_opt.o $(CPUFUNCTIONFOLDER)cpu_functions.o -o $(OUTPUTFOLDER)$(TARGET)_cuda_opt_$(shell echo $(DATATYPE) | tr A-Z a-z)_$(BLOCKSIZESQUARED) $(CUFLAGS) $(CFLAGS) -lstdc++

# End CUDA optimized

# OpenCL Part optimized
opencl-opt:  main_opencl_opt

lib_opencl_opt.o: $(OPFOLDER)lib_opencl_opt.cpp 
	$(CC) -D$(DATATYPE) -DBLOCK_SIZE=$(BLOCKSIZE) -DOPENCL -c $(OPFOLDER)lib_opencl_opt.cpp -o $(OPFOLDER)lib_opencl_opt.o $(CFLAGS) $(OPFLAGS)

main_opencl_opt: main.cpp lib_opencl_opt.o cpu_functions.o
	mkdir -p $(OUTPUTFOLDER) 
	$(CC) -D$(DATATYPE) -DOPENCL main.cpp $(OPFOLDER)lib_opencl_opt.o  $(CPUFUNCTIONFOLDER)cpu_functions.o -o $(OUTPUTFOLDER)$(TARGET)_opencl_opt_$(shell echo $(DATATYPE) | tr A-Z a-z)_$(BLOCKSIZESQUARED) $(CFLAGS) $(OPFLAGS) 

# End OpenCL optimized

# OpenMP Part optimized
openmp-opt:  main_openmp_opt

lib_omp_opt.o: $(OMPFOLDER)lib_omp_opt.cpp 
	export OMP_NUM_THREADS=8
	$(CC) -D$(DATATYPE) -DBLOCK_SIZE=$(BLOCKSIZE) -DOPENMP -c $(OMPFOLDER)lib_omp_opt.cpp -o $(OMPFOLDER)lib_omp_opt.o $(CFLAGS) $(OMPFLAGS)

main_openmp_opt: main.cpp lib_omp_opt.o cpu_functions.o
	mkdir -p $(OUTPUTFOLDER) 
	$(CC) -D$(DATATYPE) -DOPENMP main.cpp $(OMPFOLDER)lib_omp_opt.o  $(CPUFUNCTIONFOLDER)cpu_functions.o -o $(OUTPUTFOLDER)$(TARGET)_omp_opt_$(shell echo $(DATATYPE) | tr A-Z a-z) $(CFLAGS) $(OMPFLAGS)
# End OpenMP optimized

# OpenMP GPU Part optimized
openmpgpu-opt:  main_openmp_gpu_opt

lib_omp_gpu_opt.o: $(OMPFOLDER)lib_omp_opt.cpp 
	export OMP_NUM_THREADS=8
	$(NVC) -D$(DATATYPE) -DTARGET_GPU -DBLOCK_SIZE=$(BLOCKSIZE) -DOPENMP -c $(OMPFOLDER)lib_omp_opt.cpp -o $(OMPFOLDER)lib_omp_gpu_opt.o $(CFLAGS) $(OMPGPUFLAGS)

main_openmp_gpu_opt: main.cpp lib_omp_gpu_opt.o cpu_functions.o
	mkdir -p $(OUTPUTFOLDER) 
	$(NVC) -D$(DATATYPE) -DOPENMP main.cpp  $(OMPFOLDER)lib_omp_gpu_opt.o  $(CPUFUNCTIONFOLDER)cpu_functions.o -o $(OUTPUTFOLDER)$(TARGET)_omp_gpu_opt_$(shell echo $(DATATYPE) | tr A-Z a-z) $(CFLAGS) $(OMPGPUFLAGS)
# End OpenMP GPU optimized

# OpenACC Part optimized
openacc-opt:  main_openacc_opt

lib_acc_opt.o: $(ACCFOLDER)lib_acc_opt.cpp 
	$(NVC) -D$(DATATYPE) -DBLOCK_SIZE=$(BLOCKSIZE) -DOPENMP -c $(ACCFOLDER)lib_acc_opt.cpp -o $(ACCFOLDER)lib_acc_opt.o $(CFLAGS) $(ACCFLAGS)

main_openacc_opt: main.cpp lib_acc_opt.o cpu_functions.o
	mkdir -p $(OUTPUTFOLDER) 
	$(NVC) -D$(DATATYPE) -DOPENMP main.cpp  $(ACCFOLDER)lib_acc_opt.o  $(CPUFUNCTIONFOLDER)cpu_functions.o -o $(OUTPUTFOLDER)$(TARGET)_acc_opt_$(shell echo $(DATATYPE) | tr A-Z a-z) $(CFLAGS) $(ACCFLAGS)
# End OpenACC optimized

# SYCL Part 
sycl-opt: main_sycl_opt

lib_sycl_opt.o: $(SYCLFOLDER)lib_sycl_opt.cpp
	$(SYCL) -D$(DATATYPE) -D$(MODEL) -D$(DEVICETARGET) -DBLOCK_SIZE=$(BLOCKSIZE) -DSYCL -c $(SYCLFOLDER)lib_sycl_opt.cpp -o $(SYCLFOLDER)lib_sycl_opt.o $(CFLAGS) $(SYCLFLAGS)

main_sycl_opt: main.cpp lib_sycl_opt.o cpu_functions.o 
	mkdir -p $(OUTPUTFOLDER)
	$(SYCL) -D$(DATATYPE) -D$(MODEL) -D$(DEVICETARGET) main.cpp $(SYCLFOLDER)lib_sycl_opt.o $(CPUFUNCTIONFOLDER)cpu_functions.o -o $(OUTPUTFOLDER)$(TARGET)_sycl_opt_$(shell echo $(DATATYPE) | tr A-Z a-z) $(CFLAGS) $(SYCLFLAGS)
# End SYCL

# Hip part
hip-opt: main_hip_opt

lib_hip_opt.o: $(HIPFOLDER)lib_hip_opt.cpp
	$(HIP) -D$(DATATYPE) -DBLOCK_SIZE=$(BLOCKSIZE) -DHIP -c $(HIPFOLDER)lib_hip_opt.cpp -o $(HIPFOLDER)lib_hip_opt.o $(CFLAGS) $(HIPFLAGS)

main_hip_opt: main.cpp lib_hip_opt.o cpu_functions.o
	mkdir -p $(OUTPUTFOLDER) 
	$(HIP) -D$(DATATYPE) -DHIP main.cpp -x none $(HIPFOLDER)lib_hip_opt.o  $(CPUFUNCTIONFOLDER)cpu_functions.o -o $(OUTPUTFOLDER)$(TARGET)_hip_opt_$(shell echo $(DATATYPE) | tr A-Z a-z)_$(BLOCKSIZESQUARED) $(CFLAGS) $(HIPFLAGS)
# End Hip 

# CUDA part library
.PHONY: cuda
cuda-lib: main_cuda_lib

lib_cuda_lib.o: $(CUFOLDER)lib_cuda_lib.cu
	$(NVCC) -DCUDA -D$(DATATYPE) -c $(CUFOLDER)lib_cuda_lib.cu -o $(CUFOLDER)lib_cuda_lib.o $(NVCCFLAGS) 


main_cuda_lib: main.cpp lib_cuda_lib.o cpu_functions.o
	mkdir -p $(OUTPUTFOLDER)
	$(CC) -D$(DATATYPE) -DCUDA main.cpp $(CUFOLDER)lib_cuda_lib.o $(CPUFUNCTIONFOLDER)cpu_functions.o -o $(OUTPUTFOLDER)$(TARGET)_cuda_lib_$(shell echo $(DATATYPE) | tr A-Z a-z) $(CUFLAGS) $(CFLAGS) -lstdc++ -lcudnn

# End CUDA library

# OpenCL Part library
opencl-lib:  main_opencl_lib

lib_opencl_lib.o: $(OPFOLDER)lib_opencl_lib.cpp 
	$(CC) -D$(DATATYPE) -DOPENCL -c $(OPFOLDER)lib_opencl_lib.cpp -o $(OPFOLDER)lib_opencl_lib.o $(CFLAGS) $(OPFLAGS) 

main_opencl_lib: main.cpp lib_opencl_lib.o cpu_functions.o
	mkdir -p $(OUTPUTFOLDER) 
	$(CC) -D$(DATATYPE) -DOPENCL main.cpp $(OPFOLDER)lib_opencl_lib.o  $(CPUFUNCTIONFOLDER)cpu_functions.o -o $(OUTPUTFOLDER)$(TARGET)_opencl_lib_$(shell echo $(DATATYPE) | tr A-Z a-z) $(CFLAGS) $(OPFLAGS)  

# End OpenCL library

# Clean
.PHONY: clean
clean:
	rm -rf *.o
	rm -rf $(CPUFOLDER)*.o
	rm -rf $(CPUFUNCTIONFOLDER)*.o
	rm -rf $(OPFOLDER)*.o
	rm -rf $(OMPFOLDER)*.o
	rm -rf $(ACCFOLDER)*.o
	rm -rf $(SYCLFOLDER)*.o
	rm -rf $(HIPFOLDER)*.o
	rm -rf $(CUFOLDER)*.o
	rm -rf $(OUTPUTFOLDER)$(TARGET)_*