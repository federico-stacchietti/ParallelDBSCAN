################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../main.cu 

CPP_SRCS += \
../host_DBSCAN.cpp 

OBJS += \
./host_DBSCAN.o \
./main.o 

CU_DEPS += \
./main.d 

CPP_DEPS += \
./host_DBSCAN.d 


# Each subdirectory must supply rules for building sources it contributes
%.o: ../%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/bin/nvcc -G -g -O0 -gencode arch=compute_61,code=sm_61  -odir "." -M -o "$(@:%.o=%.d)" "$<"
	/usr/bin/nvcc -G -g -O0 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

%.o: ../%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/bin/nvcc -G -g -O0 -gencode arch=compute_61,code=sm_61  -odir "." -M -o "$(@:%.o=%.d)" "$<"
	/usr/bin/nvcc -G -g -O0 --compile --relocatable-device-code=false -gencode arch=compute_61,code=compute_61 -gencode arch=compute_61,code=sm_61  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


