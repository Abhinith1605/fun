NVCC = nvcc
SRC = src/main.cu src/blur.cu src/sobel.cu src/utils.cu
INC = -Isrc
FLAGS = -O3 --std=c++17

all: app

app:
	$(NVCC) $(SRC) $(INC) $(FLAGS) -o image_gpu.exe

clean:
	rm -f image_gpu.exe
	rm -rf output/*
