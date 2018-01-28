CC=gcc
NVCC=nvcc
CFLAGS=-O2 -Wall 

c:
	$(CC) $(CFLAGS) -o diffusion.out diffusion.c
cuda:
	$(NVCC) $(CFLAGS) -o diffusion_cu.out diffusion.cu
clean:
	rm -f *.out
