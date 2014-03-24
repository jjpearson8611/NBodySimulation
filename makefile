all: 
	gcc -fopenmp main.c -c
	nvcc -c cuda.cu
	nvcc -o body.exe cuda.o main.o

clean:
	rm *.o

seq:
	gcc -fopenmp main.c
	clear
	./a.out	
	
