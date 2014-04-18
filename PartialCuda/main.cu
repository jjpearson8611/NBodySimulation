#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <sys/time.h>
#include "book.h"

//Data is stored in the bodyinfo array as
// XCoord, YCoord, XVel, YVel, BodyMass, accelX, accelY
float * XCoords = 0;
float * YCoords = 0;
float * XVels = 0;
float * YVels = 0;
float * Masses = 0;
float * AccelsX = 0;
float * AccelsY = 0;

struct timeval start;
struct timeval end;

//holds the total number of bodies
int numberOfBodies;
const float G = .000000000667384f;

int numberofsteps;

//defines if we should print debugging statements 0 means yes
#define debug 1

//defines the step size
#define stepsize 10

//Defines the softening factor
#define elipson .0001

//if zero then output to file else output to standard out
#define outputtype 2

//inputtype = 0 then from file else random
#define inputtype 3

//defines the number of random bodies
#define NumRandBodies 8

//Prototypes
void MoveABody(int BodySpotInArray);
void WriteOutputToFile();
float GetR(float CoordOne, float CoordTwo);
float GetDistance(float XOne, float YOne, float XTwo, float YTwo);
float GetMag(float XOne, float XTwo);

///for now this gives a big planet in the middle and plants in a line going out from it
void FillArrayRandomly();
void UpdateBodies(int i);

__global__ void compute(float * XCoords, float * YCoords, float * XVels, float * YVels,
	float * Masses, float * AccelsX, float * AccelsY)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	
	//here we will be computing the things
	int j;
	float Ri;
	float Rj;
	float TotalX = 0;
	float TotalY = 0;
	float E = elipson * elipson;
	float NormI;
	float NormJ;

	for(j = 0; j < 8; j++)
	{	

		//get rij
		Ri = XCoords[j] - XCoords[tid];
		Rj = YCoords[j] - YCoords[tid];
		
		NormJ = sqrt((YCoords[j] - YCoords[tid]) * (YCoords[j] - YCoords[tid]));
		NormJ = NormJ * NormJ + E;
		NormJ = sqrt(NormJ * NormJ * NormJ);

		NormI = sqrt((XCoords[j] - XCoords[tid]) * (XCoords[j] - XCoords[tid]));
		NormI = NormI * NormI + E;
		NormI = sqrt(NormI * NormI * NormI);


		//final calculation
		//printf("Calulated X = %f", ((Masses[j] * Ri) / Norm));

		TotalX += (Masses[j] * Ri) / NormI;
		TotalY += (Masses[j] * Rj) / NormJ;

		//printf("TotalX = %f, TotalY = %f\n", TotalX, TotalY);
	}
	AccelsX[tid] = TotalX * G;
	AccelsY[tid] = TotalY * G;



	__syncthreads();

	//here we will be updating things

	//st = s0 + v0t + 1/2a0t^2
	XCoords[tid] = (XCoords[tid] + (XVels[tid] * stepsize) 
		+ (.5 * AccelsX[tid] * (stepsize * stepsize)));
	YCoords[tid] = (YCoords[tid] + (YVels[tid] * stepsize) 
		+ (.5 * AccelsY[tid] * (stepsize * stepsize)));

	//v = v0 + a0t
	XVels[tid] = XVels[tid] + (AccelsX[tid] * stepsize);
	YVels[tid] = YVels[tid] + (AccelsY[tid] * stepsize);
}
//main code
int main(int argc, char * argv[])
{

	
	//once we get the file reading part done change 4 to the first line of the file
	numberOfBodies = NumRandBodies;
	FillArrayRandomly();
	int CompletedSteps;

	float *devXCoords, *devYCoords, *devXVels, 
		*devYVels, *devMasses, *devAccelsX, *devAccelsY;

	numberofsteps = atoi(argv[1]);

	HANDLE_ERROR(cudaMalloc( (void**)&devXCoords, numberOfBodies * sizeof(float)));
	HANDLE_ERROR(cudaMalloc( (void**)&devYCoords, numberOfBodies * sizeof(float)));
	HANDLE_ERROR(cudaMalloc( (void**)&devXVels, numberOfBodies * sizeof(float)));
	HANDLE_ERROR(cudaMalloc( (void**)&devYVels, numberOfBodies * sizeof(float)));
	HANDLE_ERROR(cudaMalloc( (void**)&devMasses, numberOfBodies * sizeof(float)));
	HANDLE_ERROR(cudaMalloc( (void**)&devAccelsX, numberOfBodies * sizeof(float)));
	HANDLE_ERROR(cudaMalloc( (void**)&devAccelsY, numberOfBodies * sizeof(float)));

	float holder = numberOfBodies * sizeof(float);
	


	gettimeofday(&start, NULL);
	//another for loop here
	for(CompletedSteps = 0; CompletedSteps < numberofsteps; CompletedSteps++)
	{
		//we may need to copy data to the card here idk yet
		HANDLE_ERROR(cudaMemcpy( devXCoords, XCoords,holder , cudaMemcpyHostToDevice));
		HANDLE_ERROR(cudaMemcpy( devYCoords, YCoords,holder , cudaMemcpyHostToDevice));
		HANDLE_ERROR(cudaMemcpy( devXVels,   XVels,holder , cudaMemcpyHostToDevice));
		HANDLE_ERROR(cudaMemcpy( devYVels,   YVels,holder , cudaMemcpyHostToDevice));
		HANDLE_ERROR(cudaMemcpy( devMasses,  Masses,holder , cudaMemcpyHostToDevice));
		HANDLE_ERROR(cudaMemcpy( devAccelsX, AccelsX,holder , cudaMemcpyHostToDevice));
		HANDLE_ERROR(cudaMemcpy( devAccelsY, AccelsY,holder , cudaMemcpyHostToDevice));
			

		//call the kernel
		compute<<<numberOfBodies,numberOfBodies>>>(devXCoords, devYCoords, devXVels, devYVels, devMasses, devAccelsX, devAccelsY);

		//we may need to update variables again here idk yet
		HANDLE_ERROR(cudaMemcpy( XCoords, devXCoords,holder , cudaMemcpyDeviceToHost));
		HANDLE_ERROR(cudaMemcpy( YCoords, devYCoords,holder , cudaMemcpyDeviceToHost));
		HANDLE_ERROR(cudaMemcpy( XVels,   devXVels,  holder , cudaMemcpyDeviceToHost));
		HANDLE_ERROR(cudaMemcpy( YVels,   devYVels,  holder , cudaMemcpyDeviceToHost));
		HANDLE_ERROR(cudaMemcpy( Masses,  devMasses, holder , cudaMemcpyDeviceToHost));
		HANDLE_ERROR(cudaMemcpy( AccelsX, devAccelsX,holder , cudaMemcpyDeviceToHost));
		HANDLE_ERROR(cudaMemcpy( AccelsY, devAccelsY,holder , cudaMemcpyDeviceToHost));


	}	

	gettimeofday(&end, NULL);
	WriteOutputToFile();

 int timeran = (((end.tv_sec - start.tv_sec) * 1000000) +(end.tv_usec - start.tv_usec));

	printf("Time Ran Nano Seconds %d\n", timeran);

	HANDLE_ERROR(cudaFree( devXCoords));
	HANDLE_ERROR(cudaFree( devYCoords));
	HANDLE_ERROR(cudaFree( devXVels));
	HANDLE_ERROR(cudaFree( devYVels));
	HANDLE_ERROR(cudaFree( devMasses));
	HANDLE_ERROR(cudaFree( devAccelsX));
	HANDLE_ERROR(cudaFree( devAccelsY));
	

	return 0;
}

void WriteOutputToFile()
{
	FILE * ofp;
	ofp = fopen("output.txt", "w");
	int i;
	printf("\n\n");
	for(i = 0; i < numberOfBodies; i++)
	{
		#if outputtype == 0
		fprintf(ofp, "XCoord %f, YCoord %f, XVel %f, YVel %f, Mass %f, AccelX %f, AccelY %f\n",
		XCoords[i], YCoords[i],XVels[i],
		YVels[i],Masses[i], AccelsX[i], AccelsY[i]);
		#else
		printf("XCoord %f, YCoord %f, XVel %f, YVel %f, Mass %f, AccelX %e, AccelY %e\n",
		XCoords[i], YCoords[i],XVels[i],
		YVels[i],Masses[i], AccelsX[i], AccelsY[i]);
		#endif
	}
	fclose(ofp);
}

void FillArrayRandomly()
{
	srand(15);
	int i;	

	XCoords = (float *) malloc(numberOfBodies * sizeof(float));
	YCoords = (float *) malloc(numberOfBodies * sizeof(float));
	XVels   = (float *) malloc(numberOfBodies * sizeof(float));
	YVels   = (float *) malloc(numberOfBodies * sizeof(float));
	Masses  = (float *) malloc(numberOfBodies * sizeof(float));
	AccelsY  = (float *) malloc(numberOfBodies * sizeof(float));
	AccelsX  = (float *) malloc(numberOfBodies * sizeof(float));
	
	for(i = 0; i < numberOfBodies; i++)
	{
		Masses[i] = 1 + 10 * i;
		XCoords[i] = 0 + .1 * i;
		YCoords[i] = 0 + .1 * i;
		XVels[i] = -.1 * i;	
		YVels[i] = .1 * i;
		if(i == 0)
		{
			Masses[i] = 100000000;
		}
		AccelsY[i] = 0;
		AccelsX[i] = 0;
//		printf("XCoord %f, YCoord %f, XVel %f, YVel %f, Mass %f\n", XCoords[i], YCoords[i],XVels[i],YVels[i],Masses[i]);
	}
}
