#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <sys/time.h>

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
float G = .000000000667384f;

//defines the number of steps
int numberofsteps;

//defines if we should print debugging statements 0 means yes
#define debug 2

//defines the step size
#define stepsize 10

//Defines the softening factor
#define elipson .0001

//if zero then output to file else output to standard out
#define outputtype 0

//inputtype = 0 then from file else random
#define inputtype 3

//defines the number of random bodies
#define NumRandBodies 8

//Prototypes
void MoveABody(int BodySpotInArray);
void FillArrayFromFile();
void WriteOutputToFile();
float GetR(float CoordOne, float CoordTwo);
float GetDistance(float XOne, float YOne, float XTwo, float YTwo);
float GetMag(float XOne, float XTwo);

///for now this gives a big planet in the middle and plants in a line going out from it
void FillArrayRandomly();
void UpdateBodies(int i);

//main code
int main(int argc, char * argv[])
{

	
	//once we get the file reading part done change 4 to the first line of the file
	#if inputtype == 0
	FillArrayFromFile();	
	#else
	numberOfBodies = NumRandBodies;
	FillArrayRandomly();
	#endif
	int i;
	int CompletedSteps;

	numberofsteps = atoi(argv[1]);


	gettimeofday(&start, NULL);
	//another for loop here
	for(CompletedSteps = 0; CompletedSteps < numberofsteps; CompletedSteps++)
	{
		for(i = 0; i < numberOfBodies; i++)
		{
			MoveABody(i);
		}
		for(i = 0; i < numberOfBodies; i++)
		{
			UpdateBodies(i);
		}
		#if debug == 0
		//printf("\n");
		#endif
	}	
	gettimeofday(&end,NULL);
	WriteOutputToFile();

 	int timeran = (((end.tv_sec - start.tv_sec) * 1000000) +(end.tv_usec - start.tv_usec));

	printf("Time Ran Nano Seconds = %d\n", timeran);
	return 0;
}
void UpdateBodies(int i)
{
	//st = s0 + v0t + 1/2a0t^2
	XCoords[i] = (XCoords[i] + (XVels[i] * stepsize) + (.5 * AccelsX[i] * pow(stepsize, 2)));
	YCoords[i] = (YCoords[i] + (YVels[i] * stepsize) + (.5 * AccelsY[i] * pow(stepsize, 2)));

	//v = v0 + a0t
	XVels[i] = XVels[i] + (AccelsX[i] * stepsize);
	YVels[i] = YVels[i] + (AccelsY[i] * stepsize);
	#if debug == 0
	printf("XCoord %e, YCoord %e, XVel %e, YVel %e, Mass %d, AccelX %e, AccelY %e\n", XCoords[i], YCoords[i],XVels[i],YVels[i],(int)Masses[i], AccelsX[i], AccelsY[i]);
	#endif
}
//XCoord, YCoord, XVel, YVel, Mass
void MoveABody(int i)
{
	//in this method we will compare each body with all the other bodies
	int j;
	float topHalf;
	float Ri;
	float Rj;
	float TotalX = 0;
	float TotalY = 0;
	float E = elipson * elipson;
	float NormI;
	float NormJ;

	for(j = 0; j < numberOfBodies; j++)
	{	

		//get rij
		Ri = GetR(XCoords[i], XCoords[j]);
		Rj = GetR(YCoords[i], YCoords[j]);
		
		//determine bottom of equation
		NormJ = GetMag(YCoords[i], YCoords[j]);
		NormJ = NormJ * NormJ + E;
		NormJ = pow(NormJ, 1.5);

		NormI = GetMag(XCoords[i], XCoords[j]);
		NormI = NormI * NormI + E;
		NormI = pow(NormI, 1.5);

		//printf("Mass = %f, Ri = %f, Norm = %f\n", Masses[j], Ri, Norm);

		//final calculation
		//printf("Calulated X = %f", ((Masses[j] * Ri) / Norm));

		TotalX += (Masses[j] * Ri) / NormI;
		TotalY += (Masses[j] * Rj) / NormJ;

		//printf("TotalX = %f, TotalY = %f\n", TotalX, TotalY);
	}
	AccelsX[i] = TotalX * G;
	AccelsY[i] = TotalY * G;
}
float GetR(float CoordOne, float CoordTwo)
{
	return (float)(CoordTwo - CoordOne);
}


float GetMag(float XOne, float XTwo)
{
	float XDiffSqrd = pow((XTwo - XOne), 2);
	return (float)sqrt(XDiffSqrd);
}

void FillArrayFromFile()
{
	FILE * ifp;
	ssize_t read;
	size_t len = 0;

	ifp = fopen("Dataset","r");

	char * inputLine = NULL;
	char * splitLine;

	int FirstRead = 1;
	int NextOpenSpot = 0;
	
	while((read = getline(&inputLine, &len, ifp)) != -1)
	{
		if(FirstRead == 1)
		{
			FirstRead = 0;
			numberOfBodies = atoi(inputLine);
			XCoords = (float *) malloc(numberOfBodies * sizeof(float));
			YCoords = (float *) malloc(numberOfBodies * sizeof(float));
			XVels   = (float *) malloc(numberOfBodies * sizeof(float));
			YVels   = (float *) malloc(numberOfBodies * sizeof(float));
			Masses  = (float *) malloc(numberOfBodies * sizeof(float));
			AccelsX  = (float *) malloc(numberOfBodies * sizeof(float));
			AccelsY  = (float *) malloc(numberOfBodies * sizeof(float));
		}
		else
		{
			splitLine = strtok(inputLine, ",");
			XCoords[NextOpenSpot] = (float) atoi(&splitLine[0]);
			YCoords[NextOpenSpot] = (float) atoi(&splitLine[2]);
			XVels[NextOpenSpot] = (float) atoi(&splitLine[4]);
			YVels[NextOpenSpot] = (float) atoi(&splitLine[6]);
			Masses[NextOpenSpot] = (float) atoi(&splitLine[8]);
//			printf("XCoord %f, YCoord %f, XVel %f, YVel %f, Mass %f\n", XCoords[NextOpenSpot], YCoords[NextOpenSpot],XVels[NextOpenSpot],YVels[NextOpenSpot],Masses[NextOpenSpot]);
			AccelsY[NextOpenSpot] = 0;
			AccelsX[NextOpenSpot] = 0;
			NextOpenSpot++;
		}
	}
	fclose(ifp);
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
	float r;
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
