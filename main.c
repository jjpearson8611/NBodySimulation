#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <string.h>
#include <math.h>

//Data is stored in the bodyinfo array as
// XCoord, YCoord, XVel, YVel, BodyMass, acceleration
float * XCoords = 0;
float * YCoords = 0;
float * XVels = 0;
float * YVels = 0;
float * Masses = 0;
float * Accels = 0;

//holds the total number of bodies
int numberOfBodies;
float G = .000000000667384f;

//if zero then output to file else output to standard out
#define outputtype 0

//inputtype = 0 then from file else random
#define inputtype 1

//defines the number of random bodies
#define NumRandBodies 4

//Prototypes
void MoveABody(int BodySpotInArray);
int TwoBodyCompare(int FirstBody, int SecondBody);
void FillArrayFromFile();
void WriteOutputToFile();
float GetDistance(float XOne, float YOne, float XTwo, float YTwo);
void FillArrayRandomly();

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
	for(i = 0; i < numberOfBodies; i++)
	{
		MoveABody(i);
	}

	WriteOutputToFile();

	return 0;
}
//XCoord, YCoord, XVel, YVel, Mass
void MoveABody(int i)
{
	//in this method we will compare each body with all the other bodies
	int j;
	float topHalf;
	float MagDist;
	float Ri;
	float Rj;
	float ForceI;
	float ForceJ;

	for(j = 0; j < numberOfBodies; j++)
	{
		if(i != j)
		{
			topHalf = Masses[i] * Masses[j];
			MagDist = GetMag(XCoords[i], XCoords[j], YCoords[i], YCoords[j]);
			Ri = GetR(XCoords[i], XCoord[j]);
			Rj = GetR(YCoord[i], YCoord[j]);

			ForceI = (G * ((topHalf / (MagDist * MagDist)) * (Ri / MagDist)));
			ForceJ = (G * ((topHalf / (MagDist * MagDist)) * (Rj / MagDist)));
		}
	}
}
float GetR(float CoordOne, float CoordTwo)
{
	return (CoordTwo - CoordOne);
}


float GetMag(float XOne, float YOne, float XTwo, float YTwo)
{
	float XDiffSqrd = (XTwo - XOne) * (XTwo - XOne);
	float YDiffSqrd = (YTwo - YOne) * (YTwo - YOne);
	return sqrt(XDiffSqrd + YDiffSqrd);
}
int TwoBodyCompare(int FirstBody, int SecondBody)
{

return 0;

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
			Accels  = (float *) malloc(numberOfBodies * sizeof(float));
		}
		else
		{
			splitLine = strtok(inputLine, ",");
			XCoords[NextOpenSpot] = (float) atoi(&splitLine[0]);
			YCoords[NextOpenSpot] = (float) atoi(&splitLine[2]);
			XVels[NextOpenSpot] = (float) atoi(&splitLine[4]);
			YVels[NextOpenSpot] = (float) atoi(&splitLine[6]);
			Masses[NextOpenSpot] = (float) atoi(&splitLine[8]);
			printf("XCoord %f, YCoord %f, XVel %f, YVel %f, Mass %f\n", XCoords[NextOpenSpot], YCoords[NextOpenSpot],XVels[NextOpenSpot],YVels[NextOpenSpot],Masses[NextOpenSpot]);
			Accels[NextOpenSpot] = 0;
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
	for(i = 0; i < numberOfBodies; i++)
	{
		#if outputtype == 0
		fprintf(ofp, "XCoord %f, YCoord %f, XVel %f, YVel %f, Mass %f, Acceleration %f\n",
		XCoords[i], YCoords[i],XVels[i],
		YVels[i],Masses[i], Accels[i]);
		#else
		printf("XCoord %f, YCoord %f, XVel %f, YVel %f, Mass %f, Acceleration %f\n",
		XCoords[i], YCoords[i],XVels[i],
		YVels[i],Masses[i], Accels[i]);
		#endif
	}
	fclose(ofp);
}

void FillArrayRandomly()
{
	srand(time(NULL));
	float r;
	int i;	

	XCoords = (float *) malloc(numberOfBodies * sizeof(float));
	YCoords = (float *) malloc(numberOfBodies * sizeof(float));
	XVels   = (float *) malloc(numberOfBodies * sizeof(float));
	YVels   = (float *) malloc(numberOfBodies * sizeof(float));
	Masses  = (float *) malloc(numberOfBodies * sizeof(float));
	Accels  = (float *) malloc(numberOfBodies * sizeof(float));
	
	for(i = 0; i < numberOfBodies; i++)
	{
		XCoords[i] = (float) rand();
		YCoords[i] = (float) rand();
		XVels[i] = (float) rand();
		YVels[i] = (float) rand();
		Masses[i] = (float) rand();
		Accels[i] = 0;
	}
}
