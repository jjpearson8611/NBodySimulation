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

//Prototypes
void MoveABody(int BodySpotInArray);
int TwoBodyCompare(int FirstBody, int SecondBody);
void FillArrayFromFile();
void WriteOutputToFile();
float GetDistance(float XOne, float YOne, float XTwo, float YTwo);

//main code
int main(int argc, char * argv[])
{

	
	//once we get the file reading part done change 4 to the first line of the file

	FillArrayFromFile();	

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
			MagDist = GetMag(XCoord[i], XCoord[j], YCoord[i], YCoord[j]);
			Ri = GetR(XCoord[i], XCoord[j]);
			Rj = GetR(YCoord[i], YCoord[j]);

			ForceI = (G * ((topHalf / (MagDist * MagDist)) * (Ri / MagDist)));
			ForceJ = (G * ((topHalf / (MagDist * MagDist)) * (Rj / MagDist)));
		}
	}
}
float GetR(float CoordOne, float CoordTwo)
	return (CoordTwo - CoordONe);
}


float GetMag(float XOne, float YOne, float XTwo, float YTwo)
{
	float XDiffSqrd = (XTwo - XOne) * (XTwo - XOne);
	float YDiffSqrd = (YTwo - YOne) * (YTwo - YOne);
	return sqrt(XDiffSqrd + YDiffSqrd);
}
int TwoBodyCompare(int FirstBody, int SecondBody)
{



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
		XCoords =  (float *) calloc(numberOfBodies, sizeof(float));
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
	//FILE * ofp;
	//ofp = fopen("output.txt", "a+");
	int i;
	for(i = 0; i < numberOfBodies * 5; i+=6)
	{
		printf("XCoord %f, YCoord %f, XVel %f, YVel %f, Mass %f, Acceleration %f\n", 
		XCoords[i], YCoords[i],XVels[i],
		YVels[i],Masses[i], Accels[i]);
	}
	//fclose(ofp);
}
