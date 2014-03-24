#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <string.h>

//Data is stored in the bodyinfo array as
// XCoord, YCoord, XVel, YVel, BodyMass
float * BodyInfo = 0;

//holds the total number of bodies
int numberOfBodies;


//Prototypes
void MoveABody(int BodySpotInArray);
int TwoBodyCompare(int FirstBody, int SecondBody);
void FillArrayFromFile();
void WriteOutputToFile();

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

void MoveABody(int BodySpotInArray)
{
	//in this method we will compare each body with all the other bodies
	int i;
	for(i = 0; i < numberOfBodies; i++)
	{
		if(i != BodySpotInArray)
		{
			printf("comparing %i and %i\n",BodySpotInArray, i);
		}
	}
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
			BodyInfo = (float *) malloc(5 * numberOfBodies * sizeof(float));
		}
		else
		{
			splitLine = strtok(inputLine, ",");
			BodyInfo[NextOpenSpot + 0] = (double) atoi(&splitLine[0]);
			BodyInfo[NextOpenSpot + 1] = (double) atoi(&splitLine[2]);
			BodyInfo[NextOpenSpot + 2] = (double) atoi(&splitLine[4]);
			BodyInfo[NextOpenSpot + 3] = (double) atoi(&splitLine[6]);
			BodyInfo[NextOpenSpot + 4] = (double) atoi(&splitLine[8]);
			printf("XCoord %f, YCoord %f, XVel %f, YVel %f, Mass %f\n", BodyInfo[NextOpenSpot], BodyInfo[NextOpenSpot + 1],BodyInfo[NextOpenSpot + 2],BodyInfo[NextOpenSpot + 3],BodyInfo[NextOpenSpot + 4]);
			NextOpenSpot += 5;
		}
	}
	fclose(ifp);
}

void WriteOutputToFile()
{
	//FILE * ofp;
	//ofp = fopen("output.txt", "a+");
	int i;
	for(i = 0; i < numberOfBodies * 5; i+=5)
	{
		printf("XCoord %f, YCoord %f, XVel %f, YVel %f, Mass %f\n", 
		BodyInfo[i], BodyInfo[i + 1],BodyInfo[i + 2],
		BodyInfo[i + 3],BodyInfo[i + 4]);
	}
	//fclose(ofp);
}
