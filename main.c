#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

//Data is stored in the bodyinfo array as
// XCoord, YCoord, XVel, YVel, BodyMass
float * BodyInfo = 0;

//holds the total number of bodies
int numberOfBodies;


//Prototypes
void MoveABody(int BodySpotInArray);


//main code
int main(int argc, char * argv[])
{
	//once we get the file reading part done change 4 to the first line of the file
	BodyInfo = (float *) malloc(5 * 4 * sizeof(float));

	numberOfBodies = 4;
	
	int i;
	for(i = 0; i < numberOfBodies; i++)
	{
		MoveABody(i);
	}

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
