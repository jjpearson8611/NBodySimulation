//This is our cuda code here
__global__ float rhat(float rone, float rtwo){
	return ((rtwo - rone) / (abs(rtwo - rone)));
}
