#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

//cc -fPIC -shared -o lib_mine.so test2.c -lm -fopenmp // To compile this file

void num2binary(int *bin,int n,int len){
	for(int i=0; i<len;i++){
		bin[len-i-1] = (int)(n%2);
		n = (int)(n/2);
	}
}

int binary2decimal(int *n, int len_n){
	int out =0;
	for (int i = 0; i < len_n; i++){
		out = out + n[i]*pow(2,len_n-i-1);
	}
	return out;
}


double ieee_754(int *n){
	int n_expo;
	float out_expo;
	float mantissa_1=0.0;
		float mantissa;
	float out_mantissa;
	int or_input=0;
	float out_number;

	n_expo = binary2decimal(&n[1],8)-127;
	//printf("ieee_ : %d\n",n_expo);
	out_expo = pow(2.0,n_expo);
	
	for(int i=0;i<23;i++){
		mantissa_1 = n[9+i]*pow(2,-1*i-1)+mantissa_1;
	}
	
	out_mantissa = 1.0 + mantissa_1;
	
	for(int i=0;i<32;i++){
		or_input = or_input+ n[i];
	}

	if(or_input==0){
		out_expo = 126;
		out_mantissa=0;
	}
	
	if(n[0]==1){
		out_number = -1*out_mantissa * out_expo;
	}
	else if(n[0]==0){
		out_number = out_mantissa * out_expo;
	}
	
	return out_number;
		
}

void ieee_754_0(int *output,float n){

	int out_expo = 0;
	float n_abs=0;
	float bin_mantissa =0;

	if(n!=0){
		if(n>0){
			output[0]=0;
		}
		else{
			output[0]=1;
		}
		out_expo=0;

		if(n<0){
			n_abs = -1*n;
		}
		else{
			n_abs =n;
		}

		if (n_abs>=1){
			while(1){
				if(n_abs<2 && n_abs>=1){
					break;
				}
				else{
					n_abs = n_abs/2.0;
					out_expo = out_expo + 1;
				}
			}
		}
		else if (n_abs<1 && n_abs >0){
			while(1){
				if (n_abs<2 && n_abs>=1){
					break;
				}
				else{
					n_abs = n_abs*2.0;
					out_expo = out_expo - 1;
				}
			}
		}
			
		bin_mantissa = n_abs - 1;

		for(int i=0;i<23;i++){
			bin_mantissa = bin_mantissa * 2.0;
			if (bin_mantissa >= 1.0){
				output[i+9] = 1;
				bin_mantissa = bin_mantissa - 1;
			}
			else{
				output[i+9] = 0;
			}

		}

		out_expo = out_expo + 127;
		if(out_expo<=1){
			out_expo=0;
			for(int i=9;i<32;i++){
				output[i]=0;
			}
		}

		num2binary(&output[1],out_expo,8);

	}
}

// We can replace another approximate multiplier here////
float cfpm(float a,float b,int tune)
{
	return a*b;
}

void convol(float *out, float *weight,float *kernel_shape,float *x,float *x_shape,int outplanes,int h_out,int w_out,int stride,int kernel_size, float *bias)
{

	// Take weight as input of size outplanes x (ker_size) i.e. 2 dimensions
	int width = (int)(kernel_size/2);
		
	int ker_size = (int)(kernel_shape[0]*kernel_shape[1]*kernel_shape[2]);
	int x_shape_0 = (int)x_shape[0];
	int x_shape_1 = (int)x_shape[1];
	int x_shape_2 = (int)x_shape[2];
	int x_shape_3 = (int)x_shape[3];
	int outplane;


	int num_threads = 8;
	omp_set_num_threads(num_threads);
	int chunk = outplanes/num_threads;
	float ker[ker_size];
	float xx[ker_size];
	int count =0;
	int r_o=0;
	int c_o=0;
	float dot_pro = 0;
	
	#pragma omp parallel shared(ker_size,weight,outplanes,width,x_shape_1,x_shape_2,x_shape_3,x_shape_0,h_out,w_out,x,out,stride, bias) private(outplane,ker,xx,count,r_o,c_o,dot_pro)
	{	
		#pragma omp for schedule(static) nowait
		for(outplane=0;outplane<outplanes;outplane++){
			count=0;
			//printf("%d\n",ker_size);
			//for(int i =0;i<ker_size;i++){			
			//	ker[i]=*((weight+ outplane*ker_size)+i);
			//}

			for(int r=width;r<x_shape_1-width;r = r+stride){
				for(int c=width;c<x_shape_2-width;c = c+stride){
					count=0;
					dot_pro = 0;
					for(int inplane =0;inplane<x_shape_0;inplane++){
						for(int rr =r-width;rr<r+width+1;rr++){
							for(int cc =c-width;cc<c+width+1;cc++){
								//xx[count]=*(((x+inplane*x_shape_1*x_shape_2)+rr*x_shape_2)+cc);
								dot_pro = dot_pro+ cfpm(*((weight+ outplane*ker_size)+count),*(((x+inplane*x_shape_1*x_shape_2)+rr*x_shape_2)+cc),5);
								//dot_pro = dot_pro+ *((weight+ outplane*ker_size)+count)*(*(((x+inplane*x_shape_1*x_shape_2)+rr*x_shape_2)+cc));
								count = count+1;
							}
						}
					}
					r_o= (int)((r-width)/stride);
					c_o= (int)((c-width)/stride);
					//*(((out+outplane*h_out*w_out)+r_o*w_out)+c_o) = my_dotPro(&ker[0],&xx[0],ker_size,5);
					*(((out+outplane*h_out*w_out)+r_o*w_out)+c_o) = dot_pro + *(bias+outplane);
				}
			}
	
		}
	}

}

