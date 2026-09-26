/*  solver_mult_inverse.cu

    Tim Behrens, Saad Jbabdi, Stam Sotiropoulos, Moises Hernandez  - FMRIB Image Analysis Group

    Copyright (C) 2005 University of Oxford  */

/*  CCOPYRIGHT  */

#include "options.h"

//X = A.i() * B . Used in Levenberg-Marquardt
//MATRIX INVERSE AS NEWMAT LU SOLVER
//implemented in NEWMAT:newmat7.cpp GeneralSolvI.
__device__ void solver(	//INPUT
			float *A, 
			float *P,
			int length,
			//TO USE
			float *C,
			float *el,
			int *indx,	
			//OUTPUT
			float *B)  
{  
	//double C[NPARAMS*NPARAMS];

	for(int i=0;i<length;i++){
		for(int j=0;j<length;j++){
			C[i*length+j]=A[i*length+j];
		}
	}
	
 	bool d=true; 
  	//int indx[NPARAMS];

   	float* akk = C;   
	float big = fabs(*akk); 
	int mu = 0; 
	float* ai = akk; 
	int k;

	for (k = 1; k<length; k++){
      		ai += length; 
		const float trybig = fabs(*ai);
      		if (big < trybig){ 
			big = trybig; 
			mu = k; 
		}
   	}

   	if(length) for (k = 0;;){

		indx[k] = mu;
		if (mu != k){
         		float* a1 = C + length*k; 
			float* a2 = C + length*mu; 
			d = !d;
         		int j = length;
         		while (j--){ 
				const float temp = *a1; 
				*a1++ = *a2; 
				*a2++ = temp; 
			}
      		}

      		float diag = *akk; 
		big = 0; 
		mu = k + 1;
      		if (diag != 0){
         		ai = akk; 
			int i = length - k - 1;
         		while (i--){
            			ai += length; 
				float* al = ai; 
				float mult = *al / diag; 
				*al = mult;
            			int l = length - k - 1; 
				float* aj = akk;
				if (l-- != 0){
				
					float aux=al[1]-(mult* *(++aj));
					*(++al) = aux;
					//*(++al) = __dadd_rn (*al,-mult* *(++aj)); //FAIL in cuda 4.2 compiler
					
               				const float trybig = fabs(*al);
               				if (big < trybig){ 
						big = trybig; 
						mu = length - i - 1; 
					}
               				while (l--){ 
						float aux= al[1]-(mult* *(++aj));
						*(++al) = aux;
						//*(++al) = __dadd_rn (*al,-mult* *(++aj)); //FAIL in cuda 4.2 compiler
					}
           			 }
         		}
      		}
      		if (++k == length) break;      
      		akk += length + 1;
   	}


//////////////////////////////

	//double el[NPARAMS];

	for(int e=0;e<length;e++){
		el[e]=P[e];		
    	}
		
   	int j;
	int ii = length; 
	int ip;    
	float temp;
	int i;
     
	for (i=0; i<length; i++){
 		ip = indx[i]; 
		temp = el[ip]; 
		el[ip] = el[i];
		el[i] = temp;
      		if (temp != 0.0) { ii = i; break; }
   	}
	
  	float* bi; 
	float* ai2;
   	i = ii + 1;

  	if (i < length){
      		bi = el + ii; 
		ai2 = C + ii + i * length;
      		for (;;){
         		int ip = indx[i]; 
			float sum = el[ip]; 
			el[ip] = el[i];
         		float* aij = ai2; 
			float* bj = bi; 
			j = i - ii;
         		while (j--){ 
				sum -=  *aij++* *bj++; 
			}
         		el[i] = sum;
         		if (++i == length) break;
         		ai2 += length;
      		}
   	}

   	ai2 = C + length*length;

   	for (i = length - 1; i >= 0; i--){
      		float* bj = el+i; 
		ai2 -= length; 
		float* ajx = ai2+i;
      		float sum = *bj; 
		float diag = *ajx;
      		j = length - i; 
		while(--j){ 
			sum -= *(++ajx)* *(++bj);  
		}
      		el[i] = sum / diag;
			
   	}
	for(int e=0;e<length;e++){
		B[e]=el[e];
    	}
}

