#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <string.h>
#include <stdint.h>
#define _OPEN_SYS_ITOA_EXT
// #include <posit.h>


//function to convert from decimal to binary number
void num2binary(int *bin, int n, int len){
    for(int i = 0; i<len; i++){
        bin[len-i-1] = (int)(n%2);
        n = (int)(n/2);
    }
}

//function to convert from 
int binary2decimal(int *n, int len_n){
    int out = 0;
    for (int i = 0; i<len_n; i++){
        out = out + n[i]*pow(2,len_n-i-1);
    }
    return out;
}

int binary2decimal_new(int *n, int start, int end){
    int out = 0;
    for (int i = end; i>= start; i--){
        out = out + n[i]*pow(2,end-i);
    }
    return out;
}

typedef union {
    float f;
    struct 
    {
        unsigned int mantissa : 23;
        unsigned int exponent : 8;
        unsigned int sign : 1;
    }raw;
}myfloat;


double ieee_754(int *n){
    int n_expo;
    float out_expo;
    float mantissa_1 = 0.0;
    float mantissa;
    float out_mantissa;
    int or_input;
    float out_number;

    n_expo = binary2decimal(&n[1],8)-127;
    out_expo = pow(2.0, n_expo);

    for(int i = 0; i<23; i++){
        mantissa_1 = n[9+i]*pow(2,-1*i-1) + mantissa_1;
    }

    out_mantissa = 1.0 + mantissa_1;

    for(int i=0; i<32; i++){
        or_input = or_input + n[i];
    }

    if(or_input == 0){
        out_expo = 126;
        out_mantissa = 0;
    }

    if(n[0] == 1){
        out_number = -1*out_mantissa * out_expo;
    } else if(n[0] == 0){
        out_number = out_mantissa*out_expo;
    }

    return out_number;
}

void ieee_754_0(int *output, float n){

    int out_expo = 0;
    float n_abs = 0;
    float bin_mantissa = 0;

    if(n!=0){

        if(n>0){
            output[0] = 0;
        } 
        else{
            output[0] = 1;
        }
        out_expo = 0;

        if(n<0){
            n_abs = -1*n;
        } 
        else{
            n_abs = n;
        }

        if(n_abs>=1){
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
        else if(n_abs<1 && n_abs>0){
            while(1){
                if(n_abs<2 && n_abs>=1){
                    break;
                }
                else{
                    n_abs = n_abs*2.0;
                    out_expo = out_expo-1;
                }
            }
        }

        bin_mantissa = n_abs-1;

        for(int i=0; i<23; i++){
            bin_mantissa = bin_mantissa *2.0;
            if(bin_mantissa >= 1.0){
                output[i+9] = 1;
                bin_mantissa = bin_mantissa -1;
            }
            else{
                output[i+9] = 0;
            }
        }

        out_expo = out_expo + 127;
        if(out_expo<=1){
            out_expo=0;
            for(int i=9; i<32; i++){
                output[i] = 0;
            }
        }

        num2binary(&output[1],out_expo,8);
    }
}


unsigned int posit_length = 6;
unsigned int es = 2;           
const int useed = 16; 
       //2^es
unsigned int regime_len = 2;   
unsigned int frc_len = 1;
//posit adder
float posit_addsub(float op1, float op2)
{
    int op1_sign,  op1_exp, op2_sign, op2_exp, op1_k,  op2_k, op1_e, op2_e, op1_regimelen,  op2_regimelen, op1_bitsremaining, op2_bitsremaining, op1_eslen, op2_eslen;
    int op1_e_remaining_max, op2_e_remaining_max, op1_frclen, op2_frclen;
    int posit_sign, posit_exp, posit_e, posit_k, posit_regime_len, posit_bitsremaining, posit_eslen, posit_e_remaining_max, posit_frclen;
    float posit_frc, op1_frc_new, op2_frc_new, frc_new, posit_f, posit_fnew;
    float op1_frc, op2_frc, op1_f, op2_f, op1_fnew, op2_fnew;
    float result;
    int kmax = posit_length - 1;
    int kmin = -1*(posit_length - 1);

    float tmp;
    if (abs((double)op1) < (double)abs(op2))
    {
        tmp = op1;
        op1 = op2;
        op2 = tmp;
    }   
    
    // if(KnobLog)
    // {
    //     fprintf(logfile, "%.14f %.14f", op1, op2);
    // }
    
    if(op1 < 0)
    {
        op1_sign = 1;
        op1 = -1*op1;
    }
    else
    {
        op1_sign = 0;
    }
    if(op2 < 0)
    {
        op2_sign = 1;
        op2 = -1*op2;
    }
    else
    {
        op2_sign = 0;
    }
    // if(op1 == 0.0)
    // {
    //     result =  0.0;
    //     if(KnobLog)
    //     {
    //         fprintf(logfile, " %.14f\n", result);
    //     }
    //     return result;
    // }
    // if(op2 == 0.0)
    // {
    //     result =  0.0;
    //     if(KnobLog)
    //     {
    //         fprintf(logfile, " %.14f\n", result);
    //     }
    //     return result;
    // }

    op1_exp = (int)floor(log(op1)/log(2));
    op2_exp = (int)floor(log(op2)/log(2));
    op1_k   = (int)floor(op1_exp/(float)useed);
    op2_k   = (int)floor(op2_exp/(float)useed);
    if (op1_k > kmax)
    {
        op1_k = kmax;
    }
    else if (op1_k < kmin)
    {
        op1_k = kmin;
    }
    if (op2_k > kmax)
    {
        op2_k = kmax;
    }
    else if (op2_k < kmin)
    {
        op2_k = kmin;
    }

    if (op1_k >= 0)
    {
        if (op1_k == kmax)
        {
            op1_regimelen = op1_k + 1; 
        }
        else
        {
            op1_regimelen = op1_k + 2;
        }
    }
    else
    {
        if (op1_k == kmin)
        {
            op1_regimelen = (int)abs((double)op1_k); 
        }
        else
        {
            op1_regimelen = (int)abs((double)op1_k) + 1;
        }
    }
    if (op2_k >= 0)
    {
        if (op2_k == kmax)
        {
            op2_regimelen = op2_k + 1; 
        }
        else
        {
            op2_regimelen = op2_k + 2;
        }
    }
    else
    {
        if (op2_k == kmin)
        {
            op2_regimelen = (int)abs((double)op2_k); 
        }
        else
        {
            op2_regimelen = (int)abs((double)op2_k) + 1;
        }
    }

    op1_bitsremaining = posit_length - 1 - op1_regimelen;
    op2_bitsremaining = posit_length - 1 - op2_regimelen;   

    if (op1_bitsremaining >= es)
    {
        op1_eslen = es;
    }
    else
    {
        op1_eslen = op1_bitsremaining;
    }
    if (op2_bitsremaining >= es)
    {
        op2_eslen = es;
    }
    else
    {
        op2_eslen = op2_bitsremaining;
    }

    op1_e = op1_exp - op1_k*useed;
    op2_e = op2_exp - op2_k*useed;

    op1_e_remaining_max = (int)pow(2.0, op1_eslen) - 1;
    op2_e_remaining_max = (int)pow(2.0, op2_eslen) - 1;

    if(op1_e > op1_e_remaining_max)
    {
        op1_e = op1_e_remaining_max;
    }
    if(op2_e > op2_e_remaining_max)
    {
        op2_e = op2_e_remaining_max;
    }

    op1_frclen = posit_length - 1 - op1_regimelen - op1_eslen;
    op2_frclen = posit_length - 1 - op2_regimelen - op2_eslen;

    op1_f = (float)op1/pow(2.0, op1_exp);
    op2_f = (float)op2/pow(2.0, op2_exp);
    op1_fnew = op1_f - 1;
    op2_fnew = op2_f - 1;

    for (int i = 1; i < op1_frclen + 1; i++)
    {
        op1_fnew = op1_fnew -  pow(2.0, (-1*i)) ;
        if (op1_fnew < 0)
        {
            op1_fnew = op1_fnew + pow(2.0, (-1*i));
        }
    }
    for (int i = 1; i < op2_frclen + 1; i++)
    {
        op2_fnew = op2_fnew - pow(2.0, (-1*i));
        if (op2_fnew < 0)
        {
            op2_fnew = op2_fnew + pow(2.0, (-1*i));
        }
    }

    op1_f = op1_f - op1_fnew;
    op2_f = op2_f - op2_fnew;
 

   posit_sign = op1_sign;
   int sign_check = op1_sign + op2_sign;
   if (sign_check > 1)
   {
       sign_check = 0;
   }
   
   op1_e = op1_e + op1_k*useed;
   op2_e = op2_e + op2_k*useed;

   int shift = op1_e - op2_e;
   op2_f  = op2_f/pow(2.0, shift);

    posit_e = op1_e;
    if (sign_check == 0)
    {
        posit_f = op1_f + op2_f;
    }
    else
    {
        posit_f = op1_f - op2_f;
    }

    if (posit_f == 0)
    {
        result = 0.0;
        // if (KnobLog)
        // {
        //     fprintf(logfile, " %.14f\n", result);
        // }
        return result;
    }
    

    while (posit_f >= 2)
    {
        posit_f = posit_f/2;
        posit_e = posit_e + 1;
    }

    while (posit_f < 1)
    {
        posit_f = posit_f*2;
        posit_e = posit_e - 1;
    }


    posit_k = (int)floor(posit_e/(float)useed);

    if (posit_k > kmax)
    {
        posit_k = kmax;
    }
    else if (posit_k < kmin)
    {
        posit_k = kmin;
    }

    if (posit_k >= 0)
    {
        if (posit_k == kmax)
        {
            posit_regime_len = posit_k + 1;
        }
        else
        {
            posit_regime_len = posit_k + 2;
        }
    }
    else
    {
        if (posit_k == kmin)
        {
            posit_regime_len = (int)abs((double)posit_k);
        }
        else
        {
            posit_regime_len = (int)abs((double)posit_k) + 1;
        }
    }

    posit_bitsremaining = posit_length - 1 - posit_regime_len;
    if (posit_bitsremaining >= es)
    {
        posit_eslen = es;
    }
    else
    {
        posit_eslen = posit_bitsremaining;
    }
     
    posit_e = posit_e - posit_k*useed;
    posit_e_remaining_max = pow(2.0, posit_eslen) - 1;
    if (posit_e > posit_e_remaining_max)
    {
        posit_e = posit_e_remaining_max;
    }

    posit_frclen = posit_length - 1 - posit_regime_len - posit_eslen;
    posit_fnew = posit_f - 1;
    
    for (int i = 1; i < posit_frclen + 1; i++)
    {
        posit_fnew = posit_fnew - pow(2.0, (-1*i));
        if (posit_fnew < 0)
        {
            posit_fnew += pow(2.0, (-1*i));
        }
        
    }

    if (posit_sign == 1)
    {
        result = -1*((posit_f - posit_fnew)*pow(2.0, useed*posit_k + posit_e));
    }
    else
    {
        result = (posit_f - posit_fnew)*pow(2.0, useed*posit_k + posit_e);
    }

    // if(KnobLog)
    // {
    //     fprintf(logfile, " %.14f\n", result);
    // }

    return result;
}


float exact_multiply(float a, float b){
    float a1=a, b1=b;
    if(a<0)
        a1=-a;
    if(b<0)
        b1=-b;

    int8_t s = a1*32;
    int8_t w = b1*32;
    // printf("%d %d %d\n",s,w,s*w);
    int16_t result;
    result = s*w;

    int sign;
    if (a*b >= 0)
        sign = 1;
    else
        sign = -1;

    return sign*result/(32.0*32.0);
}


float posit_multiply(float a, float b, int tune)  
{
    //Input : Operand of multiplication
    //Ouput : Result of multiplication in IEEE-754 format
    //Flow  : The input operands are in IEEE-754 format. We convert them to posit format and then perform a posit multiplication.
    //        The result is obtained in posit format. We then return the result after converting it to IEEE-754 format.  
    // printf("%f, %f\n", a, b);
    float op1 = a;
    float op2 = b;
    int op1_sign,  op1_exp, op2_sign, op2_exp, op1_k,  op2_k, op1_e, op2_e, op1_regimelen,  op2_regimelen, op1_bitsremaining, op2_bitsremaining, op1_eslen, op2_eslen;
    int op1_e_remaining_max, op2_e_remaining_max, op1_frclen, op2_frclen;
    int posit_sign, posit_exp, posit_e, posit_k, posit_regime_len, posit_bitsremaining, posit_eslen, posit_e_remaining_max, posit_frclen;
    float posit_frc, op1_frc_new, op2_frc_new, frc_new, posit_f, posit_fnew;
    float op1_frc, op2_frc, op1_f, op2_f, op1_fnew, op2_fnew;
    float result;
    int kmax = posit_length - 2;
    int kmin = -1*(posit_length - 1);
    
    // if(KnobLog)
    // {
    //     fprintf(logfile, "%.14f %.14f", op1, op2);
    // }
    
    if(op1 < 0)
    {
        op1_sign = 1;
        op1 = -1*op1;
    }
    else
    {
        op1_sign = 0;
    }
    if(op2 < 0)
    {
        op2_sign = 1;
        op2 = -1*op2;
    }
    else
    {
        op2_sign = 0;
    }
    if(op1 == 0.0)
    {
        result =  0.0;
        // if(KnobLog)
        // {
        //     fprintf(logfile, " %.14f\n", result);
        // }
        return result;
    }
    if(op2 == 0.0)
    {
        result =  0.0;
        // if(KnobLog)
        // {
        //     fprintf(logfile, " %.14f\n", result);
        // }
        return result;
    }

    op1_exp = (int)floor(log(op1)/log(2));
    op2_exp = (int)floor(log(op2)/log(2));
    op1_k   = (int)floor(op1_exp/(float)useed);
    op2_k   = (int)floor(op2_exp/(float)useed);
    if (op1_k > kmax)
    {
        op1_k = kmax;
    }
    else if (op1_k < kmin)
    {
        op1_k = kmin;
    }
    if (op2_k > kmax)
    {
        op2_k = kmax;
    }
    else if (op2_k < kmin)
    {
        op2_k = kmin;
    }
    if (op1_k >= 0)
    {
        if (op1_k == kmax)
        {
            op1_regimelen = op1_k + 1; 
        }
        else
        {
            op1_regimelen = op1_k + 2;
        }
    }
    else
    {
        if (op1_k == kmin)
        {
            op1_regimelen = (int)abs((double)op1_k); 
        }
        else
        {
            op1_regimelen = (int)abs((double)op1_k) + 1;
        }
    }
    if (op2_k >= 0)
    {
        if (op2_k == kmax)
        {
            op2_regimelen = op2_k + 1; 
        }
        else
        {
            op2_regimelen = op2_k + 2;
        }
    }
    else
    {
        if (op2_k == kmin)
        {
            op2_regimelen = (int)abs((double)op2_k); 
        }
        else
        {
            op2_regimelen = (int)abs((double)op2_k) + 1;
        }
    }

    op1_bitsremaining = posit_length - 1 - op1_regimelen;
    op2_bitsremaining = posit_length - 1 - op2_regimelen;   

    if (op1_bitsremaining >= es)
    {
        op1_eslen = es;
    }
    else
    {
        op1_eslen = op1_bitsremaining;
    }
    if (op2_bitsremaining >= es)
    {
        op2_eslen = es;
    }
    else
    {
        op2_eslen = op2_bitsremaining;
    }

    op1_e = op1_exp - op1_k*useed;
    op2_e = op2_exp - op2_k*useed;

    op1_e_remaining_max = (int)pow(2.0, op1_eslen) - 1;
    op2_e_remaining_max = (int)pow(2.0, op2_eslen) - 1;

    if(op1_e > op1_e_remaining_max)
    {
        op1_e = op1_e_remaining_max;
    }
    if(op2_e > op2_e_remaining_max)
    {
        op2_e = op2_e_remaining_max;
    }

    op1_frclen = posit_length - 1 - op1_regimelen - op1_eslen;
    op2_frclen = posit_length - 1 - op2_regimelen - op2_eslen;

    op1_f = (float)op1/pow(2.0, op1_exp);
    op2_f = (float)op2/pow(2.0, op2_exp);
    op1_fnew = op1_f - 1;
    op2_fnew = op2_f - 1;

    for (int i = 1; i < op1_frclen + 1; i++)
    {
        op1_fnew = op1_fnew -  pow(2.0, (-1*i)) ;
        if (op1_fnew < 0)
        {
            op1_fnew = op1_fnew + pow(2.0, (-1*i));
        }
    }
    for (int i = 1; i < op2_frclen + 1; i++)
    {
        op2_fnew = op2_fnew - pow(2.0, (-1*i));
        if (op2_fnew < 0)
        {
            op2_fnew = op2_fnew + pow(2.0, (-1*i));
        }
    }

    op1_f = op1_f - op1_fnew;
    op2_f = op2_f - op2_fnew;

    posit_e = op1_e + op2_e;
    posit_f = op1_f * op2_f;

    if (posit_f >= 2)
    {
        posit_f = posit_f/2;
        posit_e = posit_e + 1;
    }

    posit_k = (int)floor(posit_e/(float)useed);

    if (posit_k > kmax)
    {
        posit_k = kmax;
    }
    else if (posit_k < kmin)
    {
        posit_k = kmin;
    }
    if (posit_k >= 0)
    {
        if (posit_k == kmax)
        {
            posit_regime_len = posit_k + 1;
        }
        else
        {
            posit_regime_len = posit_k + 2;
        }
    }
    else
    {
        if (posit_k == kmin)
        {
            posit_regime_len = (int)abs((double)posit_k);
        }
        else
        {
            posit_regime_len = (int)abs((double)posit_k) + 1;
        }
    }

    posit_bitsremaining = posit_length - 1 - posit_regime_len;
    if (posit_bitsremaining >= es)
    {
        posit_eslen = es;
    }
    else
    {
        posit_eslen = posit_bitsremaining;
    }
    
    posit_e = posit_e - posit_k*useed;
    posit_e_remaining_max = pow(2.0,  posit_eslen) - 1;
    if (posit_e > posit_e_remaining_max)
    {
        posit_e = posit_e_remaining_max;
    }

    posit_frclen = posit_length - 1 - posit_regime_len - posit_eslen;

    posit_fnew = posit_f - 1;
    for (int i = 1; i < posit_frclen + 1; i++)
    {
        posit_fnew = posit_fnew - pow(2.0, (-1*i));
        if (posit_fnew < 0)
        {
            posit_fnew += pow(2.0, (-1*i));
        }
        
    }

    posit_sign = op1_sign ^ op2_sign;

    if (posit_sign == 1)
    {
        result = -1*((posit_f - posit_fnew)*pow(2.0, useed*posit_k + posit_e));
    }
    else
    {
        result = (posit_f - posit_fnew)*pow(2.0, useed*posit_k + posit_e);
    }

    // if(KnobLog)
    // {
    //     fprintf(logfile, " %.14f\n", result);
    // }
    printf("%f, %f, %f\n", a, b, result);
    return result;
}







float posit_fixmultiply_new(float op1, float op2)  
{
    //Input : Operand of multiplication
    //Ouput : Result of multiplication in IEEE-754 format
    //Flow  : The input operands are in IEEE-754 format. We convert them to posit format and then perform a posit multiplication.
    //        The result is obtained in posit format. We then return the result after converting it to IEEE-754 format.  
    int posit_length = 6;
    int es = 2;
    int useed =4;//2^es
    int regime_len = 2;
    int frc_len = posit_length - es - regime_len - 1 + 1;
    // long int start   = -128;
    int op1_sign,  op1_exp, op2_sign, op2_exp; 
    int posit_sign, posit_k, posit_exp, posit_e;
    float posit_frc, op1_frc_new, op2_frc_new, frc_new;
    float op1_frc, op2_frc;
    float result;


    if(op1 < 0)
    {
        op1_sign = 1;
        op1 = -1*op1;
    }
    else
    {
        op1_sign = 0;
    }
    if(op2 < 0)
    {
        op2_sign = 1;
        op2 = -1*op2;
    }
    else
    {
        op2_sign = 0;
    }
    if(op1 == 0.0)
    {
        result =  0.0;
        return result;
    }
    if(op2 == 0.0)
    {
        result =  0.0;
        return result;
    }

    op1_exp = (int)floor(log2(op1));
    op2_exp = (int)floor(log2(op2));
    // printf("expo %d, %d\n", op1_exp, op2_exp);
    // op1_k   = (int)floor(op1_exp/(float)useed);
    // op2_k   = (int)floor(op2_exp/(float)useed);
    op1_frc = (float)op1/pow(2.0, op1_exp) - 1;
    op2_frc = (float)op2/pow(2.0, op2_exp) - 1;
    // printf(" frac %f, %f\n", op1_frc, op2_frc);
    
    // op2_frc = (float)op2/two_pows[op2_exp - start] - 1;
    op1_frc_new = 1;
    op2_frc_new = 1;
    // op1_e   = op1_exp - op1_k*useed;
    // op2_e   = op2_exp - op2_k*useed;
    posit_sign = op1_sign ^ op2_sign;
    posit_exp = op1_exp + op2_exp;
    // printf("posit sign and posit exponent %d, %d\n", posit_sign, posit_exp);

    for(int i=0; i < frc_len; i++)
    {
        op1_frc = op1_frc*2;
        if(op1_frc >= 1)
        {
            op1_frc_new = op1_frc_new + pow(2.0, -(i+1));
            op1_frc = op1_frc - 1; 
        }
        op2_frc = op2_frc*2;
        if(op2_frc >= 1)
        {
            op2_frc_new = op2_frc_new + pow(2.0, -(i+1));
            op2_frc = op2_frc - 1; 
        }
    }
    
    frc_new = (op1_frc_new * op2_frc_new);

    if(frc_new >=2)
    {
        frc_new = frc_new/2;
        posit_exp++;
    }
    frc_new = frc_new - 1;
    posit_frc = 1;
    for(int i=0; i < frc_len; i++)
    {
        frc_new = frc_new*2;
        if(frc_new >= 1)
        {
            posit_frc = posit_frc + pow(2.0, -(i+1));
            frc_new = frc_new - 1;
        }
    }
    posit_k = (int)floor((float)posit_exp/useed);
    posit_e = posit_exp - posit_k*useed;

    // printf(" posit %d, %d, %f\n", posit_k, posit_e, posit_frc);
    if(posit_sign == 1)
    {
            result =  -1*posit_frc*pow(2.0, useed*posit_k + posit_e);
            // printf("%d", pow(2.0, useed*posit_k + posit_e));
    }
    else
    {
            result = posit_frc*pow(2.0, useed*posit_k + posit_e);
            //  printf("above result %f, %d, %d\n", pow(2.0, useed*posit_k + posit_e), useed, useed*posit_k+posit_e);
    }

    // if(KnobLog)
    // {
    //     fprintf(logfile, " %.14f\n", result);
    // }
    // printf(" result%f\n",result);
    return result;
}

// int getarray(int arr[],int n)  
// {  
//     printf("Elements of array are : ");  
//     for(int i=0;i<n-1;i++)  
//     {  
//         printf("%d ", arr[i]);  
//     }
// }  

// void num2binary(int *bin, int n, int len){
//     for(int i = 0; i<len; i++){
//         bin[len-i-1] = (int)(n%2);
//         n = (int)(n/2);
//     }
// }

void *half_adder(int *out, int a, int b){
    int sum_h = a^b;
    int carry_h = a&b;
    out[0] = sum_h;
    out[1] = carry_h;
}

void *full_adder(int *out, int a, int b, int c){
    int sum_f = (a^b)^c;
    int carry_f = (a&b)|(b&c)|(c&a);
    out[0] = sum_f;
    out[1] = carry_f;
}

void *exact_compressor(int *out, int x1,int x2,int x3,int x4,int tin){
    int sum_e = ((x1^x2)^(x3^x4))^tin;
    int temp = (x1^x2)^x3;
    int carry_e = (temp & x4)|(temp & tin)|(tin & x4);  
    int tout = (x1 & x2)|(x3 & x2)|(x1 & x3);
    out[0] = sum_e;
    out[1] = carry_e;
    out[2] = tout;
}

void *yang2(int *out, int x1, int x2,int x3,int x4){
    int num_bi[4] = {x4, x3, x2, x1};
    int sum_a;
    int carry_a;
    char string[4];
    int num_de = 0;
    
    for(int i=0;i<4;i++){
        num_de = num_de + num_bi[i]*pow(2,i);
    }
    
    if (num_de == 0){
        sum_a = 0;
        carry_a = 0;
    }
    else if (num_de == 1){
        sum_a = 1;
        carry_a = 0;        
    }
    else if (num_de == 2){
        sum_a = 1;
        carry_a = 0;
    }
    else if (num_de ==3){
        sum_a = 0;
        carry_a = 1;
    }
    else if (num_de==4){
        sum_a = 1;
        carry_a = 0;
    }
    else if (num_de ==5){
        sum_a = 0;
        carry_a = 1;
    }
    else if (num_de ==6){
        sum_a = 0;
        carry_a = 1;
    }
    else if (num_de ==7){
        sum_a = 1;
        carry_a = 1;
    }
    else if (num_de ==8){
        sum_a = 1;
        carry_a = 0;
    }
    else if (num_de ==9){
        sum_a = 0;
        carry_a = 1;
    }
    else if (num_de ==10){
        sum_a = 0;
        carry_a = 1;
    }
    else if (num_de ==11){
        sum_a = 1;
        carry_a = 1;
    }
    else if (num_de ==12){
        sum_a = 1;
        carry_a = 1;
    }
    else if (num_de ==13){
        sum_a = 1;
        carry_a = 1;
    }
    else if (num_de ==14){
        sum_a = 1;
        carry_a = 1;
    }
    else{
        sum_a = 1;
        carry_a = 1;
    }
       
    out[0] = sum_a;
    out[1] = carry_a;
}

void *yang1(int *out, int x1, int x2,int x3,int x4){
    int num_bi[4] = {x4, x3, x2, x1};
    int sum_a;
    int carry_a;
    char string[4];
    int num_de = 0;
    
    for(int i=0;i<4;i++){
        num_de = num_de + num_bi[i]*pow(2,i);
    }
    
    if (num_de == 0){
        sum_a = 0;
        carry_a = 0;
    }
    else if (num_de == 1){
        sum_a = 1;
        carry_a = 0;        
    }
    else if (num_de == 2){
        sum_a = 1;
        carry_a = 0;
    }
    else if (num_de ==3){
        sum_a = 0;
        carry_a = 1;
    }
    else if (num_de==4){
        sum_a = 1;
        carry_a = 0;
    }
    else if (num_de ==5){
        sum_a = 0;
        carry_a = 1;
    }
    else if (num_de ==6){
        sum_a = 0;
        carry_a = 1;
    }
    else if (num_de ==7){
        sum_a = 1;
        carry_a = 1;
    }
    else if (num_de ==8){
        sum_a = 1;
        carry_a = 0;
    }
    else if (num_de ==9){
        sum_a = 0;
        carry_a = 1;
    }
    else if (num_de ==10){
        sum_a = 0;
        carry_a = 1;
    }
    else if (num_de ==11){
        sum_a = 1;
        carry_a = 1;
    }
    else if (num_de ==12){
        sum_a = 0;
        carry_a = 1;
    }
    else if (num_de ==13){
        sum_a = 1;
        carry_a = 1;
    }
    else if (num_de ==14){
        sum_a = 1;
        carry_a = 1;
    }
    else{
        sum_a = 1;
        carry_a = 1;
    }
       
    out[0] = sum_a;
    out[1] = carry_a;
}

void *yang3(int *out, int x1, int x2,int x3,int x4){
    int num_bi[4] = {x4, x3, x2, x1};
    int sum_a;
    int carry_a;
    char string[4];
    int num_de = 0;
    
    for(int i=0;i<4;i++){
        num_de = num_de + num_bi[i]*pow(2,i);
    }
    
    if (num_de == 0){
        sum_a = 0;
        carry_a = 0;
    }
    else if (num_de == 1){
        sum_a = 1;
        carry_a = 0;        
    }
    else if (num_de == 2){
        sum_a = 1;
        carry_a = 0;
    }
    else if (num_de ==3){
        sum_a = 0;
        carry_a = 1;
    }
    else if (num_de==4){
        sum_a = 1;
        carry_a = 0;
    }
    else if (num_de ==5){
        sum_a = 0;
        carry_a = 1;
    }
    else if (num_de ==6){
        sum_a = 0;
        carry_a = 1;
    }
    else if (num_de ==7){
        sum_a = 1;
        carry_a = 1;
    }
    else if (num_de ==8){
        sum_a = 1;
        carry_a = 0;
    }
    else if (num_de ==9){
        sum_a = 0;
        carry_a = 1;
    }
    else if (num_de ==10){
        sum_a = 0;
        carry_a = 1;
    }
    else if (num_de ==11){
        sum_a = 1;
        carry_a = 1;
    }
    else if (num_de ==12){
        sum_a = 1;
        carry_a = 1;
    }
    else if (num_de ==13){
        sum_a = 0;
        carry_a = 1;
    }
    else if (num_de ==14){
        sum_a = 0;
        carry_a = 1;
    }
    else{
        sum_a = 1;
        carry_a = 1;
    }
       
    out[0] = sum_a;
    out[1] = carry_a;
}

void *lin(int *out, int x1, int x2,int x3,int x4){
    int num_bi[4] = {x4, x3, x2, x1};
    int sum_a;
    int carry_a;
    char string[4];
    int num_de = 0;
    
    for(int i=0;i<4;i++){
        num_de = num_de + num_bi[i]*pow(2,i);
    }
    
    if (num_de == 0){
        sum_a = 0;
        carry_a = 0;
    }
    else if (num_de == 1){
        sum_a = 1;
        carry_a = 0;        
    }
    else if (num_de == 2){
        sum_a = 1;
        carry_a = 0;
    }
    else if (num_de ==3){
        sum_a = 0;
        carry_a = 1;
    }
    else if (num_de==4){
        sum_a = 1;
        carry_a = 0;
    }
    else if (num_de ==5){
        sum_a = 0;
        carry_a = 1;
    }
    else if (num_de ==6){
        sum_a = 0;
        carry_a = 1;
    }
    else if (num_de ==7){
        sum_a = 1;
        carry_a = 1;
    }
    else if (num_de ==8){
        sum_a = 1;
        carry_a = 0;
    }
    else if (num_de ==9){
        sum_a = 0;
        carry_a = 1;
    }
    else if (num_de ==10){
        sum_a = 0;
        carry_a = 1;
    }
    else if (num_de ==11){
        sum_a = 1;
        carry_a = 1;
    }
    else if (num_de ==12){
        sum_a = 0;
        carry_a = 1;
    }
    else if (num_de ==13){
        sum_a = 1;
        carry_a = 1;
    }
    else if (num_de ==14){
        sum_a = 1;
        carry_a = 1;
    }
    else{
        sum_a = 0;
        carry_a = 1;
    }
       
    out[0] = sum_a;
    out[1] = carry_a;
}

void *strollo1(int *out, int x1, int x2,int x3,int x4){
    int num_bi[4] = {x4, x3, x2, x1};
    int sum_a;
    int carry_a;
    char string[4];
    int num_de = 0;
    
    for(int i=0;i<4;i++){
        num_de = num_de + num_bi[i]*pow(2,i);
    }
    
    if (num_de == 0){
        sum_a = 0;
        carry_a = 0;
    }
    else if (num_de == 1){
        sum_a = 1;
        carry_a = 0;        
    }
    else if (num_de == 2){
        sum_a = 1;
        carry_a = 0;
    }
    else if (num_de ==3){
        sum_a = 0;
        carry_a = 1;
    }
    else if (num_de==4){
        sum_a = 1;
        carry_a = 0;
    }
    else if (num_de ==5){
        sum_a = 0;
        carry_a = 1;
    }
    else if (num_de ==6){
        sum_a = 0;
        carry_a = 1;
    }
    else if (num_de ==7){
        sum_a = 1;
        carry_a = 1;
    }
    else if (num_de ==8){
        sum_a = 1;
        carry_a = 0;
    }
    else if (num_de ==9){
        sum_a = 0;
        carry_a = 1;
    }
    else if (num_de ==10){
        sum_a = 0;
        carry_a = 1;
    }
    else if (num_de ==11){
        sum_a = 1;
        carry_a = 1;
    }
    else if (num_de ==12){
        sum_a = 1;
        carry_a = 0;
    }
    else if (num_de ==13){
        sum_a = 0;
        carry_a = 1;
    }
    else if (num_de ==14){
        sum_a = 0;
        carry_a = 1;
    }
    else{
        sum_a = 1;
        carry_a = 1;
    }
       
    out[0] = sum_a;
    out[1] = carry_a;
}

void *sabetz(int *out, int x1, int x2,int x3,int x4){
    int num_bi[4] = {x4, x3, x2, x1};
    int sum_a;
    int carry_a;
    char string[4];
    int num_de = 0;
    
    for(int i=0;i<4;i++){
        num_de = num_de + num_bi[i]*pow(2,i);
    }
    
    if (num_de == 0){
        sum_a = 1;
        carry_a = 0;
    }
    else if (num_de == 1){
        sum_a = 1;
        carry_a = 0;        
    }
    else if (num_de == 2){
        sum_a = 1;
        carry_a = 0;
    }
    else if (num_de ==3){
        sum_a = 1;
        carry_a = 0;
    }
    else if (num_de==4){
        sum_a = 1;
        carry_a = 0;
    }
    else if (num_de ==5){
        sum_a = 1;
        carry_a = 1;
    }
    else if (num_de ==6){
        sum_a = 1;
        carry_a = 0;
    }
    else if (num_de ==7){
        sum_a = 1;
        carry_a = 1;
    }
    else if (num_de ==8){
        sum_a = 1;
        carry_a = 0;
    }
    else if (num_de ==9){
        sum_a = 1;
        carry_a = 1;
    }
    else if (num_de ==10){
        sum_a = 1;
        carry_a = 0;
    }
    else if (num_de ==11){
        sum_a = 1;
        carry_a = 1;
    }
    else if (num_de ==12){
        sum_a = 1;
        carry_a = 1;
    }
    else if (num_de ==13){
        sum_a = 1;
        carry_a = 1;
    }
    else if (num_de ==14){
        sum_a = 1;
        carry_a = 1;
    }
    else{
        sum_a = 1;
        carry_a = 1;
    }
       
    out[0] = sum_a;
    out[1] = carry_a;
}

void *strollo2(int *out, int x1, int x2,int x3,int x4){
    int num_bi[4] = {x4, x3, x2, x1};
    int sum_a;
    int carry_a;
    char string[4];
    int num_de = 0;
    
    for(int i=0;i<4;i++){
        num_de = num_de + num_bi[i]*pow(2,i);
    }
    
    if (num_de == 0){
        sum_a = 0;
        carry_a = 0;
    }
    else if (num_de == 1){
        sum_a = 1;
        carry_a = 0;        
    }
    else if (num_de == 2){
        sum_a = 1;
        carry_a = 0;
    }
    else if (num_de ==3){
        sum_a = 0;
        carry_a = 1;
    }
    else if (num_de==4){
        sum_a = 1;
        carry_a = 0;
    }
    else if (num_de ==5){
        sum_a = 0;
        carry_a = 1;
    }
    else if (num_de ==6){
        sum_a = 0;
        carry_a = 1;
    }
    else if (num_de ==7){
        sum_a = 0;
        carry_a = 1;
    }
    else if (num_de ==8){
        sum_a = 1;
        carry_a = 0;
    }
    else if (num_de ==9){
        sum_a = 0;
        carry_a = 1;
    }
    else if (num_de ==10){
        sum_a = 0;
        carry_a = 1;
    }
    else if (num_de ==11){
        sum_a = 1;
        carry_a = 1;
    }
    else if (num_de ==12){
        sum_a = 0;
        carry_a = 1;
    }
    else if (num_de ==13){
        sum_a = 1;
        carry_a = 1;
    }
    else if (num_de ==14){
        sum_a = 1;
        carry_a = 1;
    }
    else{
        sum_a = 1;
        carry_a = 1;
    }
       
    out[0] = sum_a;
    out[1] = carry_a;
}

void *momeni(int *out, int x1, int x2,int x3,int x4){
    int num_bi[4] = {x4, x3, x2, x1};
    int sum_a;
    int carry_a;
    char string[4];
    int num_de = 0;
    
    for(int i=0;i<4;i++){
        num_de = num_de + num_bi[i]*pow(2,i);
    }
    
    if (num_de == 0){
        sum_a = 1;
        carry_a = 0;
    }
    else if (num_de == 1){
        sum_a = 1;
        carry_a = 0;        
    }
    else if (num_de == 2){
        sum_a = 1;
        carry_a = 0;
    }
    else if (num_de ==3){
        sum_a = 1;
        carry_a = 0;
    }
    else if (num_de==4){
        sum_a = 1;
        carry_a = 0;
    }
    else if (num_de ==5){
        sum_a = 0;
        carry_a = 1;
    }
    else if (num_de ==6){
        sum_a = 0;
        carry_a = 1;
    }
    else if (num_de ==7){
        sum_a = 1;
        carry_a = 1;
    }
    else if (num_de ==8){
        sum_a = 1;
        carry_a = 0;
    }
    else if (num_de ==9){
        sum_a = 0;
        carry_a = 1;
    }
    else if (num_de ==10){
        sum_a = 0;
        carry_a = 1;
    }
    else if (num_de ==11){
        sum_a = 1;
        carry_a = 1;
    }
    else if (num_de ==12){
        sum_a = 1;
        carry_a = 0;
    }
    else if (num_de ==13){
        sum_a = 1;
        carry_a = 1;
    }
    else if (num_de ==14){
        sum_a = 1;
        carry_a = 1;
    }
    else{
        sum_a = 1;
        carry_a = 1;
    }
       
    out[0] = sum_a;
    out[1] = carry_a;
}

void *venka(int *out, int x1, int x2,int x3,int x4){
    int num_bi[4] = {x4, x3, x2, x1};
    int sum_a;
    int carry_a;
    char string[4];
    int num_de = 0;
    
    for(int i=0;i<4;i++){
        num_de = num_de + num_bi[i]*pow(2,i);
    }
    
    if (num_de == 0){
        sum_a = 0;
        carry_a = 0;
    }
    else if (num_de == 1){
        sum_a = 1;
        carry_a = 0;        
    }
    else if (num_de == 2){
        sum_a = 1;
        carry_a = 0;
    }
    else if (num_de ==3){
        sum_a = 0;
        carry_a = 1;
    }
    else if (num_de==4){
        sum_a = 1;
        carry_a = 0;
    }
    else if (num_de ==5){
        sum_a = 1;
        carry_a = 0;
    }
    else if (num_de ==6){
        sum_a = 1;
        carry_a = 0;
    }
    else if (num_de ==7){
        sum_a = 1;
        carry_a = 1;
    }
    else if (num_de ==8){
        sum_a = 1;
        carry_a = 0;
    }
    else if (num_de ==9){
        sum_a = 1;
        carry_a = 0;
    }
    else if (num_de ==10){
        sum_a = 1;
        carry_a = 0;
    }
    else if (num_de ==11){
        sum_a = 1;
        carry_a = 1;
    }
    else if (num_de ==12){
        sum_a = 0;
        carry_a = 1;
    }
    else if (num_de ==13){
        sum_a = 1;
        carry_a = 1;
    }
    else if (num_de ==14){
        sum_a = 1;
        carry_a = 1;
    }
    else{
        sum_a = 1;
        carry_a = 1;
    }
       
    out[0] = sum_a;
    out[1] = carry_a;
}

void *akbar1(int *out, int x1, int x2,int x3,int x4){
    int num_bi[4] = {x4, x3, x2, x1};
    int sum_a;
    int carry_a;
    char string[4];
    int num_de = 0;
    
    for(int i=0;i<4;i++){
        num_de = num_de + num_bi[i]*pow(2,i);
    }
    
    if (num_de == 0){
        sum_a = 0;
        carry_a = 0;
    }
    else if (num_de == 1){
        sum_a = 1;
        carry_a = 0;        
    }
    else if (num_de == 2){
        sum_a = 1;
        carry_a = 0;
    }
    else if (num_de ==3){
        sum_a = 0;
        carry_a = 0;
    }
    else if (num_de==4){
        sum_a = 1;
        carry_a = 0;
    }
    else if (num_de ==5){
        sum_a = 1;
        carry_a = 0;
    }
    else if (num_de ==6){
        sum_a = 1;
        carry_a = 0;
    }
    else if (num_de ==7){
        sum_a = 1;
        carry_a = 0;
    }
    else if (num_de ==8){
        sum_a = 1;
        carry_a = 1;
    }
    else if (num_de ==9){
        sum_a = 1;
        carry_a = 1;
    }
    else if (num_de ==10){
        sum_a = 1;
        carry_a = 1;
    }
    else if (num_de ==11){
        sum_a = 1;
        carry_a = 1;
    }
    else if (num_de ==12){
        sum_a = 0;
        carry_a = 1;
    }
    else if (num_de ==13){
        sum_a = 1;
        carry_a = 1;
    }
    else if (num_de ==14){
        sum_a = 1;
        carry_a = 1;
    }
    else{
        sum_a = 0;
        carry_a = 1;
    }
       
    out[0] = sum_a;
    out[1] = carry_a;
}

void *akbar2(int *out, int x1, int x2,int x3,int x4){
    int num_bi[4] = {x4, x3, x2, x1};
    int sum_a;
    int carry_a;
    char string[4];
    int num_de = 0;
    
    for(int i=0;i<4;i++){
        num_de = num_de + num_bi[i]*pow(2,i);
    }
    
    if (num_de == 0){
        sum_a = 0;
        carry_a = 0;
    }
    else if (num_de == 1){
        sum_a = 1;
        carry_a = 0;        
    }
    else if (num_de == 2){
        sum_a = 1;
        carry_a = 0;
    }
    else if (num_de ==3){
        sum_a = 0;
        carry_a = 1;
    }
    else if (num_de==4){
        sum_a = 1;
        carry_a = 0;
    }
    else if (num_de ==5){
        sum_a = 1;
        carry_a = 1;
    }
    else if (num_de ==6){
        sum_a = 1;
        carry_a = 1;
    }
    else if (num_de ==7){
        sum_a = 1;
        carry_a = 1;
    }
    else if (num_de ==8){
        sum_a = 1;
        carry_a = 0;
    }
    else if (num_de ==9){
        sum_a = 1;
        carry_a = 1;
    }
    else if (num_de ==10){
        sum_a = 1;
        carry_a = 1;
    }
    else if (num_de ==11){
        sum_a = 1;
        carry_a = 1;
    }
    else if (num_de ==12){
        sum_a = 0;
        carry_a = 1;
    }
    else if (num_de ==13){
        sum_a = 1;
        carry_a = 1;
    }
    else if (num_de ==14){
        sum_a = 1;
        carry_a = 1;
    }
    else{
        sum_a = 0;
        carry_a = 1;
    }
       
    out[0] = sum_a;
    out[1] = carry_a;
}

void *ahma(int *out, int x1, int x2,int x3,int x4){
    int num_bi[4] = {x4, x3, x2, x1};
    int sum_a;
    int carry_a;
    char string[4];
    int num_de = 0;
    
    for(int i=0;i<4;i++){
        num_de = num_de + num_bi[i]*pow(2,i);
    }
    
    if (num_de == 0){
        sum_a = 0;
        carry_a = 0;
    }
    else if (num_de == 1){
        sum_a = 1;
        carry_a = 0;        
    }
    else if (num_de == 2){
        sum_a = 1;
        carry_a = 0;
    }
    else if (num_de ==3){
        sum_a = 1;
        carry_a = 0;
    }
    else if (num_de==4){
        sum_a = 1;
        carry_a = 0;
    }
    else if (num_de ==5){
        sum_a = 1;
        carry_a = 1;
    }
    else if (num_de ==6){
        sum_a = 1;
        carry_a = 1;
    }
    else if (num_de ==7){
        sum_a = 1;
        carry_a = 1;
    }
    else if (num_de ==8){
        sum_a = 1;
        carry_a = 0;
    }
    else if (num_de ==9){
        sum_a = 1;
        carry_a = 1;
    }
    else if (num_de ==10){
        sum_a = 1;
        carry_a = 1;
    }
    else if (num_de ==11){
        sum_a = 1;
        carry_a = 1;
    }
    else if (num_de ==12){
        sum_a = 1;
        carry_a = 0;
    }
    else if (num_de ==13){
        sum_a = 1;
        carry_a = 1;
    }
    else if (num_de ==14){
        sum_a = 1;
        carry_a = 1;
    }
    else{
        sum_a = 1;
        carry_a = 1;
    }
       
    out[0] = sum_a;
    out[1] = carry_a;
}

void *ranjbar1(int *out, int x1, int x2,int x3,int x4){
    int num_bi[4] = {x4, x3, x2, x1};
    int sum_a;
    int carry_a;
    char string[4];
    int num_de = 0;
    
    for(int i=0;i<4;i++){
        num_de = num_de + num_bi[i]*pow(2,i);
    }
    
    if (num_de == 0){
        sum_a = 0;
        carry_a = 0;
    }
    else if (num_de == 1){
        sum_a = 0;
        carry_a = 1;        
    }
    else if (num_de == 2){
        sum_a = 0;
        carry_a = 1;
    }
    else if (num_de ==3){
        sum_a = 0;
        carry_a = 1;
    }
    else if (num_de==4){
        sum_a = 1;
        carry_a = 0;
    }
    else if (num_de ==5){
        sum_a = 0;
        carry_a = 1;
    }
    else if (num_de ==6){
        sum_a = 0;
        carry_a = 1;
    }
    else if (num_de ==7){
        sum_a = 1;
        carry_a = 1;
    }
    else if (num_de ==8){
        sum_a = 1;
        carry_a = 0;
    }
    else if (num_de ==9){
        sum_a = 0;
        carry_a = 1;
    }
    else if (num_de ==10){
        sum_a = 0;
        carry_a = 1;
    }
    else if (num_de ==11){
        sum_a = 1;
        carry_a = 1;
    }
    else if (num_de ==12){
        sum_a = 1;
        carry_a = 0;
    }
    else if (num_de ==13){
        sum_a = 1;
        carry_a = 1;
    }
    else if (num_de ==14){
        sum_a = 1;
        carry_a = 1;
    }
    else{
        sum_a = 1;
        carry_a = 1;
    }
       
    out[0] = sum_a;
    out[1] = carry_a;
}

void *ranjbar2(int *out, int x1, int x2,int x3,int x4){
    int num_bi[4] = {x4, x3, x2, x1};
    int sum_a;
    int carry_a;
    char string[4];
    int num_de = 0;
    
    for(int i=0;i<4;i++){
        num_de = num_de + num_bi[i]*pow(2,i);
    }
    
    if (num_de == 0){
        sum_a = 0;
        carry_a = 0;
    }
    else if (num_de == 1){
        sum_a = 1;
        carry_a = 0;        
    }
    else if (num_de == 2){
        sum_a = 0;
        carry_a = 1;
    }
    else if (num_de ==3){
        sum_a = 0;
        carry_a = 1;
    }
    else if (num_de==4){
        sum_a = 0;
        carry_a = 1;
    }
    else if (num_de ==5){
        sum_a = 0;
        carry_a = 1;
    }
    else if (num_de ==6){
        sum_a = 0;
        carry_a = 1;
    }
    else if (num_de ==7){
        sum_a = 1;
        carry_a = 1;
    }
    else if (num_de ==8){
        sum_a = 1;
        carry_a = 0;
    }
    else if (num_de ==9){
        sum_a = 1;
        carry_a = 0;
    }
    else if (num_de ==10){
        sum_a = 0;
        carry_a = 1;
    }
    else if (num_de ==11){
        sum_a = 1;
        carry_a = 1;
    }
    else if (num_de ==12){
        sum_a = 0;
        carry_a = 1;
    }
    else if (num_de ==13){
        sum_a = 1;
        carry_a = 1;
    }
    else if (num_de ==14){
        sum_a = 1;
        carry_a = 1;
    }
    else{
        sum_a = 1;
        carry_a = 1;
    }
       
    out[0] = sum_a;
    out[1] = carry_a;
}

void *ranjbar3(int *out, int x1, int x2,int x3,int x4){
    int num_bi[4] = {x4, x3, x2, x1};
    int sum_a;
    int carry_a;
    char string[4];
    int num_de = 0;
    
    for(int i=0;i<4;i++){
        num_de = num_de + num_bi[i]*pow(2,i);
    }
    
    if (num_de == 0){
        sum_a = 0;
        carry_a = 0;
    }
    else if (num_de == 1){
        sum_a = 0;
        carry_a = 1;        
    }
    else if (num_de == 2){
        sum_a = 1;
        carry_a = 0;
    }
    else if (num_de ==3){
        sum_a = 0;
        carry_a = 1;
    }
    else if (num_de==4){
        sum_a = 0;
        carry_a = 1;
    }
    else if (num_de ==5){
        sum_a = 0;
        carry_a = 1;
    }
    else if (num_de ==6){
        sum_a = 0;
        carry_a = 1;
    }
    else if (num_de ==7){
        sum_a = 1;
        carry_a = 1;
    }
    else if (num_de ==8){
        sum_a = 1;
        carry_a = 0;
    }
    else if (num_de ==9){
        sum_a = 0;
        carry_a = 1;
    }
    else if (num_de ==10){
        sum_a = 1;
        carry_a = 0;
    }
    else if (num_de ==11){
        sum_a = 1;
        carry_a = 1;
    }
    else if (num_de ==12){
        sum_a = 0;
        carry_a = 1;
    }
    else if (num_de ==13){
        sum_a = 1;
        carry_a = 1;
    }
    else if (num_de ==14){
        sum_a = 1;
        carry_a = 1;
    }
    else{
        sum_a = 1;
        carry_a = 1;
    }
       
    out[0] = sum_a;
    out[1] = carry_a;
}


float approx_multiplier(float a,float b){
    
   
    int a_bi[8];
    int b_bi[8];
    num2binary(a_bi, a, 8);
    num2binary(b_bi, b, 8);
    
    int p[8][8];

    int a1_bi[8] = {(a_bi[0]),(a_bi[1]),(a_bi[2]),(a_bi[3]),(a_bi[4]),(a_bi[5]),(a_bi[6]),(a_bi[7])};
    int b1_bi[8] = {(b_bi[0]),(b_bi[1]),(b_bi[2]),(b_bi[3]),(b_bi[4]),(b_bi[5]),(b_bi[6]),(b_bi[7])};
    
    // Partial product matrix generation
    for (int i=0;i<8;i++){
        for (int j=0;j<8;j++){
            p[i][j]=a1_bi[7-j]*b1_bi[7-i]; 
        }
    }
    
    // sy11,cy11 = ahma(p[0,5],p[1,4],p[2,3],p[3,2])
    int yangf1[2],yangf2[2],yangf3[2],yangf4[2],ahmaa_out[2],ahmab_out[2],ahmac_out[2],ahmad_out[2],ahmae_out[2];
    int halfaddr1_out[2],halfaddr2_out[2],halfaddr3_out[2],halfaddr4_out[2],halfaddr5_out[2];
    int fulladdr1_out[2],fulladdr2_out[2],fulladdr3_out[2],fulladdr4_out[2],fulladdr5_out[2],fulladdr6_out[2],fulladdr7_out[2],fulladdr8_out[2],fulladdr9_out[2],fulladdr10_out[2],fulladdr11_out[2],fulladdr12_out[2],fulladdr13_out[2],fulladdr14_out[2],fulladdr15_out[2],fulladdr16_out[2];
    int exact_compressor0[3],exact_compressor1[3],exact_compressor2[3],exact_compressor1_out[3],exact_compressor2_out[3],exact_compressor3_out[3],exact_compressor4_out[3],exact_compressor5_out[3];
    
    //First stage of PP reduction from 8 to 4 in C-N configuration
    //calahmag half adders
    half_adder(halfaddr1_out,p[0][4],p[1][3]);
    int sh1 = halfaddr1_out[0];
    int ch1 = halfaddr1_out[1];
    half_adder(halfaddr2_out,p[4][2],p[5][1]);
    int sh2 = halfaddr2_out[0];
    int ch2 = halfaddr2_out[1];
    half_adder(halfaddr3_out,p[6][3],p[7][2]);
    int sh3 = halfaddr3_out[0];
    int ch3 = halfaddr3_out[1]; 
    
    ahma(yangf1,p[0][5],p[1][4],p[2][3],p[3][2]);
    int sy11 = yangf1[0];
    int cy11 = yangf1[1];
    // printf("%d\n",sy11);
    // sy12,cy12 = ahma(p[0,6],p[1,5],p[2,4],p[3,3])
    ahma(yangf2,p[0][6],p[1][5],p[2][4],p[3][3]);
    int sy12 = yangf2[0];
    int cy12 = yangf2[1];
    // sy13,cy13 = ahma(p[0,7],p[1,6],p[2,5],p[3,4])
    ahma(yangf3,p[0][7],p[1][6],p[2][5],p[3][4]);
    int sy13 = yangf3[0];
    int cy13 = yangf3[1];
    // sy14,cy14 = ahma(p[4,3],p[5,2],p[6,1],p[7,0])
    ahma(yangf4,p[4][3],p[5][2],p[6][1],p[7][0]);
    int sy14 = yangf4[0];
    int cy14 = yangf4[1];   
    

    //calahmag full adders
    full_adder(fulladdr1_out,p[5][3],p[6][2],p[7][1]);
    int sf1 = fulladdr1_out[0];
    int cf1 = fulladdr1_out[1];

    //calahmag approximate compressor ahma which is used in lower 8 columns
    // se11,ce11,tout11 = exact_compressor(p[1,7],p[2,6],p[3,5],p[4,4],0)
    exact_compressor(exact_compressor0,p[1][7],p[2][6],p[3][5],p[4][4],0);
    int se11 = exact_compressor0[0];
    // printf("%d\n",se11);
    int ce11 = exact_compressor0[1];
    // printf("%d",ce11);
    int tout11 = exact_compressor0[2];
    //printf("%d",tout11);
    // [se12,ce12,tout12] = exact_compressor(p[2,7],p[3,6],p[4,5],p[5,4],tout11);
    exact_compressor(exact_compressor1,p[2][7],p[3][6],p[4][5],p[5][4],tout11);
    int se12 = exact_compressor1[0];
    // printf("%d ", se12);
    int ce12 = exact_compressor1[1];
    int tout12 = exact_compressor1[2];
    // [se13,ce13,tout13] = exact_compressor(p[3,7],p[4,6],p[5,5],p[6,4],tout12);
    exact_compressor(exact_compressor2,p[3][7],p[4][6],p[5][5],p[6][4],tout12);
    int se13 = exact_compressor2[0];
    int ce13 = exact_compressor2[1];
    int tout13 = exact_compressor2[2];

    //calahmag full adder which takes carry output of exact compressor 3 as its third input
    full_adder(fulladdr2_out,p[4][7],p[5][6],tout13);
    int sf2 = fulladdr2_out[0];
    int cf2 = fulladdr2_out[1];


    //Second stage of PP reduction from 4 to 2 in C-N configuration
   
    //calahmag half adders
    half_adder(halfaddr4_out,p[0][2],p[1][1]);
    int sh4 = halfaddr4_out[0];
    int ch4 = halfaddr4_out[1]; 

    //calahmag approximate compressor ahma which is used in lower 8 columns
    // [sy21,cy21] = ahma(p[0,3],p[1,2],p[2,1],p[3,0]);
    ahma(ahmaa_out,p[0][3],p[1][2],p[2][1],p[3][0]);
    int sy21 = ahmaa_out[0];
    int cy21 = ahmaa_out[1];
    // [sy22,cy22] = ahma(sh1,p[2,2],p[3,1],p[4,0]);
    ahma(ahmab_out,sh1,p[2][2],p[3][1],p[4][0]);
    int sy22 = ahmab_out[0];
    int cy22 = ahmab_out[1];
    // [sy23,cy23] = ahma(sy11,ch1,p[4,1],p[5,0]);
    ahma(ahmac_out,sy11,ch1,p[4][1],p[5][0]);
    int sy23 = ahmac_out[0];
    int cy23 = ahmac_out[1];
    // [sy24,cy24] = ahma(sy12,cy11,sh2,p[6,0]);
    ahma(ahmad_out,sy12,cy11,sh2,p[6][0]);
    int sy24 = ahmad_out[0];
    int cy24 = ahmad_out[1];
    // [sy25,cy25] = ahma(sy13,cy12,sy14,ch2);
    ahma(ahmae_out,sy13,cy12,sy14,ch2);
    int sy25 = ahmae_out[0];
    int cy25 = ahmae_out[1];

    //calahmag exact compressor  which is used in higher 8 columns

    // [se21,ce21,tout21] = exact_compressor(se11,cy13,sf1,cy14,0);
    exact_compressor(exact_compressor1_out,se11,cy13,sf1,cy14,0);
    int se21 = exact_compressor1_out[0];
    int ce21 = exact_compressor1_out[1];
    int tout21 = exact_compressor1_out[2];
    exact_compressor(exact_compressor2_out,se12,ce11,sh3,cf1,tout21);
    int se22 = exact_compressor2_out[0];
    int ce22 = exact_compressor2_out[1];
    int tout22 = exact_compressor2_out[2];
    exact_compressor(exact_compressor3_out,se13,ce12,p[7][3],ch3,tout22);
    int se23 = exact_compressor3_out[0];
    int ce23 = exact_compressor3_out[1];
    int tout23 = exact_compressor3_out[2];
    exact_compressor(exact_compressor4_out,sf2,ce13,p[6][5],p[7][4],tout23);
    int se24 = exact_compressor4_out[0];
    int ce24 = exact_compressor4_out[1];
    int tout24 = exact_compressor4_out[2];
    exact_compressor(exact_compressor5_out,p[5][7],cf2,p[6][6],p[7][5],tout24);
    int se25 = exact_compressor5_out[0];
    int ce25 = exact_compressor5_out[1];
    int tout25 = exact_compressor5_out[2];

    //calahmag full adder which takes carry output of exact compressor 5 as its third input
    full_adder(fulladdr3_out,p[6][7],p[7][6],tout25);
    int sf3 = fulladdr3_out[0];
    int cf3 = fulladdr3_out[1];


    //Third stage of carry propagation addition of two final rows in C-N configuration

    //calahmag half adders
    half_adder(halfaddr5_out,p[0][1],p[1][0]);
    int sh5 = halfaddr5_out[0];
    int ch5 = halfaddr5_out[1];

    //calahmag full adders
    // [sf4,cf4] = full_adder(sh4,p[2,0],ch5);
    full_adder(fulladdr4_out,sh4,p[2][0],ch5);
    int sf4 = fulladdr4_out[0];
    int cf4 = fulladdr4_out[1];
    // [sf5,cf5] = full_adder(sy21,ch4,cf4);
    full_adder(fulladdr5_out,sy21,ch4,cf4);
    int sf5 = fulladdr5_out[0];
    //printf("%d\n", sf5);
    int cf5 = fulladdr5_out[1];
    //printf("%d\n", cf5);
    // [sf6,cf6] = full_adder(sy22,cy21,cf5);
    full_adder(fulladdr6_out,sy22,cy21,cf5);
    int sf6 = fulladdr6_out[0];
    int cf6 = fulladdr6_out[1];
    // [sf7,cf7] = full_adder(sy23,cy22,cf6);
    full_adder(fulladdr7_out,sy23,cy22,cf6);
    int sf7 = fulladdr7_out[0];
    int cf7 = fulladdr7_out[1];
    // [sf8,cf8] = full_adder(sy24,cy23,cf7);
    full_adder(fulladdr8_out,sy24,cy23,cf7);
    int sf8 = fulladdr8_out[0];
    int cf8 = fulladdr8_out[1];
    // [sf9,cf9] = full_adder(sy25,cy24,cf8);
    full_adder(fulladdr9_out,sy25,cy24,cf8);
    int sf9 = fulladdr9_out[0];
    int cf9 = fulladdr9_out[1];
    // [sf10,cf10] = full_adder(se21,cy25,cf9);
    full_adder(fulladdr10_out,se21,cy25,cf9);
    int sf10 = fulladdr10_out[0];
    int cf10 = fulladdr10_out[1];
    // [sf11,cf11] = full_adder(se22,ce21,cf10);
    full_adder(fulladdr11_out,se22,ce21,cf10);
    int sf11 = fulladdr11_out[0];
    int cf11 = fulladdr11_out[1];
    // [sf12,cf12] = full_adder(se23,ce22,cf11);
    full_adder(fulladdr12_out,se23,ce22,cf11);
    int sf12 = fulladdr12_out[0];
    int cf12 = fulladdr12_out[1];
    // [sf13,cf13] = full_adder(se24,ce23,cf12);
    full_adder(fulladdr13_out,se24,ce23,cf12);
    int sf13 = fulladdr13_out[0];
    int cf13 = fulladdr13_out[1];
    // [sf14,cf14] = full_adder(se25,ce24,cf13);
    full_adder(fulladdr14_out,se25,ce24,cf13);
    int sf14 = fulladdr14_out[0];
    int cf14 = fulladdr14_out[1];
    // [sf15,cf15] = full_adder(sf3,ce25,cf14);
    full_adder(fulladdr15_out,sf3,ce25,cf14);
    int sf15 = fulladdr15_out[0];
    int cf15 = fulladdr15_out[1];
    // [sf16,cf16] = full_adder(p[7,7],cf3,cf15);
    full_adder(fulladdr16_out,p[7][7],cf3,cf15);
    int sf16 = fulladdr16_out[0];
    int cf16 = fulladdr16_out[1];
            
    int product_bin[16] = {cf16, sf16, sf15, sf14, sf13, sf12, sf11, sf10, sf9, sf8, sf7, sf6, sf5, sf4, sh5, p[0][0]};
    // int product_bin[16] = {p[0][0],sh5,sf4,sf5,sf6,sf7,sf8,sf9,sf10,sf11,sf12,sf13,sf14,sf15,sf16,cf16};
    // printf("%d\n",product_bin[14]+product_bin[12]);
    float product_dec = 0;
    // printf("%d\n",product_dec);
    // int product_dec = 0;
    for(int i=0;i<16;i++){
        product_dec = product_dec + product_bin[i] * pow(2,(15-i));
    }
    return product_dec;
}




float approx_multiply(float a, float b){
    float a1=a, b1=b;
    if(a<0)
        a1=-a;
    if(b<0)
        b1=-b;

    // int8_t s = a1*32;
    // int8_t w = b1*32;
    // printf("%d %d %d\n",s,w,s*w);
    a = a * 32;
    b = b * 32;
    float result;
    //result = s*w;
    result = approx_multiplier(b, a);

    int sign;
    if (a*b >= 0)
        sign = 1;
    else
        sign = -1;

    return sign*result/(32.0*32.0);
}


float posit_fixmultiply(float op1, float op2)  
{
    //Input : Operand of multiplication
    //Ouput : Result of multiplication in IEEE-754 format
    //Flow  : The input operands are in IEEE-754 format. We convert them to posit format and then perform a posit multiplication.
    //        The result is obtained in posit format. We then return the result after converting it to IEEE-754 format.  
    int posit_length = 10;
    int es = 2;
    int useed = 4;//2^es
    int regime_len = 2;
    int frc_len = posit_length - es - regime_len - 1 + 1;
    // long int start   = -128;
    int op1_sign,  op1_exp, op2_sign, op2_exp; 
    int posit_sign, posit_k, posit_exp, posit_e;
    float posit_frc, op1_frc_new, op2_frc_new, frc_new;
    float op1_frc, op2_frc;
    float result;


    if(op1 < 0)
    {
        op1_sign = 1;
        op1 = -1*op1;
    }
    else
    {
        op1_sign = 0;
    }
    if(op2 < 0)
    {
        op2_sign = 1;
        op2 = -1*op2;
    }
    else
    {
        op2_sign = 0;
    }
    if(op1 == 0.0)
    {
        result =  0.0;
        return result;
    }
    if(op2 == 0.0)
    {
        result =  0.0;
        return result;
    }

    op1_exp = (int)floor(log2(op1));
    op2_exp = (int)floor(log2(op2));
    // printf("expo %d, %d\n", op1_exp, op2_exp);
    // op1_k   = (int)floor(op1_exp/(float)useed);
    // op2_k   = (int)floor(op2_exp/(float)useed);
    op1_frc = (float)op1/pow(2.0, op1_exp) - 1;
    op2_frc = (float)op2/pow(2.0, op2_exp) - 1;
    // printf(" frac %f, %f\n", op1_frc, op2_frc);
    
    // op2_frc = (float)op2/two_pows[op2_exp - start] - 1;
    op1_frc_new = 1;
    op2_frc_new = 1;
    // op1_e   = op1_exp - op1_k*useed;
    // op2_e   = op2_exp - op2_k*useed;
    posit_sign = op1_sign ^ op2_sign;
    posit_exp = op1_exp + op2_exp;
    // printf("posit sign and posit exponent %d, %d\n", posit_sign, posit_exp);

    for(int i=0; i < frc_len; i++)
    {
        op1_frc = op1_frc*2;
        if(op1_frc >= 1)
        {
            op1_frc_new = op1_frc_new + pow(2.0, -(i+1));
            op1_frc = op1_frc - 1; 
        }
        op2_frc = op2_frc*2;
        if(op2_frc >= 1)
        {
            op2_frc_new = op2_frc_new + pow(2.0, -(i+1));
            op2_frc = op2_frc - 1; 
        }
    }

    frc_new = op1_frc_new * op2_frc_new;

    if(frc_new >=2)
    {
        frc_new = frc_new/2;
        posit_exp++;
    }
    frc_new = frc_new - 1;
    posit_frc = 1;
    for(int i=0; i < frc_len; i++)
    {
        frc_new = frc_new * 2;
        if(frc_new >= 1)
        {
            posit_frc = posit_frc + pow(2.0, -(i+1));
            frc_new = frc_new - 1;
        }
    }
    //posit_k = (int)floor((float)posit_exp/useed);
    posit_k = -1;
    posit_e = posit_exp - posit_k*useed;

    // printf(" posit %d, %d, %f\n", posit_k, posit_e, posit_frc);
    if(posit_sign == 1)
    {
            result =  -1*posit_frc*pow(2.0, useed*posit_k + posit_e);
            // printf("%d", pow(2.0, useed*posit_k + posit_e));
    }
    else
    {
            result = posit_frc*pow(2.0, useed*posit_k + posit_e);
            //  printf("above result %f, %d, %d\n", pow(2.0, useed*posit_k + posit_e), useed, useed*posit_k+posit_e);
    }

    // if(KnobLog)
    // {
    //     fprintf(logfile, " %.14f\n", result);
    // }
    // printf(" result%f\n",result);
    return result;
}

float ieee754multiplier(float op1, float op2){
    int e1,e2 = 0;
    float m1=1,m2=1;
    float m1_new,m2_new,m1_final,m2_final;
    int s1,s2;
    float a,b, ef,sf, mf, mf_new,mf_final;
    float e_max = pow(2,7) - 1;
    float e_min = -1*(pow(2,7));
    if(op1 == 0 || op2 == 0){
        return 0.0;
    }
    if(op1 < 0){
        a = -op1;
        s1 = 1;
    }
    if(op2 < 0){
        b = -op2;
        s2 = 1;
    }
    if(op1 > 0){
        a = op1;
        s1 = 0;
    }
    if(op2 > 0){
        b = op2;
        s2 = 0;
    }
    sf = s1^s2;
    e1 = floor(log2(a));
    if(e1 > e_max){
        e1 = e_max;
    }
    else if(e1 < e_min){
        e1 = e_min;
    }
    e2 = floor(log2(b));
    if(e2 > e_max){
        e2 = e_max;
    }
    else if(e2 < e_min){
        e2 = e_min;
    }
    m1 = a/pow(2.0,e1);
    m2 = b/pow(2.0,e2);
    m1_new = m1 - 1;
    m2_new = m2-1;
    for(int i=1;i<8;i++){
        m1_new-=pow(2,-i);
        if(m1_new<0){
            m1_new+=pow(2,-i);
        }
    }
    m1_final = m1 - m1_new;

    for(int i=1;i<8;i++){
        m2_new-=pow(2,-i);
        if(m2_new<0){
            m2_new+=pow(2,-i);
        }
    }
    m2_final = m2 - m2_new;
    if(s1 + e1 + m1_final == 0 || s2 + e2 + m2_final == 0){
        return 0.0;
    }
    ef = e1 + e2;
    mf = approx_multiply(m1_final, m2_final);
    if(mf>=2){
        mf = mf/2;
        ef = ef + 1;
    }
    if(ef > e_max){
        ef = e_max;
    }
    else if(ef < e_min){
        ef = e_min;
    }
    mf_new = mf - 1;
    for(int i=1;i<8;i++){
        mf_new-=pow(2,-i);
        if(mf_new<0){
            mf_new+=pow(2,-i);
        }
    }
    mf_final = mf - mf_new;
    if(sf == 1){
        return (-1*(mf_final)*pow(2,ef));
    }
    else{
        return ((mf_final)*pow(2,ef));
    }
}
/*float fp_weight_error(float weight, int bit_position){
        
    int fp_length = 16;
    int es = 8;
    int frc_len = fp_length - es - 1;
    // long int start   = -128;
    int op_sign,  op_exp; 
    float op_frc_new;
    float op_frc;
    float result;


    if(weight < 0)
    {
        op_sign = 1;
        weight = -1*weight;
    }
    else
    {
        op_sign = 0;
    }

    if(bit_position == 0){
        if(op_sign == 0){
            op_sign = 1;
        } else{
            op_sign = 0;
        }
    }


    if(weight == 0.0)
    {
        result =  0.0;
        return result;
    }


    op_exp = (int)floor(log2(weight));


    op_frc = (float)weight/pow(2.0, op_exp)-1;

    // printf(" frac %f, %f\n", op1_frc, op2_frc);
    
    // op2_frc = (float)op2/two_pows[op2_exp - start] - 1;
    op_frc_new = 1;
    // op1_e   = op1_exp - op1_k*useed;
    // op2_e   = op2_exp - op2_k*useed;
    // posit_sign = op1_sign ^ op2_sign;
    // posit_exp = op1_exp + op2_exp;
    // printf("posit sign and posit exponent %d, %d\n", posit_sign, posit_exp);

    for(int i=0; i < frc_len; i++)
    {
        op_frc = op_frc*2;
        
        if(op_frc >= 1)
        {
            if(bit_position-es-1 != i){
                op_frc_new = op_frc_new + pow(2.0, -(i+1));
                op_frc = op_frc - 1;
            }
             
        }
        if(bit_position - es-1 == i && op_frc <1){
            op_frc_new = op_frc_new + pow(2.0, -(i+1));
            op_frc = op_frc - 1;
        }
    }

    //op_frc_new = op_frc_new - 1;


    // printf(" posit %d, %d, %f\n", posit_k, posit_e, posit_frc);
    if(op_sign == 1)
    {
            result =  -1*op_frc_new*pow(2.0,op_exp);
            // printf("%d", pow(2.0, useed*posit_k + posit_e));
    }
    else
    {
            result = op_frc_new*pow(2.0,op_exp);
            //  printf("above result %f, %d, %d\n", pow(2.0, useed*posit_k + posit_e), useed, useed*posit_k+posit_e);
    }
    return result;
}*/

float weight_error(float weight, int bit_position){
        
    int posit_length = 6;
    int es = 2;
    int useed = 4;//2^es
    int regime_len = 2;
    int frc_len = posit_length - es - regime_len - 1 + 1;
    // long int start   = -128;
    int op_sign,  op_exp; 
    int posit_sign, posit_k, posit_exp, posit_e;
    float op_frc_new;
    float op_frc;
    float result;


    if(weight < 0)
    {
        op_sign = 1;
        weight = -1*weight;
    }
    else
    {
        op_sign = 0;
    }

    if(bit_position == 0){
        if(op_sign == 0){
            op_sign = 1;
        } else{
            op_sign = 0;
        }
    }


    if(weight == 0.0)
    {
        result =  0.0;
        return result;
    }


    op_exp = (int)floor(log2(weight));

    op_frc = (float)weight/pow(2.0, op_exp) - 1;
    // printf(" frac %f, %f\n", op1_frc, op2_frc);
    
    // op2_frc = (float)op2/two_pows[op2_exp - start] - 1;
    op_frc_new = 1;
    // op1_e   = op1_exp - op1_k*useed;
    // op2_e   = op2_exp - op2_k*useed;
    // posit_sign = op1_sign ^ op2_sign;
    // posit_exp = op1_exp + op2_exp;
    // printf("posit sign and posit exponent %d, %d\n", posit_sign, posit_exp);

    for(int i=0; i < frc_len; i++)
    {
        op_frc = op_frc*2;
        
        if(op_frc >= 1)
        {
            if(bit_position-es-regime_len-1 != i){
                op_frc_new = op_frc_new + pow(2.0, -(i+1));
                op_frc = op_frc - 1;
            }
             
        }
        if(bit_position - es-regime_len-1 == i && op_frc <1){
            op_frc_new = op_frc_new + pow(2.0, -(i+1));
            op_frc = op_frc - 1;
        }
    }

    //op_frc_new = op_frc_new - 1;
    posit_k = (int)floor((float)op_exp/useed);
    posit_e = op_exp - posit_k*useed;

    // printf(" posit %d, %d, %f\n", posit_k, posit_e, posit_frc);
    if(op_sign == 1)
    {
            result =  -1*op_frc_new*pow(2.0, useed*posit_k + posit_e);
            // printf("%d", pow(2.0, useed*posit_k + posit_e));
    }
    else
    {
            result = op_frc_new*pow(2.0, useed*posit_k + posit_e);
            //  printf("above result %f, %d, %d\n", pow(2.0, useed*posit_k + posit_e), useed, useed*posit_k+posit_e);
    }
    return result;
}

void convol(float *out, float *weight, float *kernel_shape, float *x, float *x_shape, int outplanes, int h_out, int w_out, int stride, int kernel_size, float *bias)
{

    int width = (int)(kernel_size/2);

    int ker_size = (int)(kernel_shape[0]*kernel_shape[1]*kernel_shape[2]);
    int x_shape_0 = (int)x_shape[0];
    int x_shape_1 = (int)x_shape[1];
    int x_shape_2 = (int)x_shape[2];
    int x_shape_3 = (int)x_shape[3];
    int outplane;

    int num_threads = 64;
    omp_set_num_threads(num_threads);
    int chunk = outplane/num_threads;
    float ker[ker_size];
    float xx[ker_size];
    int count = 0;
    int r_o = 0;
    int c_o = 0;
    float dot_pro = 0;
    float dot_pro1 = 0;
    int bit_error = 0;
    int bit_position = 0;
    float bit_error_rate = 0;

    srand(time(0));

    #pragma omp parallel shared(ker_size,weight,outplanes,width,x_shape_1,x_shape_2,x_shape_3,x_shape_0,h_out,w_out,x,out,stride,bias) private(outplane,ker,xx,count,r_o,c_o,dot_pro, bit_position, bit_error)
    {

        #pragma omp for schedule(static) nowait
        for(outplane=0;outplane<outplanes;outplane++){
            count = 0;
            for(int r=width; r<x_shape_1-width; r = r+stride){
                for(int c=width; c<x_shape_2-width; c=c+stride){
                    count = 0;
                    dot_pro = 0;
                    for(int inplane=0; inplane<x_shape_0; inplane++){
                        for(int rr = r-width; rr<r+width+1; rr++){
                            for(int cc = c-width; cc<c+width+1; cc++){
                                /*bit_error = rand();
                                if(bit_error<RAND_MAX*bit_error_rate){
                                    bit_error = 1;
                                } else{
                                    bit_error = 0;
                                }

                                if(bit_error==1){
                                    bit_position = rand();
                                    bit_position = (int)bit_position*posit_length/RAND_MAX;
                                    *(weight) = weight_error(*weight, bit_position);

                                }*/
                                // float *cc_temp = cc;
                                 //printf("%f\n",*weight);
                                // inp1 = *((weight+ outplane*ker_size)+count);
                                //float inp2 = *(((x+inplane*x_shape_1*x_shape_2)+rr*x_shape_2)+cc);
                                // printf("%f, %f, %f\n", *(x+inplane*x_shape_1*x_shape_2), *((x+inplane*x_shape_1*x_shape_2)+rr*x_shape_2), *(((x+inplane*x_shape_1*x_shape_2)+rr*x_shape_2)+cc));
                                // dot_pro1 = posit_multiply(inp1,inp2);
                                // dot_pro = dot_pro+dot_pro1;
                                // dot_pro = (dot_pro) + posit_fixmultiply(*((weight+ outplane*ker_size)+count),(*(((x+inplane*x_shape_1*x_shape_2)+rr*x_shape_2)+cc)));
                                dot_pro = (dot_pro) + posit_fixmultiply(*((weight+ outplane*ker_size)+count),(*(((x+inplane*x_shape_1*x_shape_2)+rr*x_shape_2)+cc)));
                                // dot_pro = (dot_pro) + ieee754multiplier(((weight+ outplane*ker_size)+count),((((x+inplane*x_shape_1*x_shape_2)+rr*x_shape_2)+cc)));
                                // printf("%f\n", dot_pro);
                                
                               //dot_pro = dot_pro+ *((weight+ outplane*ker_size)+count)*(*(((x+inplane*x_shape_1*x_shape_2)+rr*x_shape_2)+cc));
								count = count+1;

                            }
                        }
                    }
                    r_o = (int)((r-width)/stride);
                    c_o = (int)((c-width)/stride);

                    *(((out+outplane*h_out*w_out)+r_o*w_out)+c_o) = dot_pro+ *(bias+outplane);

                }
            }
            
        }
    }
}
