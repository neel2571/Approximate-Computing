#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <math.h>
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


unsigned int posit_length = 8;
unsigned int es = 4;           
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





float posit_fixmultiply(float op1, float op2)  
{
    //Input : Operand of multiplication
    //Ouput : Result of multiplication in IEEE-754 format
    //Flow  : The input operands are in IEEE-754 format. We convert them to posit format and then perform a posit multiplication.
    //        The result is obtained in posit format. We then return the result after converting it to IEEE-754 format.  
    int posit_length = 9;
    int es = 4;
    int useed =16;//2^es
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
    frc_new = op1_frc_new*op2_frc_new;
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

    #pragma omp parallel shared(ker_size,weight,outplanes,width,x_shape_1,x_shape_2,x_shape_3,x_shape_0,h_out,w_out,x,out,stride,bias) private(outplane,ker,xx,count,r_o,c_o,dot_pro)
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
                                // float *cc_temp = cc;
                                // printf("%f\n",*weight);
                                // float inp1 = *((weight+ outplane*ker_size)+count);
                                // float inp2 = *(((x+inplane*x_shape_1*x_shape_2)+rr*x_shape_2)+cc);
                                // printf("%f, %f, %f\n", *(x+inplane*x_shape_1*x_shape_2), *((x+inplane*x_shape_1*x_shape_2)+rr*x_shape_2), *(((x+inplane*x_shape_1*x_shape_2)+rr*x_shape_2)+cc));
                                // dot_pro1 = posit_multiply(inp1,inp2);
                                // dot_pro = dot_pro+dot_pro1;
                                dot_pro = dot_pro + posit_fixmultiply(*((weight+ outplane*ker_size)+count),*(((x+inplane*x_shape_1*x_shape_2)+rr*x_shape_2)+cc));
                                // printf("%f, %f, %f\n", inp1, inp2, dot_pro);
                                
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
