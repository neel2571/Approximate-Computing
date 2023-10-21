cc -fPIC -shared -o lib_mine.so bit_error_test_posit.c -lm -fopenmp -std=c99
python3 run_posit.py lenet mnist test_posit
# python3 ../scripts/accuracy.py
echo 'Done : lenet on mnist using posit'
