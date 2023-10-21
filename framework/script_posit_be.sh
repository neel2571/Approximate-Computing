cc -fPIC -shared -o lib_mine.so test_posit.c -lm -fopenmp -std=c99

cd ..
cd bit_error
python3 weights_extract.py 0.0001
cd ..
cd scripts
python3 lenet_mnist_posit.py
cd ..
cd framework
python3 run_posit.py lenet mnist test_posit
cd ..
cd scripts
python3 accuracy.py

cd ..
cd bit_error
python3 weights_extract.py 0.0002
cd ..
cd scripts
python3 lenet_mnist_posit.py
cd ..
cd framework
python3 run_posit.py lenet mnist test_posit
cd ..
cd scripts
python3 accuracy.py

cd ..
cd bit_error
python3 weights_extract.py 0.0003
cd ..
cd scripts
python3 lenet_mnist_posit.py
cd ..
cd framework
python3 run_posit.py lenet mnist test_posit
cd ..
cd scripts
python3 accuracy.py

cd ..
cd bit_error
python3 weights_extract.py 0.0005
cd ..
cd scripts
python3 lenet_mnist_posit.py
cd ..
cd framework
python3 run_posit.py lenet mnist test_posit
cd ..
cd scripts
python3 accuracy.py

cd ..
cd bit_error
python3 weights_extract.py 0.001
cd ..
cd scripts
python3 lenet_mnist_posit.py
cd ..
cd framework
python3 run_posit.py lenet mnist test_posit
cd ..
cd scripts
python3 accuracy.py

cd ..
cd bit_error
python3 weights_extract.py 0.005
cd ..
cd scripts
python3 lenet_mnist_posit.py
cd ..
cd framework
python3 run_posit.py lenet mnist test_posit
cd ..
cd scripts
python3 accuracy.py

echo 'Done : lenet on mnist using posit'
