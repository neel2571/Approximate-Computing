cc -fPIC -shared -o lib_mine.so test_posit.c -lm -fopenmp -std=c99
nohup python3 run.py &> nohup_vgg19_posit_12_4_2.out
# python3 ../scripts/accuracy.py
echo 'Done : resnet18 on cifar10 using posit'
