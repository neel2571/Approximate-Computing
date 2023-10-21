cc -fPIC -shared -o lib_mine.so test_approxlptune1.c -lm -fopenmp -std=c99
python3 runadv_tensor.py resnet18 cifar10 deepfool approxlptune1
echo 'Done : resnet18 on cifar10 | deepfool | approxlptune1'
cc -fPIC -shared -o lib_mine.so test_approxlptune2.c -lm -fopenmp -std=c99
python3 runadv_tensor.py resnet18 cifar10 deepfool approxlptune2
echo 'Done : resnet18 on cifar10 | deepfool | approxlptune2'
cc -fPIC -shared -o lib_mine.so test_approxlptune3.c -lm -fopenmp -std=c99
python3 runadv_tensor.py resnet18 cifar10 deepfool approxlptune3
echo 'Done : resnet18 on cifar10 | deepfool | approxlptune3'
cc -fPIC -shared -o lib_mine.so test_cfputune1.c -lm -fopenmp -std=c99
python3 runadv_tensor.py resnet18 cifar10 deepfool cfputune1
echo 'Done : resnet18 on cifar10 | deepfool | cfputune1'
cc -fPIC -shared -o lib_mine.so test_cfputune2.c -lm -fopenmp -std=c99
python3 runadv_tensor.py resnet18 cifar10 deepfool cfputune2
echo 'Done : resnet18 on cifar10 | deepfool | cfputune2'
cc -fPIC -shared -o lib_mine.so test_cfputune3.c -lm -fopenmp -std=c99
python3 runadv_tensor.py resnet18 cifar10 deepfool cfputune3
echo 'Done : resnet18 on cifar10 | deepfool | cfputune3'
cc -fPIC -shared -o lib_mine.so test_cfputune4.c -lm -fopenmp -std=c99
python3 runadv_tensor.py resnet18 cifar10 deepfool cfputune4
echo 'Done : resnet18 on cifar10 | deepfool | cfputune4'
cc -fPIC -shared -o lib_mine.so test_fpcamtune1.c -lm -fopenmp -std=c99
python3 runadv_tensor.py resnet18 cifar10 deepfool fpcamtune1
echo 'Done : resnet18 on cifar10 | deepfool | fpcamtune1'
cc -fPIC -shared -o lib_mine.so test_fpcamtune2.c -lm -fopenmp -std=c99
python3 runadv_tensor.py resnet18 cifar10 deepfool fpcamtune2
echo 'Done : resnet18 on cifar10 | deepfool | fpcamtune2'
cc -fPIC -shared -o lib_mine.so test_fpcamtune3.c -lm -fopenmp -std=c99
python3 runadv_tensor.py resnet18 cifar10 deepfool fpcamtune3
echo 'Done : resnet18 on cifar10 | deepfool | fpcamtune3'
cc -fPIC -shared -o lib_mine.so test_fpcamtune4.c -lm -fopenmp -std=c99
python3 runadv_tensor.py resnet18 cifar10 deepfool fpcamtune4
echo 'Done : resnet18 on cifar10 | deepfool | fpcamtune4'
cc -fPIC -shared -o lib_mine.so test_rmactune1.c -lm -fopenmp -std=c99
python3 runadv_tensor.py resnet18 cifar10 deepfool rmactune1
echo 'Done : resnet18 on cifar10 | deepfool | rmactune1'
cc -fPIC -shared -o lib_mine.so test_rmactune2.c -lm -fopenmp -std=c99
python3 runadv_tensor.py resnet18 cifar10 deepfool rmactune2
echo 'Done : resnet18 on cifar10 | deepfool | rmactune2'
cc -fPIC -shared -o lib_mine.so test_rmactune3.c -lm -fopenmp -std=c99
python3 runadv_tensor.py resnet18 cifar10 deepfool rmactune3
echo 'Done : resnet18 on cifar10 | deepfool | rmactune3'
cc -fPIC -shared -o lib_mine.so test_rmactune4.c -lm -fopenmp -std=c99
python3 runadv_tensor.py resnet18 cifar10 deepfool rmactune4
echo 'Done : resnet18 on cifar10 | deepfool | rmactune4'
cc -fPIC -shared -o lib_mine.so test_mbm.c -lm -fopenmp -std=c99
python3 runadv_tensor.py resnet18 cifar10 deepfool mbm
echo 'Done : resnet18 on cifar10 | deepfool | mbm'
cc -fPIC -shared -o lib_mine.so test_exact.c -lm -fopenmp -std=c99
python3 runadv_tensor.py resnet18 cifar10 deepfool exact
echo 'Done : resnet18 on cifar10 | deepfool | exact'
cc -fPIC -shared -o lib_mine.so test_approxlptune1.c -lm -fopenmp -std=c99
python3 runadv_tensor.py lenet mnist deepfool approxlptune1
echo 'Done : lenet on mnist | deepfool | approxlptune1'
cc -fPIC -shared -o lib_mine.so test_approxlptune2.c -lm -fopenmp -std=c99
python3 runadv_tensor.py lenet mnist deepfool approxlptune2
echo 'Done : lenet on mnist | deepfool | approxlptune2'
cc -fPIC -shared -o lib_mine.so test_approxlptune3.c -lm -fopenmp -std=c99
python3 runadv_tensor.py lenet mnist deepfool approxlptune3
echo 'Done : lenet on mnist | deepfool | approxlptune3'
cc -fPIC -shared -o lib_mine.so test_cfputune1.c -lm -fopenmp -std=c99
python3 runadv_tensor.py lenet mnist deepfool cfputune1
echo 'Done : lenet on mnist | deepfool | cfputune1'
cc -fPIC -shared -o lib_mine.so test_cfputune2.c -lm -fopenmp -std=c99
python3 runadv_tensor.py lenet mnist deepfool cfputune2
echo 'Done : lenet on mnist | deepfool | cfputune2'
cc -fPIC -shared -o lib_mine.so test_cfputune3.c -lm -fopenmp -std=c99
python3 runadv_tensor.py lenet mnist deepfool cfputune3
echo 'Done : lenet on mnist | deepfool | cfputune3'
cc -fPIC -shared -o lib_mine.so test_cfputune4.c -lm -fopenmp -std=c99
python3 runadv_tensor.py lenet mnist deepfool cfputune4
echo 'Done : lenet on mnist | deepfool | cfputune4'
cc -fPIC -shared -o lib_mine.so test_fpcamtune1.c -lm -fopenmp -std=c99
python3 runadv_tensor.py lenet mnist deepfool fpcamtune1
echo 'Done : lenet on mnist | deepfool | fpcamtune1'
cc -fPIC -shared -o lib_mine.so test_fpcamtune2.c -lm -fopenmp -std=c99
python3 runadv_tensor.py lenet mnist deepfool fpcamtune2
echo 'Done : lenet on mnist | deepfool | fpcamtune2'
cc -fPIC -shared -o lib_mine.so test_fpcamtune3.c -lm -fopenmp -std=c99
python3 runadv_tensor.py lenet mnist deepfool fpcamtune3
echo 'Done : lenet on mnist | deepfool | fpcamtune3'
cc -fPIC -shared -o lib_mine.so test_fpcamtune4.c -lm -fopenmp -std=c99
python3 runadv_tensor.py lenet mnist deepfool fpcamtune4
echo 'Done : lenet on mnist | deepfool | fpcamtune4'
cc -fPIC -shared -o lib_mine.so test_rmactune1.c -lm -fopenmp -std=c99
python3 runadv_tensor.py lenet mnist deepfool rmactune1
echo 'Done : lenet on mnist | deepfool | rmactune1'
cc -fPIC -shared -o lib_mine.so test_rmactune2.c -lm -fopenmp -std=c99
python3 runadv_tensor.py lenet mnist deepfool rmactune2
echo 'Done : lenet on mnist | deepfool | rmactune2'
cc -fPIC -shared -o lib_mine.so test_rmactune3.c -lm -fopenmp -std=c99
python3 runadv_tensor.py lenet mnist deepfool rmactune3
echo 'Done : lenet on mnist | deepfool | rmactune3'
cc -fPIC -shared -o lib_mine.so test_rmactune4.c -lm -fopenmp -std=c99
python3 runadv_tensor.py lenet mnist deepfool rmactune4
echo 'Done : lenet on mnist | deepfool | rmactune4'
cc -fPIC -shared -o lib_mine.so test_mbm.c -lm -fopenmp -std=c99
python3 runadv_tensor.py lenet mnist deepfool mbm
echo 'Done : lenet on mnist | deepfool | mbm'
cc -fPIC -shared -o lib_mine.so test_exact.c -lm -fopenmp -std=c99
python3 runadv_tensor.py lenet mnist deepfool exact
echo 'Done : lenet on mnist | deepfool | exact'
