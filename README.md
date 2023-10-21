# Approximate-Computing

Directory Structure:
- framework: Contains codebase of Chandan's PyTorch framework
- models   : Trained original models as well as bit-error introduced models used/generated in this project stored in this directory
- scripts  : Contains scripts for running various experiments, contains script to find accuracy also
- tensors  : Contains tensors of generated adversarial images



# Updated Framework
The following files have been added to the framework:
- test_posit.c (Updated) : The types of multipliers that are supported have been increased. You could change the multiplier used to multiply dot_pro in the convolution operation (convol function).

- test_posit_compressor: Includes approximate multiplier designs designed using compressors (8x8 unsigned integer). Change the compressor used in the {approx_multiplier} function wherever it is found. (Ctrl-F would be an easy way).

Includes the following compressors:
-Yang1,Yang2,Yang3,Lin, Strollo1, Strollo2, Momeni, Sabetz, Venka, Akbar1, Akbar2, Ahma, Ranjbar1, Ranjbar2, Ranjbar3

For changing the operand order, you could swap the order inside {approx_multiply} function, where the {approx_multiplier} is being called.


# Steps for determining accuracy:
Before starting, you might need to create a virtual environment and make sure all requirements are imported. As an example, after creating the virtual environment:
 ```bash
 source ~/virtualenv/bin/activate
 ```
 
-> For generating tensors:

Create a folder named `lenet_mnist` or `resnet18_cifar10` inside the {results} directory based on the network you want to run simulations for. Run the following commands:

```bash
cd scripts/
python3 lenet_mnist_posit.py
```
OR
```bash
cd scripts/
python3 resnet18_cifar10_posit.py
```

This will create the tensors required.

Also open `accuracy.py` file and change the name of the csv file where you want to save the actual tensors and the predicted tensors.

After this, go to the {framework} directory. You could use the following command:

```bash
cd ..
cd framework/
```
If you want to run the simulations for accuracy when using a particular multiplier, go to the `test_posit.c` file and make sure that the multiplier is defined. You could change the multiplier when multiplying dot_pro in the {convol} function.

After the required changes are made, open `script_posit.sh` file and change it as follows:

```bash
cc -fPIC -shared -o lib_mine.so {name of C file} -lm -fopenmp -std=c99
python3 run_posit.py {network name} {dataset} posit
python3 ../scripts/accuracy.py
```
If the file names are not changed, 

- {name of C file} == test_posit.c
- {network name} == lenet/resnet18/any other network supported
- {dataset} == mnist/cifar10/imagenet

Finally, run the following command:

```bash
sh script_posit.sh
```

# Approx-Fixed Posit

-> For running simulations using approximate fixed-posit multiplier, you would need to use `posit_fixmultiply` function in the {test_posit_compressor.c} file. Change it in the convolution function.

->For changing the Posit parameters, you could change it at the parameters defined at the beginning of the function.

-> Rest of the steps remain the same.

## Generating adversarial images

To add adversarial noise to images run the files in scripts directory with following format : {model}\_{dataset}\_{attack}\_tensor.py
For example to add adversarial noise of fgsm attack for resnet18 on cifar10, run the commands below:
```bash
cd scripts/
python3 resnet18_cifar10_fgsm_tensor.py
```
Running these scripts will generate tensors of attacked images and stores them in tensors directory. They will also generate a file named {attack}\_{model}\_{dataset}\_tensor\_result.csv in the results directory. These files will have information of the ground truth, prediction before attack and prediction after attack for all the images. This information is useful to identify teh adversarial images since all attacked images might not be adversarial.

Then to identify adversarial images run
```bash
cd scripts/
python3 adv_img_collector_tensor.py
```
Running this script will generate files named advimgnums_{attack}\_{model}\_{dataset}\_tensor.csv in the results directory. These files contain information only for adversarial images.

## Inferencing on adversarial images with approximate hardware
To generate script to perform inference on adversarial images with approximate designs run the following
```bash
cd framework
python3 scriptmaker.py
```

This will generate a `script.sh` file. Run it using
```bash
cd framework
sh script.sh
```

To compute success rate of attacks, run
```bash
cd scripts/
python3 success_rate_script_new.py
```
