script  = open("script_posit.sh", "w")

model_name = "resnet18"
dataset_name = "cifar10"
design_name = "posit"

script.write(f"cc -fPIC -shared -o lib_mine.so test_posit.c -lm -fopenmp -std=c99\n")
script.write(f"python3 run_posit.py {model_name} {dataset_name} {design_name}\n")
script.write(f"python3 ../scripts/accuracy.py\n")
script.write(f"echo 'Done : {model_name} on {dataset_name} using {design_name}'\n")

script.close()
