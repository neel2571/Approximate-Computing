script = open("script.sh", "w")

model_dataset_pairs = [("resnet18", "cifar10"), ("lenet","mnist")]
attacks = [ "deepfool" ]
approx_designs = ["approxlptune1", "approxlptune2", "approxlptune3", "cfputune1", "cfputune2", "cfputune3", "cfputune4", "fpcamtune1", "fpcamtune2", "fpcamtune3", "fpcamtune4",  "rmactune1", "rmactune2", "rmactune3", "rmactune4", "mbm", "exact"]
#approx_designs = ["approxlptune2"]

for pair in model_dataset_pairs:
    model_name = pair[0]
    dataset_name = pair[1]
    for attack in attacks:
        for design in approx_designs:
            script.write(f"cc -fPIC -shared -o lib_mine.so test_{design}.c -lm -fopenmp -std=c99\n")
            script.write(f"python3 runadv_tensor.py {model_name} {dataset_name} {attack} {design}\n")
            script.write(f"echo 'Done : {model_name} on {dataset_name} | {attack} | {design}'\n")

script.close()
