import pandas as pd

model_dataset_pairs = [("resnet18", "cifar10"), ("lenet","mnist")]
attacks = ["fgsm", "pgd", "deepfool", "hsja", "cw2"]
approx_designs = ["approxlptune1", "approxlptune2", "approxlptune3", "cfputune1", "cfputune2", "cfputune3", "cfputune4", "fpcamtune1", "fpcamtune2", "fpcamtune3","fpcamtune4",  "rmactune1", "rmactune2", "rmactune3", "rmactune4", "mbm", "exact"]
#approx_designs = ["exactwoframework"]
#approx_designs = ["exactactual_exactactual"]

for pair in model_dataset_pairs:
    model_name = pair[0]
    dataset_name = pair[1]
    results_file = open(f"../results/successrate_{model_name}_{dataset_name}_tensor.csv", "w")
    header = "Design"
    for attack in attacks:
        header = header + f",{attack}"
    header = header + "\n"
    results_file.write(header)
    for design in approx_designs:
        data = f"{design}"
        for attack in attacks:
            df_exact = pd.read_csv(f"../results/advout_{attack}_{model_name}_{dataset_name}_exact_tensor.csv")
            df1 = pd.read_csv(f"../results/advimgnums_{attack}_{model_name}_{dataset_name}_tensor.csv")
            df2 = pd.read_csv(f"../results/advout_{attack}_{model_name}_{dataset_name}_{design}_tensor.csv")
            assert((df1["AdvP"] == df2["AdvP"]).all())
            df3 = pd.concat([df1,df2["ApxP"]], axis=1)
            to_subtract = 0
            for i in range(len(df3)):
                if (df3["ApxP"][i] !=  df3["AdvP"][i] and df3["ApxP"][i] ==  df3["GT"][i] and df_exact["AdvP"][i] ==  df_exact["ApxP"][i]):
                    to_subtract += 1
            total = len(df_exact[df_exact["AdvP"] ==  df_exact["ApxP"]])
            success_rate = (total - to_subtract)/total
            # succes_rate = (len(df3) - len(df3[(df3["ApxP"] != df3["AdvP"]) & (df3["ApxP"] == df3["GT"])]))/(len(df3))
            print(f"{model_name} {dataset_name} {attack} : {len(df3[(df3['ApxP'] != df3['AdvP']) & (df3['ApxP'] == df3['GT'])])}/{len(df3)}")
            print(f"{model_name} {dataset_name} {attack} : {len(df3[(df3['ApxP'] != df3['AdvP'])])}/{len(df3)}")
            print(f"{model_name} {dataset_name} {attack} : {success_rate}")
            data = data + f",{success_rate*100}"
        data = data + "\n"
        results_file.write(data)
    results_file.close()


