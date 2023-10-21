import pandas as pd 

model_dataset_pairs = [("resnet18", "cifar10"), ("lenet","mnist")]
attacks = ["fgsm", "pgd", "deepfool", "hsja", "cw2"]

for pair in model_dataset_pairs:
    model = pair[0]
    dataset = pair[1]
    for attack in attacks:
        df = pd.read_csv(f"../results/{attack}_{model}_{dataset}_tensor_result.csv")
        df2 = df[(df["GT"] == df["NP"]) & (df["GT"] != df["AdvP"])]
        df2["ImgNum"].to_csv(f"../results/advimgnums_{attack}_{model}_{dataset}_tensor.csv", index = False)
