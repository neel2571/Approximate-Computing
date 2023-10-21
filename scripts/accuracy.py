import pandas as pd
import numpy as np

df1 = pd.read_csv(f"../results/posit_lenet_mnist_tensor_result_be.csv")

df2 = pd.read_csv(f"../results/positout_biterror_lenet_mnist_test_posit_tensor.csv")

gt = df1[' GT'].tolist()
np = df1[' NP'].tolist()
pp = df2['ApxP'].tolist()

count1 = 0
count2 = 0

for i in range(len(gt)):
    if gt[i] == np[i]:
        count1  +=1
    if gt[i] == pp[i]:
        count2 +=1

print("Without Posit: ", count1/100, "With Posit: ", count2/100)
