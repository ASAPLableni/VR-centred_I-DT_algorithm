import pandas as pd

from IDT_alg_VR_centred import IDT_VR

df_et = pd.read_csv("Data/eyetracking.txt", sep=";")

print("Shape", df_et.shape)
print("Columns", df_et.columns)
df_et_res = IDT_VR(df_et, time="elapsedTime")
