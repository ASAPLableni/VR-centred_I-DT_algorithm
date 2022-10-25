import pandas as pd

from IDT_alg_VR_centred import IDTVR

df_et = pd.read_csv("Data/eyetracking.txt", sep=";")

print("Shape", df_et.shape)
print("Columns", df_et.columns)

idt_vr = IDTVR()

df_et_res = idt_vr.fit_compute(df_et, time="elapsedTime")
