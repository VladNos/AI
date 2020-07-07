import numpy as np
from pandas import read_csv
import pandas as pd
import math

# Load dataset
url = "https://raw.githubusercontent.com/lauradiosan/AI-2019-2020/master/exam/5/wine.csv"
dataset = read_csv(url, header=0)

df = pd.DataFrame(dataset)

cols = [7, 10] #coloanele care ne intereseaza (densitate si alcool)
df = df[df.columns[cols]]

media_alcool = df.mean()["alcohol"]
print(media_alcool)

num = 0
sum = 0.0

for densitate, alcool in df.values:
    if alcool > media_alcool:
        sum+=densitate
        num+=1

print(sum/num)

