import numpy as np
from pandas import read_csv
import pandas as pd
import math

# Load dataset
url = "https://raw.githubusercontent.com/lauradiosan/AI-2019-2020/master/exam/4/tshirts.csv"
dataset = read_csv(url, header=0)

df = pd.DataFrame(dataset)

cols = [1, 2] #coloanele care ne intereseaza (densitate si alcool)
df = df[df.columns[cols]]

if sum(df["femaleTshirts"]) > sum(df["maleTshirts"]):
    print("female")
else:
    print("male")

