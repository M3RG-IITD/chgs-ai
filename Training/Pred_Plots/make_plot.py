from shadow.plot import *
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
from yvsyplot import *
add_custom_fonts()

R2_ = pd.read_csv("./R2.csv")
info = json.load(open("./info.json"))
props = list(R2_["Property"])
print("Props: ", props)

for i in props:
    print(i)
    info_ = info[i]
    info_['scale'] = info_['prop_scale']
    info_['label'] = info_['label_name']

    mask = R2_["Property"].values==i
    R2_test = R2_[mask]["Test_R2"].values[0]
    R2_train = R2_[mask]["Train_R2"].values[0]
    data_test = pd.read_csv(f"{i}/test_ypr_yac.csv")
    data_train = pd.read_csv(f"{i}/train_ypr_yac.csv")

    y_ac = data_test["y_ac"].values
    y_pr = data_test["y_pr"].values

    rng = [min(y_ac.min(), y_pr.min()) - info_['margin'], max(y_ac.max(), y_pr.max()) + info_['margin'],]
    info_['limits'] = rng

    fig = make_den_plot(info_, y_ac, y_pr, R2_train, R2_test)
    # save2file(f"plots/{i}_denplot.pfig", fig)
    plt.savefig(f"plots/{i}_denplot.png", dpi=500)
    plt.close()
