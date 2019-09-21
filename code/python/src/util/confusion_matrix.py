import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sn

def replace_labels(v):
    if v==0:
        return "Advocate"
    elif v==1:
        return "IHP"
    elif v==2:
        return "OHP"
    elif v==3:
        return "Other"
    elif v==4:
        return "Patient"
    elif v==5:
        return "Researcher"

in_file_name = "/home/zz/Work/msm4phi/output/classifier/raw/oir/scnn/predictions-dnn-stakeholdercls_dnn_text+autodictext_.csv"

df = pd.read_csv(in_file_name, header=0, delimiter=",", quoting=0, encoding="utf-8",
                 )
df1 = df.as_matrix()

header = list(df.columns.values)

gold = []
pred = []

for i in range(len(df1)):
    df1_row = list(df1[i])
    p = df1_row[0]
    pred.append(replace_labels(p))
    t = df1_row[1]
    gold.append(replace_labels(t))

class_labels=["Advocate", "IHP", "OHP","Other","Patient","Researcher"]
cm=confusion_matrix(gold, pred, labels=class_labels)

df_cm = pd.DataFrame(cm, index=class_labels, columns=class_labels)
#plt.figure(figsize = (10,7))
sn.set(font_scale=1.2)#for label size
plot=sn.heatmap(df_cm, annot=True,cmap='Blues',annot_kws={"size": 12},fmt='g')# font size
plot.set_yticklabels(plot.get_yticklabels(), rotation = 0)
fig = plot.get_figure()
fig.savefig("confusion_matrix.png")

print("done")