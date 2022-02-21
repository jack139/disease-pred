import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import seaborn as sns

# 转入数据
data = pd.read_csv("../datasets/heart_failure_clinical_records_dataset.csv")

# 数据头
data.head()

# 字段信息
data.info()

# 数据统计
data.describe().T

# 目标 DEATH_EVENT 平衡分析
cols= ["#6daa9f","#774571"]
sns.countplot(x= data["DEATH_EVENT"], palette= cols)
plt.show()

# 检查所有特征的相关性
cmap = sns.diverging_palette(275,150,  s=40, l=65, n=9)
corrmat = data.corr()
plt.subplots(figsize=(18,18))
sns.heatmap(corrmat,cmap= cmap,annot=True, square=True)
plt.show()

# 评估年龄分布
plt.figure(figsize=(20,12))
Days_of_week=sns.countplot(x=data['age'],data=data, hue ="DEATH_EVENT",palette = cols)
Days_of_week.set_title("Distribution Of Age", color="#774571")
plt.show()


# 一些非二元特征的 Boxen 和 swarm 图
feature = ["age","creatinine_phosphokinase","ejection_fraction","platelets","serum_creatinine","serum_sodium", "time"]
for i in feature:
    plt.figure(figsize=(8,8))
    sns.swarmplot(x=data["DEATH_EVENT"], y=data[i], color="black", alpha=0.5)
    sns.boxenplot(x=data["DEATH_EVENT"], y=data[i], palette=cols)
    plt.show()


# 时间和年龄的 kdeplot
sns.kdeplot(x=data["time"], y=data["age"], hue =data["DEATH_EVENT"], palette=cols)
plt.show()


# 数据处理

# 拆分 X 和 y
X=data.drop(["DEATH_EVENT"],axis=1)
y=data["DEATH_EVENT"]

# 特征进行标准缩放
col_names = list(X.columns)
s_scaler = preprocessing.StandardScaler()
X_df= s_scaler.fit_transform(X)
X_df = pd.DataFrame(X_df, columns=col_names)   
#X_df.describe().T

# 查看处理后的数据
colours =["#774571","#b398af","#f1f1f1" ,"#afcdc7", "#6daa9f"]
plt.figure(figsize=(20,10))
sns.boxenplot(data = X_df,palette = colours)
plt.xticks(rotation=90)
plt.show()

###---------第 2 个-------------------------------------------------------------------------

# 使用已有数据
train_df = data


fig, ax = plt.subplots()
ax.boxplot([train_df.platelets, train_df.creatinine_phosphokinase])
plt.xticks([1,2],['platelets','creatinine_phosphokinase'])
plt.legend()
plt.show()

# 作为描述和箱线图的结果，血小板的最大值比其他特征显示的值太高，因此通过缩放器进行缩放。
fix_features = pd.concat([train_df.platelets],axis=1)
scaler = MinMaxScaler(feature_range=(0,1))
fix_features = pd.DataFrame(scaler.fit_transform(fix_features))
df = pd.concat([train_df,fix_features],axis=1)
df.drop(['platelets'],axis=1,inplace=True)
df = df.rename(columns={0:'creatinine_phosphokinase',1:'platelets'})

# 各特征 范围
train_df.hist(figsize=(15,10))
plt.show()

# DEATH_EVENT 与 age 关系
f,ax = plt.subplots(1,1,figsize=(15,10))
sns.countplot('age',hue='DEATH_EVENT',data=train_df,ax=ax)
plt.show()

# DEATH_EVENT 是否平衡
df.DEATH_EVENT.value_counts(normalize=True).plot(kind='bar')
print(df.DEATH_EVENT.value_counts(normalize=True)*100)
plt.show()

# 与 smoking 关系
pd.crosstab(train_df.smoking,train_df.DEATH_EVENT).plot(kind='bar')
plt.ylabel('Death num')
plt.xlabel('Smoking')
plt.title('Smoking and Death')
plt.xticks(ticks=(0,1),labels=['None smoking','Smoking'])
plt.show()

# 特征直接关系 热图
corr_data = df[df.keys()]
k = len(train_df.keys())
cols = corr_data.corr().nlargest(k,'DEATH_EVENT')['DEATH_EVENT'].index
color_map = plt.cm.PuBu
cm = np.corrcoef(df[cols].values.T)
f,ax = plt.subplots(figsize=(14,12))
plt.title('Correlation with Death event')
heatmap = sns.heatmap(cm, vmax=1, linewidths=0.1,square=True,annot=True,cmap=color_map, linecolor="white",xticklabels = cols.values ,yticklabels = cols.values)
plt.show()

# 与 age 关系
f,ax = plt.subplots(figsize=(14,12))
sns.histplot(x='age',data=train_df,hue='DEATH_EVENT',kde=True)
plt.show()

# 与 time 关系
f,ax = plt.subplots(figsize=(14,12))
sns.histplot(x='time',data=train_df,hue='DEATH_EVENT',kde=True)
plt.ylabel('Num people')
plt.show()


###---------第 3 个-------------------------------------------------------------------------

df_heart = data

# 与 age 分布关系
sns.FacetGrid(df_heart,hue='DEATH_EVENT',height=8).map(sns.distplot,'age').set_axis_labels('age','DEATH_EVENT').add_legend()
plt.show()

# 与  anaemia 关系
sns.FacetGrid(df_heart,hue='DEATH_EVENT',height=8).map(sns.distplot,'anaemia').set_axis_labels('anaemia', 'DEATH_EVENT').add_legend()
plt.show()

# 异常值 可视化

# age
sns.boxplot(x = df_heart.age, color = 'black')
plt.show()

# creatinine_phosphokinase
sns.boxplot(x = df_heart.creatinine_phosphokinase, color = 'black')
plt.show()

# ejection_fraction
sns.boxplot(x = df_heart.ejection_fraction, color = 'black')
plt.show()

# 在这里我们发现它检测到异常值，所以我们不得不删除它们以免分散数据
df_heart = df_heart[df_heart['ejection_fraction']<70]

sns.boxplot(x = df_heart.ejection_fraction, color = 'black')
plt.show()

# 特征交互关系 热图
corrmat = df_heart.corr()
f, ax = plt.subplots(figsize=(20, 9))
sns.heatmap(corrmat, vmax=.8, annot=True)
plt.show()
