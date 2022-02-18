# 疾病预测测试



## 数据集



### [Cardio Disease](https://www.kaggle.com/raminhashimzade/cardio-disease)

```
id number 身份证号码
age in days 天数
age in years 年龄
gender (1 - women, 2 - men) 性别（1 - 女性，2 - 男性）
height cm 身高厘米
weight kg 体重公斤
ap_hi (Systolic blood pressure) ap_hi（收缩压）
ap_lo (Diastolic blood pressure) ap_lo（舒张压）
cholesterol (1: normal, 2: above normal, 3: well above normal) 胆固醇（1：正常，2：高于正常，3：远高于正常）
gluc (1: normal, 2: above normal, 3: well above normal) gluc（1：正常，2：高于正常，3：远高于正常）
smoke (whether patient smokes or not(0 = no, 1 = yes) 吸烟（患者是否吸烟（0 = 否，1 = 是）
alco Binary feature (0 = no, 1 = yes) alco 二进制特征（0 = 否，1 = 是）
active Binary feature (0 = passive life, 1 = active life) 主动二元特征（0 = 被动寿命，1 = 主动寿命）
cardio Target variable(0 = no, 1 = yes) 有氧运动目标变量（0 = 否，1 = 是）
```



### [Heart Disease UCI](https://www.kaggle.com/ronitf/heart-disease-uci)

```
age 年龄
sex 性别
chest pain type (4 values) 胸痛类型（4个值）
resting blood pressure 静息血压
serum cholestoral in mg/dl 以 mg/dl 为单位的血清胆甾醇
fasting blood sugar > 120 mg/dl 空腹血糖 > 120 mg/dl
resting electrocardiographic results (values 0,1,2) 静息心电图结果（值 0、1、2）
maximum heart rate achieved 达到的最大心率
exercise induced angina 运动性心绞痛
oldpeak = ST depression induced by exercise relative to rest 运动相对于休息引起的 ST 段压低
the slope of the peak exercise ST segment 峰值运动ST段的斜率
number of major vessels (0-3) colored by flourosopy 荧光染色的主要血管数量（0-3）
thal: 3 = normal; 6 = fixed defect; 7 = reversable defect  3 = 正常； 6 = 固定缺陷； 7 = 可逆缺陷
```



### [Heart Failure Prediction](https://www.kaggle.com/andrewmvd/heart-failure-clinical-data)

```
age - Age 年龄
anaemia - Decrease of red blood cells or hemoglobin (boolean) 红细胞或血红蛋白减少（布尔值）
creatinine_phosphokinase - Level of the CPK enzyme in the blood (mcg/L) 血液中 CPK 酶的水平 (mcg/L)
diabetes - If the patient has diabetes (boolean) 如果患者患有糖尿病（布尔值）
ejection_fraction - Percentage of blood leaving the heart at each contraction (percentage) 每次收缩时离开心脏的血液百分比（百分比）
high_blood_pressure - If the patient has hypertension (boolean) 如果患者患有高血压（布尔值）
platelets - Platelets in the blood (kiloplatelets/mL) 血液中的血小板（千血小板/mL）
serum_creatinine - Level of serum creatinine in the blood (mg/dL) 血液中的血清肌酐水平 (mg/dL)
serum_sodium - Level of serum sodium in the blood (mEq/L) 血液中的血清钠水平 (mEq/L)
sex - Woman or man (binary) 女人或男人（二进制）
smoking - If the patient smokes or not (boolean) 患者是否吸烟（布尔值）
time - Follow-up period (days) 随访期（天）
DEATH_EVENT - If the patient deceased during the follow-up period (boolean) 如果患者在随访期间死亡（布尔值）
```



### [Medical Data](https://www.kaggle.com/karimnahas/medicaldata)

```
id
gender 性别
dob 出生日期
zipcode 邮政编码
employment_status 就业状况
education 教育
marital_status 婚姻状况
children 孩子
ancestry 祖先
avg_commute 平均通勤时间
daily_internet_use 日常互联网使用
available_vehicles 可用车辆
military_service 兵役
disease 疾病
```