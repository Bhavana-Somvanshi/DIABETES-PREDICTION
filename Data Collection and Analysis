#Data Collection and Analysis:

# loading the dataset
df = pd.read_csv('diabetes.csv')
df.head()


df.shape
(768, 9)
df['Outcome'].value_counts()


#Checking the Dataset:
df.describe()
df.info()

#Replacing the zeroes with mean:
df_mean = df[zero_features].mean()
df[zero_features] = df[zero_features].replace(0, df_mean)
df.describe()

#Checking the Correlation between the various features:
plt.figure(figsize=(13,10))
sns.heatmap(df.corr(), annot = True)

plt.figure(figsize=(13,6))
sns.distplot(df["Pregnancies"][df["Outcome"]==1], color = "red", label = "Positive", kde_kws={'shade' : True}, hist = False)
sns.distplot(df["Pregnancies"][df["Outcome"]==0], color = "green", label = "Negative", kde_kws={'shade' : True}, hist = False)
plt.legend()

plt.figure(figsize=(13,6))
sns.distplot(df["Glucose"][df["Outcome"]==1], color = "red", label = "Positive", kde_kws={'shade': True}, hist = False)
sns.distplot(df["Glucose"][df["Outcome"]==0], color = "green", label = "Negative", kde_kws={'shade': True}, hist = False)
plt.legend()

plt.figure(figsize=(13,6))
sns.distplot(df["BloodPressure"][df["Outcome"]==1],color="red",label="Positive", kde_kws={'shade': True}, hist = False)
sns.distplot(df["BloodPressure"][df["Outcome"]==0],color="green",label="Negative", kde_kws={'shade': True}, hist = False)
plt.legend()

plt.figure(figsize=(13,6))
sns.distplot(df["Age"][df["Outcome"]==1],color="red",label="Positive", kde_kws={'shade': True}, hist = False)
sns.distplot(df["Age"][df["Outcome"]==0],color="green",label="Negative", kde_kws={'shade': True}, hist = False)
plt.legend()


#Checking for Outliers:
plt.figure(figsize=(20,10))
sns.scatterplot(data=df, x="Glucose", y="BMI", hue="Age", size="Age")

sns.catplot(y="BloodPressure",x="Outcome",data=df,kind="box")
plt.ylabel("Blood Pressure")
plt.xlabel("Outcome")

sns.catplot(y="Age",x="Outcome",data=df,kind="box")
plt.ylabel("Age")
plt.xlabel("Outcome")

sns.catplot(y="Glucose",x="Outcome",data=df,kind="box")
plt.ylabel("Glucose")
plt.xlabel("Outcome")

sns.catplot(y="DiabetesPedigreeFunction",x="Outcome",data=df,kind="box")
plt.ylabel("DiabetesPedigreeFunction")
plt.xlabel("Outcome")


##Removing the Outliers to improve the performance of the machine learning algorithms:
def detect_outliers(df,n,features):
outlier_indices = []

# iterate over features(columns)
for col in features:
Q1 = np.percentile(df[col], 25)
Q3 = np.percentile(df[col],75)
IQR = Q3 - Q1

# outlier step
outlier_step = 1.5 * IQR

# Determine a list of indices of outliers for feature col
outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step )].index

# append the found outlier indices for col to the list of outlier indices
outlier_indices.extend(outlier_list_col)

# select observations containing more than 2 outliers
outlier_indices = Counter(outlier_indices)
multiple_outliers = list( k for k, v in outlier_indices.items() if v > n )
return multiple_outliers

# detect outliers from numeric features
outliers_to_drop = detect_outliers(df, 2 ,["Pregnancies", 'Glucose', 'BloodPressure', 'BMI', 'DiabetesPedigreeFunction', 'SkinThickness', 'Insulin', 'Age'])
In [27]:

# Show the outliers rows
df.loc[outliers_to_drop]

df.drop(df.loc[outliers_to_drop].index, inplace=True)

#Seperating the data and labels
X = df.drop(columns = 'Outcome', axis = 1) # axis will be 1 as we are dropping the column
Y = df['Outcome']
X


#Splilting Data into Training and Testing Data:
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2,stratify = Y, random_state = 0)
# There is a chance that all the positive diabetes cases might go to train data and vice-versa
# To prevent that we use the function startify on Y.
# Random State is used to seed the data. Basically, it means that every time I run the program, the data will split the same way. It won't vary.

print(X.shape, X_train.shape, X_test.shape)
(767, 8) (613, 8) (154, 8)
