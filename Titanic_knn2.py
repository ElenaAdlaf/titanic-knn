# Titanic-knn.py
# Predicts the survival of Titanic passengers based on the training and test data
# found in the titanic datasets on kaggle.com.
# Elena Adlaf
# Version 1.6, 10/16/17

# Import necessary modules.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Load the csv file containing Titanic training set into a Pandas dataframe, setting empty values as NaN.
traindata = pd.read_csv('E:/Coding/Practice datasets/Titanic/train.csv', na_values='Nothing')

# Explore the data by printing the head of the dataset.
print('Column names and data of first twenty passengers - training set:\n{}'.format(traindata.head(20)))

# Look at the number of missing values in each column
def missing_values(x):
    return sum(x.isnull())
print('Missing values per column:\n{}'.format(traindata.apply(missing_values, axis=0))) # axis=0 applies to columns,
# axis=1 applies to rows

# Specify the columns to be included in the analysis. PassengerID, Name, Ticket and Cabin were excluded because the values
# are unique to each row and cannot be categorized. SibSp and Parch have very few filled values, and
# Age has a large proportion of missing ('NaN') values that would not be productive to impute, so these will also be excluded.
# This leaves the target, Survived, and features Pclass, Sex, Fare, and Embarked.
included_columns = ['Survived', 'Pclass', 'Sex', 'Fare', 'Embarked']
df = traindata[included_columns]

# Preprocess data: Note that most classifier algorithms require all features to be continuous and free of missing values.

# Look at the 2 rows with NaN values in 'Embarked'
embarked_null = df[df['Embarked'].isnull()]
print('Passengers whose embarkation location is blank:\n{}'.format(embarked_null))

# Seeing that the two passengers have identical Pclass, Sex and Fare values, we can deduce the Embarked values
# based on other passengers who payed fares between $75 and $85 (all fares in range are 1st class). Impute the missing
# Embarked data with the most common port of departure among this subset of passengers.
fare_75to85 = df[np.logical_and(df['Fare']>75.0, df['Fare']<85.0)]
deduce_embark = fare_75to85['Embarked'].mode()[0]
print('Other passengers with similar fare most frequently embarked from port: {}'.format(deduce_embark))
df['Embarked'].fillna(deduce_embark, inplace=True)

# Confirm that the values were filled correctly
print('Embarked NaN imputed:\n{}'.format(df[df['Fare']==80.0]))
print('Missing values per column with Embarked NaN imputed:\n{}'.format(df.apply(missing_values, axis=0)))

# Encode categorical features (Pclass, Sex, Embarked) numerically by converting to dummy variables. This will create two more
# columns for Pclass, one more for Sex and 2 more for Embarked, giving a total of 9 feature columns filled with 1 for yes
# and 0 for no.
df = pd.get_dummies(df, columns=['Pclass', 'Sex', 'Embarked'])
print('Dataframe with categorical features expressed numerically:\n{}'.format(df.head()))

# Create numpy arrays for the features and target variables
y = df['Survived'].values
X = df.drop('Survived', axis=1).values
print('Shape of target data: {}'.format(y.data.shape))
print('Shape of features data: {}'.format(X.data.shape))

# Hyperparameter tuning: choosing the best k for k-Nearest Neighbors algorithm using cross-validation
knn = KNeighborsClassifier()
param_grid = {'n_neighbors': np.arange(1,50)}
knn_cv = GridSearchCV(knn, param_grid, cv=10)
knn_cv.fit(X,y)
print('Best k value: {}'.format(knn_cv.best_params_))
print('Best score: {}'.format(knn_cv.best_score_))

# Test the accuracy of the K-Nearest Neighbors classifier algorithm on the training data and show prediction
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=21)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print('Accuracy of k-NN test: {}'.format(knn.score(X_test, y_test)))
print('k-Nearest Neighbors training set survival predictions:\n{}'.format(y_pred))

# Show the predicted confusion matrix to report the accuracy, precision, recall and F1 score of the classifier.
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# Prepare the test dataset
# Load the csv file containing Titanic test set into a Pandas dataframe, setting empty values as NaN.
testdata = pd.read_csv('E:/Coding/Practice datasets/Titanic/test.csv', na_values='Nothing')

# Explore the data by printing the head of the dataset.
print('Column names and data of first twenty passengers - test set:\n{}'.format(testdata.head(20)))

# Look at the number of missing values in each column
print('Missing values per column:\n{}'.format(testdata.apply(missing_values, axis=0))) # axis=0 applies to columns,
# axis=1 applies to rows

# Specify the columns to be included in the analysis. They are the same as in the training data, Pclass, Sex,
# Fare and Embarked, excluding the target column Survived.
included_columns2 = ['Pclass', 'Sex', 'Fare', 'Embarked']
df2 = testdata[included_columns2]

# Look at the 1 row with NaN values in 'Fare'
fare_null = df2[df2['Fare'].isnull()]
print('Passenger with blank fare:\n{}'.format(fare_null))

# Seeing that the passenger is a third class male who embarked from S, we can deduce the fare value based on other
# passengers who were third class and S. Impute the missing data with the median Fare among this subset of passengers.
pclass3_s = df2[np.logical_and(df2['Pclass']==3, df2['Embarked']=='S')]
deduce_fare = pclass3_s['Fare'].median()
print('Other third-class passengers departing from Southampton had a median fare of: {}'.format(deduce_fare))
df2['Fare'].fillna(deduce_fare, inplace=True)

# Confirm that the values were filled correctly
print('Fare NaN imputed:\n{}'.format(df2.iloc[[152]]))
print('Missing values per column with Fare NaN imputed:\n{}'.format(df2.apply(missing_values, axis=0)))

# Encode categorical features (Pclass, Sex, Embarked) numerically by converting to dummy variables.
df2 = pd.get_dummies(df2, columns=['Pclass', 'Sex', 'Embarked'])
print('Test dataframe with categorical features expressed numerically:\n{}'.format(df2.head()))

# Create a numpy array for the test data features
X_new = df2.values
print('Shape of test set features data: {}'.format(X_new.data.shape))

# Fit the optimized knn classifier algorithm to the training data and predict on the test data.
knn.fit(X,y)
pred_survivors = knn.predict(X_new)
print('kNN survival predictions:\n{}'.format(pred_survivors))

# Add the predicted survival data into the test dataframe in a newly-created 'Survived' column.
df2['Survived'] = pred_survivors

# Combine the training and test dataframes into one.
df3 = df.append(df2, ignore_index=True)
print('Confirm type of combined dataframes: {}'.format(type(df3)))
print('Shape of final dataframe: {}'.format(df3.values.data.shape))
print('First five observations of final dataframe:\n{}'.format(df3.head()))

# Create variables to plot the survival data.
lived = df3[df3['Survived']==1]
died = df3[df3['Survived']==0]
total_lived = lived['Survived'].count()
total_died = died['Survived'].count()
print('Total living: {}'.format(total_lived))
print('Total dying: {}'.format(total_died))

# Create variables to plot the fare data.
capfarelived = lived[lived['Fare']<150]
capfaredied = died[died['Fare']<150]
fares_lived = capfarelived['Fare']
fares_died = capfaredied['Fare']
print('Number of fares ($0-150) who lived: {}'.format(fares_lived.count()))
print('Number of fares ($0-150) who died: {}'.format(fares_died.count()))

def ratio_survived(x):
    total = len(df3[df3[x]==1])
    total_lived = len(df3[np.logical_and(df3[x]==1, df3['Survived']==1)])
    return total_lived/total

def get_sex_ratios():
    male_ratio = ratio_survived('Sex_male')
    female_ratio = ratio_survived('Sex_female')
    return male_ratio, female_ratio

def get_class_ratios():
    first_ratio = ratio_survived('Pclass_1')
    second_ratio = ratio_survived('Pclass_2')
    third_ratio = ratio_survived('Pclass_3')
    return first_ratio, second_ratio, third_ratio

def get_port_ratios():
    C_ratio = ratio_survived('Pclass_1')
    Q_ratio = ratio_survived('Pclass_2')
    S_ratio = ratio_survived('Pclass_3')
    return C_ratio, Q_ratio, S_ratio

# Show all graphs on one sheet.
fig, axes = plt.subplots(nrows=2, ncols=2)
ax0, ax1, ax2, ax3 = axes.flatten()

# Plot an unfilled stack step histogram of fares for survivors and non
data_fare = [fares_lived, fares_died]
legend_survive = ['Lived', 'Died']
ax0.hist(data_fare, bins=10, histtype='step', stacked=True, fill=False, label=legend_survive)
ax0.legend(prop={'size': 10})
ax0.set_title('Distribution of fares (< $150)')

# Plot a bar graph showing survival rate by sex.
bar_position_sex = [1,2]
data_sex = get_sex_ratios()
labels_sex = ['Male', 'Female']
ax1.bar(bar_position_sex, data_sex, tick_label=labels_sex, color='mediumorchid')
ax1.set_title('Survival rate by sex')

# Plot a bar graph showing survival rate by ticket class.
bar_position_class = [1,2,3]
data_class = get_class_ratios()
labels_class = ['1st', '2nd', '3rd']
ax2.bar(bar_position_class, data_class, tick_label=labels_class, color='green')
ax2.set_title('Survival rate by ticket class')

# Plot a bar graph showing survival rate by port of embarkation.
bar_position_port = [1,2,3]
data_port = get_port_ratios()
labels_port = ['Cherbourg', 'Queenstown', 'Southampton']
ax3.bar(bar_position_port, data_port, tick_label=labels_port, color='darkturquoise')
ax3.set_xticklabels(labels_port, rotation=45, ha='right')
ax3.set_title('Survival rate by port of embarkation')

# Show the graphs.
fig.tight_layout()
plt.show()

