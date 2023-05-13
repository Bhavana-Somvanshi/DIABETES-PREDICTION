#Logistic Regression:

logreg = LogisticRegression(solver='lbfgs', max_iter = 1000)
logreg.fit(X_train, Y_train)
y_pred = logreg.predict(X_test)
lg_accuracy = round(accuracy_score(y_pred, Y_test), 2)*100
lg_accuracy


#K Neighbour Classifier:

knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(X_train, Y_train)
y_pred = knn.predict(X_test)
knn_accuracy = round(accuracy_score(y_pred, Y_test), 2)*100
knn_accuracy


#Random Forest Classifier:

classifier = RandomForestClassifier(n_estimators=6, criterion='entropy', random_state=0)
classifier.fit(X_train, Y_train)
y_pred = classifier.predict(X_test)
rf_accuracy = round(accuracy_score(y_pred, Y_test), 2)*100
rf_accuracy

cm = confusion_matrix(Y_test, y_pred)
new_cm = pd.DataFrame(cm , index = ['Diabetic','Not Diabetic'] , columns = ['Diabetic','Not Diabetic'])
sns.heatmap(new_cm,cmap= 'Blues', annot = True, fmt='',xticklabels = ['Diabetic','Not Diabetic'], yticklabels = ['Diabetic','Not Diabetic'])
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title('Confusion matrix On Test Data')
plt.show()


#Predictive System:

df.info()

#print("Enter the values of the following parameters: ")
#pregnancies = int(input("Pregnancies: "))
#glucose = float(input("Glucose: "))
#bp = float(input("Blood Pressure: "))
#st = float(input("Skin Thickness: "))
#insulin = float(input("Insulin: "))
#bmi = float(input("BMI: "))
#dbf = float(input("Diabetes Pedigree Function: "))
#age = int(input("Age"))
#input_data = [[pregnancies, glucose, bp, st, insulin, bmi, dbf, age]]
input_data = [[0, 137, 40, 35, 168, 43.1, 2.288, 33]]
prediction = classifier.predict(input_data)
if(prediction[0] == 1):
print("\nYou are Diabetic")
else:
print("\nYou are NOT Diabetic")


#Saving the model

filename = 'diabetes-prediction-model.pkl'
pickle.dump(classifier, open(filename, 'wb'))
