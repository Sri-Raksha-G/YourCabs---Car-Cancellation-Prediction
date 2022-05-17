import pandas as pd
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg' , 'plas' , 'pres', 'skin' , 'test', 'mass' , 'pedi', 'age', 'class']

df  = pd.read_csv(url , names=names)
print(df)
array=df.values
x=array[:,0:8]
y=array[:,8]

x_train, x_test, y_train, y_test = model_selection.train_test_split(x,y, test_size=0.2, random_state=0)


model=LogisticRegression(random_state=0)
model.fit(x_train,y_train)

result=model.predict(x_test)
# print(result)

score=model.score(x_test, y_test)
print(score)





#importing joblib
import joblib
joblib.dump(model, 'deployment_model1.pkl' )
