import warnings
import pandas as pandas
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings('ignore')

train_data = pandas.read_csv("datasets/credit_train.csv") #carregar csv de treino
test_data = pandas.read_csv("datasets/credit_test.csv") #carregar csv de teste

train_data.drop(['Months since last delinquent', 'Loan ID', 'Customer ID'],axis=1, inplace=True)
train_data.dropna(axis = 0, inplace = True) #retirar valores nulos
train_data.drop_duplicates(inplace = True) #retirar valores duplicados

test_data.drop(['Months since last delinquent', 'Loan ID', 'Customer ID'],axis=1, inplace=True)
test_data.dropna(axis = 0, inplace = True) #retirar valores nulos
test_data.drop_duplicates(inplace = True) #retirar valores duplicados

label_enconder = LabelEncoder()
train_data['Loan Status'] = label_enconder.fit_transform(train_data['Loan Status'])
train_data['Term'] = label_enconder.fit_transform(train_data['Term'])
train_data['Purpose'] = label_enconder.fit_transform(train_data['Purpose'])
train_data['Home Ownership'] = label_enconder.fit_transform(train_data['Home Ownership'])
train_data['Years in current job'] = label_enconder.fit_transform(train_data['Years in current job'])

test_data['Term'] = label_enconder.fit_transform(test_data['Term'])
test_data['Purpose'] = label_enconder.fit_transform(test_data['Purpose'])
test_data['Home Ownership'] = label_enconder.fit_transform(test_data['Home Ownership'])
test_data['Years in current job'] = label_enconder.fit_transform(test_data['Years in current job'])

# Criação de dados de treino e de dados de teste
X = train_data.drop(labels = 'Loan Status', axis = 1).values
y = train_data['Loan Status'].values

x_train , x_test , y_train , y_test = train_test_split(X,y, test_size= 0.30, random_state =86)

standard_scaler = StandardScaler()
x_train = standard_scaler.fit_transform(x_train)
x_test = standard_scaler.fit_transform(x_test)

test_data = standard_scaler.fit_transform(test_data)

dec_tree = DecisionTreeClassifier(random_state=86, criterion='entropy', max_depth=9, max_features='sqrt')
dec_tree.fit(x_train, y_train)


predict = dec_tree.predict(test_data)

predict = pandas.DataFrame(predict, columns=['Loan Status']).to_csv('./output/decision_tree_predict.csv')