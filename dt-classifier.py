import sys, pandas
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
import graphviz 
from graphviz import Source

def check_arguments():

    if(len(sys.argv) < 2 or len(sys.argv) > 3):
        print('\n\tCall python3 dt-classifier.py data_file criterion(optional, default="gini")\n')
        exit()

    if(len(sys.argv) == 3):
    	return sys.argv[2]
    else:
    	return 'gini'

def read_data_file():

	data_file = open(sys.argv[1])
	data = pandas.read_csv(data_file, sep='\t')
	target = data.target
	del data['target']
	attributes = data
	feature_cols = list(data.columns)
	return attributes, target, feature_cols

criterion = check_arguments()

try:
	attributes, target, feature_cols = read_data_file()
	attributes_train, attributes_test, target_train, target_test = train_test_split(attributes, target,test_size=0.2, shuffle=False)

	dtc = DecisionTreeClassifier(criterion=criterion, random_state=1)
	dtc = dtc.fit(attributes_train, target_train)

	test_pred = dtc.predict(attributes_test)

	print("Accuracy:",metrics.accuracy_score(target_test, test_pred))

	tree.export_graphviz(dtc,
                     out_file="tree.dot",
                     feature_names = feature_cols, 
                     class_names=['democrat', 'republican'],
                     filled = True)

except:
	print('\n\tAn except occurred. Check the inputs.')
	print('\n\tExecution failed.\n')