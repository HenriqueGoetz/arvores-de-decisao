import sys, pandas
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
import graphviz 
from graphviz import Source

def check_arguments():

    if(len(sys.argv) != 2 and len(sys.argv) != 5):
        print('\n\tCall python3 dt-classifier.py data_file { criterion max_depth ccp_alpha (optional, default="gini None 0.0") } \n')
        exit()

    criterion = 'gini'
    max_depth = None
    ccp_alpha = 0.0

    if(len(sys.argv) == 5):
    	criterion = sys.argv[2]
    	max_depth = int(sys.argv[3]) if sys.argv[3].isnumeric() else None
    	ccp_alpha = float(sys.argv[4])

    return criterion, max_depth, ccp_alpha


def read_data_file():

	data_file = open(sys.argv[1])
	data = pandas.read_csv(data_file, sep='\t')
	target = data.target
	classes = set()	
	for c in target:
		classes.add(c)

	del data['target']
	attributes = data
	feature_cols = list(data.columns)
	return attributes, target, feature_cols, list(classes)


criterion, max_depth, ccp_alpha = check_arguments()

print('\n\t*** Decision Tree Classifier ***')
print('\n\tChosen criterion: ' + criterion)
print('\tChosen max_depth: ' + (str(max_depth) if max_depth else 'None'))
print('\tChosen ccp_alpha: ' + (str(ccp_alpha)))
print('\tPath to data file: ' + sys.argv[1])
print('\n\t********************************')
print('\n\tRunning DT-Classifier...')

try:
	attributes, target, feature_cols, classes = read_data_file()
	attributes_train, attributes_test, target_train, target_test = train_test_split(attributes, target, test_size=0.2, shuffle=False)

	dtc = DecisionTreeClassifier(criterion=criterion, random_state=1, max_depth=max_depth, ccp_alpha=ccp_alpha)
	dtc = dtc.fit(attributes_train, target_train)

	test_pred = dtc.predict(attributes_test)

	print("\n\tAccuracy:",metrics.accuracy_score(target_test, test_pred))

	tree.export_graphviz(dtc,
                     out_file = "tree.dot",
                     feature_names = feature_cols, 
                     class_names = classes,
                     filled = True)

	print("\n\tDecision Tree saved as tree.dot", end="\n\n")

except:
	print('\n\tAn except occurred. Check the inputs.')
	print('\n\tExecution failed.\n')