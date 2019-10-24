def report(y_true, y_pred, target_names, average = 'binary'):
	from sklearn.metrics import recall_score, precision_score, f1_score
	
	tpr = recall_score(y_true, y_pred, average = average)
	precision = precision_score(y_true, y_pred, average = average)
	fpr = 1 - tpr
	
	#fscore = 2 * ((precision * tpr) / (precision + tpr))
	fscore = f1_score(y_true,y_pred, average = average)

	return tpr, precision, fpr, fscore
