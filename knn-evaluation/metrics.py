from sklearn.metrics import classification_report

def report(y_true, y_pred, target_names):

	metrics = classification_report(y_true, y_pred, target_names = target_names, output_dict = True)

#	for item in target_names:
#		print("=> Metricas do {} (total de {} registros nos dados corretos):".format(item,metrics[item]['support']))
#		print("  TPR (recall): {}".format(metrics[item]['recall']))
#		print("  Precisao (precision): {}".format(metrics[item]['precision']))
#		print("  F1-Score: {}".format(metrics[item]['f1-score']))
#		print("")
	
	
	tpr = metrics['weighted avg']['recall']
	precision = metrics['weighted avg']['precision']
	fpr = 1 - metrics['weighted avg']['recall']
	fscore = 2 * ((metrics['weighted avg']['precision'] * metrics['weighted avg']['recall']) / (metrics['weighted avg']['precision'] + metrics['weighted avg']['recall']))

	return tpr, precision, fpr, fscore
	