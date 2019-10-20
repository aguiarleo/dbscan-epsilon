'''
Refs:
	https://scikit-learn.org/stable/modules/model_evaluation.html#classification-report
	https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html
	
	Descrição na pág 40 da dissertaçãoo :
	
	Precisão(PPV), que considera os casos positivos corretamente classificados sobre o total de casos classificados como positivos.
		Precisão = (True Positives) / (True Positives + False Positives)
		
		key 'precision' no classification_report
	
	Taxa de verdadeiros positivos (TPR), ou sensibilidade, que considera todos os casos positivos corretamente classificados sobre o total de casos positivos.
		TPR = (True Positives) / (True positives + False Negatives)
		
		key 'recall' no classification_report
		
	Taxa de falsos positivos (FPR), que considera todos os falsos positivos sobre o total de casos negativos.
		FPR = (Falsos positivos) / (Falses Positives + True Negatives)
		
		-> O falso positivo de um valor é 1 - TPR do outro valor.
			Entao, no  classification_report eu posso pegar o recall do outro valor e subtrair de 1.
		
	Medida-F (F-score), é uma forma de medir a acurácia do sistema que leva em consideração a precisão e a sensibilidade.
		F-Score = 2* (Precisão * TPR) / (Precisão + TPR)
'''


from sklearn.metrics import classification_report
dados_corretos = ['x','x','y','x','y','x','y','y','x','x']
dados_preditos = ['x','x','y','x','y','x','y','x','x','y']


target_names = ['xis', 'ypsilon']
metrics = classification_report(dados_corretos, dados_preditos, target_names = target_names, output_dict = True)

for item in target_names:
	print("=> Metricas do {} (total de {} registros nos dados corretos):".format(item,metrics[item]['support']))
	print("  TPR (recall): {}".format(metrics[item]['recall']))
	print("  Precisao (precision): {}".format(metrics[item]['precision']))
	print("  F1-Score: {}".format(metrics[item]['f1-score']))
	print("")
	

	
print("=> Metricas gerais:")
print("  TPR (recall): {}".format(metrics['weighted avg']['recall']))

print("  Precisao (precision): {}".format(metrics['weighted avg']['precision']))

print("  F1-Score: {}".format(metrics['weighted avg']['f1-score']))
#fscore = 2 * ((metrics['weighted avg']['precision'] * metrics['weighted avg']['recall']) / (metrics['weighted avg']['precision'] + metrics['weighted avg']['recall']))
#print("  F-Score: {}".format(fscore))

fpr = 1 - metrics['weighted avg']['recall']
print("  FPR: {}".format(fpr))