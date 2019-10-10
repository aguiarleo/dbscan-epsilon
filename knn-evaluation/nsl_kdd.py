import pandas as pd 

# Load the data set file and get only numeric data columns 
def load_file(path, show_brief = False):

	print("[i] Loading the file {} ...".format(path))
	data_set = pd.read_csv(path, header = None,low_memory=False)
	
	if (show_brief == True):
		# info about dataset
		print('#### DATASET BRIEF ####')
		print('[i] Dimension [lines X features]: ',data_set.shape)

		#print('[i] First 10 lines:')
		#print(data_set.head(10))

		#print('[i] Statistics:')
		#print(data_set.describe())

		print('[i] Categories distribution:')
		print(data_set[41].value_counts())
		print('######################\n')
		


	
	#Removing categorical data from the data set (columns 1 to 3)
	X = data_set.iloc[:,[0,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40]].values
	Y = data_set.iloc[:,41].values #Labels
	
	print("[i] The text columns (categorical) was removed. Final data set dimension (sample X features) :{}".format(X.shape))
	
	return X,Y
	
# encode the labels like binaries
def binary_encoding_labels(Y):

	print("[i] The atack categories will be binary encoded. 0: Normal, 1: Abnormal.")

	#Binary Categories
	attack_type  = {'normal':"normal", 'neptune':"abnormal", 'warezclient':"abnormal", 'ipsweep':"abnormal",'back':"abnormal", 'smurf':"abnormal", 'rootkit':"abnormal",'satan':"abnormal", 'guess_passwd':"abnormal",'portsweep':"abnormal",'teardrop':"abnormal",'nmap':"abnormal",'pod':"abnormal",'ftp_write':"abnormal",'multihop':"abnormal",'buffer_overflow':"abnormal",'imap':"abnormal",'warezmaster':"abnormal",'phf':"abnormal",'land':"abnormal",'loadmodule':"abnormal",'spy':"abnormal",'perl':"abnormal"} 
	attack_encoding_cluster  = {'normal':0,'abnormal':1}

	Y[:] = [attack_type[item] for item in Y[:]] #Encoding the binary data
	Y[:] = [attack_encoding_cluster[item] for item in Y[:]]#Changing the names of the labels to binary labels normal and abnormal
	
	return Y