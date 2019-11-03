import pandas as pd 
import numpy as np

# Load the data set file and get only numeric data columns 
def load_file(path, show_brief = False, remove_categorical_data = True):

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
	
	X = data_set.iloc[:,:-2].values
	Y = data_set.iloc[:,41].values #Labels
	
	if (remove_categorical_data == True):
		#Mount X removing the categorical data from the data set (columns 1 to 3)
		X = np.delete(X,[1,2,3],-1)
		#X = data_set.iloc[:,[0,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40]].values
		print("[i] The text columns (categorical) was removed. Final data set dimension (sample X features) :{}".format(X.shape))
	else:
		#Drop only the Protocol column (column 1)
		X = np.delete(X,1,-1)
		#X = data_set.iloc[:,[0,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40]].values
		print("[i] The Protocol column was removed. Final data set dimension (sample X features) :{}".format(X.shape))

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

#Assinging risk values to categorical features "Servers" and "Server Errors"
def risk_encoding_data(X):

	X = pd.DataFrame(X)
	servers  = {'http':0.01, 'domain_u':0, 'sunrpc':1, 'smtp':0.01, 'ecr_i':0.87, 'iso_tsap':1, 'private':0.97, 'finger':0.27, 'ftp':0.26, 'telnet':0.48,'other':0.12,'discard':1, 'courier':1, 'pop_3':0.53, 'ldap':1, 'eco_i':0.8, 'ftp_data':0.06, 'klogin':1, 'auth':0.31, 'mtp':1, 'name':1, 'netbios_ns':1,'remote_job':1,'supdup':1,'uucp_path':1,'Z39_50':1,'csnet_ns':1,'uucp':1,'netbios_dgm':1,'urp_i':0,'domain':0.96,'bgp':1,'gopher':1,'vmnet':1,'systat':1,'http_443':1,'efs':1,'whois':1,'imap4':1,'echo':1,'link':1,'login':1,'kshell':1,'sql_net':1,'time':0.88,'hostnames':1,'exec':1,'ntp_u':0,'nntp':1,'ctf':1,'ssh':1,'daytime':1,'shell':1,'netstat':1,'nnsp':1,'IRC':0,'pop_2':1,'printer':1,'tim_i':0.33,'pm_dump':1,'red_i':0,'netbios_ssn':1,'rje':1,'X11':0.04,'urh_i':0,'http_8001':1,'aol':1,'http_2784':1,'tftp_u':0,'harvest':1}
	X[1] = [servers[item] for item in X[1]]

	servers_error  = {'REJ':0.519, 'SF':0.016, 'S0':0.998, 'RSTR':0.882, 'RSTO':0.886,'SH':0.993,'S1':0.008,'RSTOS0':1,'S3':0.08,'S2':0.05,'OTH':0.729} 
	X[2] = [servers_error[item] for item in X[2]]

	print("[i] The Server and Servers Errors columns was risk encoded.")

	return X

def load_dataset_binary_and_risk_encoded(path, show_brief = False):
	# Load the dataset
	X, Y = load_file(path, show_brief, remove_categorical_data = False)
	
	#Encoding categorical data as risk values
	X = risk_encoding_data(X)
	
	# Encoding labels as binary data
	Y = binary_encoding_labels(Y)

	return X, Y