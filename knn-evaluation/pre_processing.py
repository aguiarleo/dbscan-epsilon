# Scale data between 0 and 1
def scaling(X):
	from sklearn.preprocessing import MinMaxScaler
	
	#Transforming features by scaling each feature to the given range, (0,1)
	X =  MinMaxScaler(feature_range=(0, 1)).fit_transform(X)

	print("[i] Data has been successfully scaled with MinMaxScaler")
	
	return X
	
def scale_and_normalize(X):
	from sklearn.preprocessing import normalize, StandardScaler 
	scaler = StandardScaler() 
	X = scaler.fit_transform(X) 
	  
	# Normalizing the data so that the data 
	# approximately follows a Gaussian distribution 
	X = normalize(X) 
	
	print("[i] Data has been successfully scaled and normalized")
	
	return X