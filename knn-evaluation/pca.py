def pca_2_components(data):
	return pca_decompose(data,2)
	
def pca_decompose(data,n_components):
	from sklearn.decomposition import PCA, IncrementalPCA
	print("[i] Decomposing to {} components by PCA ...".format(n_components), end=" ")
	ipca = IncrementalPCA(n_components=n_components, batch_size=10)
	data_ipca = ipca.fit_transform(data)
	print("Done!")
	return data_ipca