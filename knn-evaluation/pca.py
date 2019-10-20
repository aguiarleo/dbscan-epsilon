def pca_2_components(data):
	from sklearn.decomposition import PCA, IncrementalPCA
	print("[i] Decomposing to 2 components by PCA ...", end=" ")
	ipca = IncrementalPCA(n_components=2, batch_size=10)
	data_ipca = ipca.fit_transform(data)
	print("Done!")
	return data_ipca