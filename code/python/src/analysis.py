from sklearn.cross_decomposition import CCA
import numpy as np

U = np.random.random_sample(500).reshape(100,5)
V = np.random.random_sample(400).reshape(100,4)

print(U.shape)
print(V.shape)

cca = CCA(n_components=4)
cca.fit(U, V)

U_c, V_c = cca.transform(U, V)

result = np.corrcoef(U_c.T, V_c.T)#[0,1]
print("end")