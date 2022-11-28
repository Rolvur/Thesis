import numpy as np
import pandas as pd 
import random
from Settings import*
from sklearn_extra.cluster import KMedoids
import matplotlib.pyplot as plt
random.seed(123)



### TEST ### 
data1 = [1,2,3,4,5,6,7,8,9,10]
data2 = [10,20,30,40,50,60,70,80,90,100]
data3 = [100,200,300,400,500,600,700,800,900,1000]
data4 = [1000,2000,3000,4000,5000,6000,7000,8000,9000,10000]
data5 = [10000,20000,30000,40000,50000,60000,70000,80000,90000,100000]
df_test = pd.DataFrame({'Data1':data1,'Data2':data2,'Data3':data3,'Data4':data4,'Data5':data5})

#df_test.corr()

DA = df_test['Data1'].tolist()
FCR =  df_test['Data2'].tolist() 
aFRR_up =  df_test['Data3'].tolist()
aFRR_down =  df_test['Data4'].tolist()
mFRR =  df_test['Data5'].tolist()


Data = [DA,FCR,aFRR_up,aFRR_down,mFRR]
Data_names = ['DA','FCR','aFRR_up','aFRR_down','mFRR']

Type = 'single'   # 'single' or 'combined'

n_samples = 50 #Number of samples to be made  

blocksize = 2 # 7days = 168 hours
sample_length = blocksize*5 # sampling 52 weeks blocks


def Bootsrap(Type,Data,Data_names,n_samples,blocksize,sample_length):


    if Type == 'single':

        for x in range(0,len(Data)):
            #Sample length
            n = len(Data[x])

            #Split sample in blocks of length blocksize

            blocks = [Data[x][i:i + blocksize ] for i in range (0,n,blocksize)]

            #Delete last block if length differs from blocksize 
            if len(blocks[-1]) != blocksize:
                del blocks[-1]

            samples = np.zeros((n_samples,sample_length))

            for i in range(0,n_samples):
                t = 0
                while t < sample_length:

                    r = random.randrange(0,len(blocks))
                    print(r)

                    for j in range(0,blocksize):
                        samples[i,t+j] = blocks[r][j]

                    t = t + blocksize
        
            if Data_names[x] == 'DA':
                DA_block = samples
            if Data_names[x] == 'FCR':
                FCR_block = samples
            if Data_names[x] == 'aFRR_up':
                aFRR_up_block = samples
            if Data_names[x] == 'aFRR_down':
                aFRR_down_block = samples
            if Data_names[x] == 'mFRR':
                mFRR_block = samples


                
    if Type == 'combined':

        ########## Multi Blocks ######## 
        ### Multi ### 
        df = pd.DataFrame({'DA':  np.array(Data[0]), 'FCR':  np.array(Data[1]) , 'aFRR_up:':  np.array(Data[2]), 'aFRR_down': np.array(Data[3]), 'mFRR':  np.array(Data[4])})

        data = df.values.tolist() #Acces element by  data[0][0]

        n = len(data)

        #Split sample in blocks of length blocksize

        blocks = [data[i:i + blocksize ] for i in range (0,n,blocksize)]#Acces element by blocks[0][0][0]

        #Delete last block if length differs from blocksize 
        if len(blocks[-1]) != blocksize:
            del blocks[-1]

        len_element = len(blocks[1][1])

        ## creating an array with zeros and same dimensions as blocks 
        samples = np.zeros((n_samples,sample_length,len_element))

        for i in range(0,n_samples):
            t = 0
            while t < sample_length:

                r = random.randrange(0,len(blocks))
                print(r)

                for j in range(0,blocksize):
                
                    samples[i,t+j] = blocks[r][j]

                t = t + blocksize

        Combined_blocks = samples

        
    if Type == 'single':
        return  DA_block,FCR_block,aFRR_up_block,aFRR_down_block,mFRR_block
    if Type == 'combined': 
        return Combined_blocks

scenarios = Bootsrap(Type,Data,Data_names,n_samples,blocksize,sample_length)

### Scenario reduction ###


scenarios_DA = scenarios[0]

from dtaidistance import dtw, clustering
 
model5 = clustering.KMedoids(dtw.distance_matrix_fast, {}, k=3)
cluster_idx = model5.fit(scenarios_DA)


model5.plot()
plt.show()


from sklearn_extra.cluster import KMedoids
import numpy as np
data=np.asarray([[9,6,0],[10,4,0],[4,4,5],[5,8,2],[3,8,7],[2,5,2],[8,5,1],[4,6,6],[8,3,4],[9,2,3]])

kmedoids = KMedoids(n_clusters=2,metric='euclidean').fit(scenarios_DA)

print(kmedoids.labels_)

kmedoids.cluster_centers_

kmedoids.n_clusters

kmedoids.medoid_indices_

scenarios_DA



#from sktime import TimeSeriesKMedoids

from sktime import clustering

#from dtaidistance import dtw, clustering
 
clustering.TimeSeriesKMedoids()

model5 = clustering.KMedoids(dtw.distance_matrix_fast, {}, k=3)


cluster_idx = model5.fit(scenarios_DA)


model5.plot()


from dtaidistance import dtw, clustering


model = clustering.KMedoids(dtw.distance_matrix_fast, {}, k=3)

cluster_idx = model.fit(scenarios_DA)

#kmedoids = KMedoids(n_clusters=2, random_state=0).fit(scenarios_DA)

model.plot("kmedoids.png")



""" 

# The example.database3 synthetic database is loaded
data(example.database3)
tsdata <- example.database3[[1]]
groundt <- example.database3[[2]]

# Apply K-medoids clusterning for different distance measures

KMedoids(data=tsdata, ground.truth=groundt, k=5, "euclidean")
KMedoids(data=tsdata, ground.truth=groundt, k=5, "cid")
KMedoids(data=tsdata, ground.truth=groundt, k=5, "pdc")

# — — — — — — -Assigning Initial Centers — — — — — — — — — — — -
centers = [[4, 5], [9, 10]]
# — — — — — — -Assigning Data: Dummy Data used in example above — — — — — — — — — — — — — — — — — — 
df=np.array([[7,8], [9,10], [11,5], [4,9], [7,5], [2,3], [4,5]])
# — — — — — — -Fit KMedoids clustering — — — — — — — — — — — -
KMobj = KMedoids(n_clusters=5).fit(df)
# — — — — — — -Assigning Cluster Labels — — — — — — — — — — — -
labels = KMobj.labels_


from sklearn_extra.cluster import KMedoids
from sklearn.datasets import make_blobs

print(__doc__)

# #############################################################################
# Generate sample data
centers = [[1, 1], [-1, -1], [1, -1]]
X, labels_true = make_blobs(
    n_samples=750, centers=centers, cluster_std=0.4, random_state=0
)

# #############################################################################
# Compute Kmedoids clustering
cobj = KMedoids(n_clusters=3).fit(X)
labels = cobj.labels_

unique_labels = set(labels)
colors = [
    plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))
]
for k, col in zip(unique_labels, colors):

    class_member_mask = labels == k

    xy = X[class_member_mask]
    plt.plot(
        xy[:, 0],
        xy[:, 1],
        "o",
        markerfacecolor=tuple(col),
        markeredgecolor="k",
        markersize=6,
    )

plt.plot(
    cobj.cluster_centers_[:, 0],
    cobj.cluster_centers_[:, 1],
    "o",
    markerfacecolor="cyan",
    markeredgecolor="k",
    markersize=6,
)

plt.title("KMedoids clustering. Medoids are represented in cyan.")

plt.show()





 """


from pyclustering.cluster.kmedoids import kmedoids
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.cluster import cluster_visualizer
from pyclustering.utils import read_sample
from pyclustering.samples.definitions import FCPS_SAMPLES
 
# Load list of points for cluster analysis.
sample = read_sample(FCPS_SAMPLES.SAMPLE_TWO_DIAMONDS)
sample = scenarios_DA
 
# Initialize initial medoids using K-Means++ algorithm
initial_medoids = kmeans_plusplus_initializer(sample, 2).initialize(return_index=True)
 
# Create instance of K-Medoids (PAM) algorithm.
kmedoids_instance = kmedoids(sample, initial_medoids)
 
# Run cluster analysis and obtain results.
kmedoids_instance.process()
clusters = kmedoids_instance.get_clusters()
medoids = kmedoids_instance.get_medoids()
 
# Print allocated clusters.
print("Clusters:", clusters)
 
# Display clustering results.
visualizer = cluster_visualizer()
visualizer.append_clusters(clusters, sample)
visualizer.append_cluster(initial_medoids, sample, markersize=12, marker='*', color='gray')
visualizer.append_cluster(medoids, sample, markersize=14, marker='*', color='black')
visualizer.show()