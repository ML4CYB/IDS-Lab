import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import AgglomerativeClustering, AffinityPropagation
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_samples, adjusted_rand_score
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, cut_tree


if os.path.exists("/home/ubuntu/IDS-Lab/HLMCC_python/Graphs") == False:
    os.makedirs('/mnt/c/Users/tjudd/OneDrive - Umich/ML4CYB/IDSL/HLMCC_code/HLMCC_python_new_data/Graphs')

invalid = True
while invalid == True:
    print('Please Enter the data that you want to use:\n\nOptions Are:')
    print('----------------------------------------------------')
    print('Enter 1 for LWSNDR Single Hop Indoor\nEnter 2 for LWSNDR Multi Hop Outdoor\nEnter 3 for satellite\nEnter 4 for IoT-23 data ')
    print('----------------------------------------------------')

    option = input()

    if option == '1':
        Data = pd.read_csv("/home/ubuntu/IDS-Lab/HLMCC_python/Datasets/original_Dataset/All_Attacks.csv")
        Data.astype(float)
        if os.path.exists("/home/ubuntu/IDS-Lab/HLMCC_python/Graphs/LWSNDR Single Hop Indoor") == False:
            os.makedirs('/home/ubuntu/IDS-Lab/HLMCC_python/Graphs/LWSNDR Single Hop Indoor')
        base_results_dir = '/home/ubuntu/IDS-Lab/HLMCC_python/Graphs/LWSNDR Single Hop Indoor'
        xcolumn = 'Humidity'
        ycolumn = 'Temperature'
        name = 'LWSNDR Single Hop Indoor'
        invalid = False
    elif option == '2':
        Data = pd.read_csv("/home/ubuntu/IDS-Lab/HLMCC_python/Datasets/original_Dataset/LWSNDR Multi Hop Indoor.csv")
        Data.astype(float)
        if os.path.exists("/home/ubuntu/IDS-Lab/HLMCC_python/Graphs/LWSNDR Multi Hop Indoor") == False:
            os.makedirs('/home/ubuntu/IDS-Lab/HLMCC_python/Graphs/LWSNDR Multi Hop Indoor')
        base_results_dir = '/home/ubuntu/IDS-Lab/HLMCC_python/Graphs/LWSNDR Multi Hop Indoor'
        xcolumn = 'Humidity'
        ycolumn = 'Temperature'
        name = 'LWSNDR Multi Hop Indoor'
        invalid = False

    elif option == '3':
        Data = pd.read_csv("/home/ubuntu/IDS-Lab/HLMCC_python/Datasets/original_Dataset/satellite.csv")
        if os.path.exists("/home/ubuntu/IDS-Lab/HLMCC_python/Graphs/satellite") == False: 
            os.makedirs('/home/ubuntu/IDS-Lab/HLMCC_python/Graphs/satellite')
        base_results_dir = '/home/ubuntu/IDS-Lab/HLMCC_python/Graphs/satellite'
        xcolumn = 'V1'
        ycolumn = 'V2'
        name = 'satellite'    
        invalid = False

    elif option == '4':
        Data = pd.read_csv("/home/ubuntu/IDS-Lab/HLMCC_python/Datasets/original_Dataset/IoT_23_data.csv")
        Data = Data.astype(float)
        if os.path.exists("/home/ubuntu/IDS-Lab/HLMCC_python/Graphs/IoT_23_data") == False: 
            os.makedirs('/home/ubuntu/IDS-Lab/HLMCC_python/Graphs/IoT_23_data')
        base_results_dir = '/home/ubuntu/IDS-Lab/HLMCC_python/Graphs/IoT_23_data'
        xcolumn = 'Index'
        ycolumn = 'duration'
        name = 'IoT_23_data'    
        invalid = False
    else:
        print('Invalid Option')
        invalid = True


os.system('clear')
x_index = Data.columns.get_loc(xcolumn)
y_index = Data.columns.get_loc(ycolumn)

# Normalization
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(Data.iloc[:, :-1])

# Ensure the DataFrame has the correct data type before assignment
normalized_Data = Data.copy()
normalized_Data.iloc[:, :-1] = pd.DataFrame(normalized_data)

# Visualization before clustering
plt.figure(figsize=(8, 6), dpi=600)
Label = Data.iloc[:, len(Data.columns) - 1]
sns.scatterplot(x=Data.iloc[:, x_index], y=Data.iloc[:, y_index], hue=Label)
plt.title(name)
plt.savefig(f"{base_results_dir}/{name}.tiff", dpi=600)

# Affinity Propagation
def negDistMat(X, r=1):
    return -squareform(pdist(X, 'euclidean'))
    
sim = negDistMat(normalized_Data.iloc[:, :len(Data.columns) - 1])
af = AffinityPropagation(affinity='euclidean').fit(sim)
cluster_centers_indices = af.cluster_centers_indices_
labels = af.labels_

agg_clustering = AgglomerativeClustering(n_clusters=2, metric ='manhattan', linkage='average')
agg_labels = agg_clustering.fit_predict(normalized_Data.iloc[:, :len(Data.columns) - 1])


# Convert cluster labels using cut_tree and rename to "Normal" and "Anomaly"
aggres = linkage(normalized_Data.iloc[:, :len(Data.columns) - 1], method='average')
Label_HAP = cut_tree(aggres, n_clusters=[2]).flatten()
Label_HAP = ["o" if label == 0 else "n" for label in Label_HAP]

Data_merge = Data.copy()
Data_merge.iloc[:, -1] = Label_HAP

Data_merge.columns = ['' for _ in range(Data_merge.shape[1])]
last_col_index = -1
new_column_name = 'ClusterLabel'
Data_merge.columns.values[last_col_index] = new_column_name

Data_merge.to_csv(f"/home/ubuntu/IDS-Lab/HLMCC_python/Datasets/clustered_Dataset/{name}.csv", index=False)

silhouette_vals = silhouette_samples(normalized_Data.iloc[:, :len(Data.columns) - 1], agg_labels)
silhouette_avg = np.mean(silhouette_vals)

# Plotting the silhouette plot
plt.figure(figsize=(8, 6), dpi=600)
x_lower = 10
    
for i in range(2):

    if len(silhouette_vals[agg_labels == (0)]) > len(silhouette_vals[agg_labels == (1)]):
        ith_cluster_silhouette_values = silhouette_vals[agg_labels == (1 - i)]  # Flip the cluster indices
    
    else:
        ith_cluster_silhouette_values = silhouette_vals[agg_labels == (i)]  # Flip the cluster indices

    
    ith_cluster_silhouette_values = np.sort(ith_cluster_silhouette_values)[::-1]  # Sort in descending order
    size_cluster_i = ith_cluster_silhouette_values.shape[0]
    x_upper = x_lower + size_cluster_i
    
    color = sns.color_palette("husl", 2)[i]
    plt.fill_between(np.arange(x_lower, x_upper), 0, ith_cluster_silhouette_values,
                      facecolor=color, edgecolor=color, alpha=0.7, label=f'Cluster {1 - i}')
    x_lower = x_upper + 10  # Add space between clusters

plt.axhline(y=silhouette_avg, color="red", linestyle="--")  # Average silhouette score
plt.ylabel("Silhouette coefficient values")
plt.xlabel("Cluster label")
plt.legend()
plt.figtext(0.5, 0, f'Silhouette Score: {round(silhouette_avg, 3)}', ha="center", fontsize=12)
plt.title(f"Silhouette plot")
plt.savefig(f"{base_results_dir}/HAP{name}.tiff")

# Clean the column by replacing non-integer values
cleaned_labels_true = pd.to_numeric(Data.iloc[:, len(Data.columns) - 1], errors='coerce').fillna(-1).astype(int)

adjusted_rand = adjusted_rand_score(cleaned_labels_true, agg_labels)

# Visualization for HAP
plt.figure(figsize=(8, 6), dpi=600)
sns.scatterplot(x=Data.iloc[:, x_index], y=Data.iloc[:, y_index], hue=Label_HAP, palette=["#56B4E9", "red"])
plt.title(f"{name} - HAP")
plt.savefig(f"{base_results_dir}/{name}HAP.tiff", dpi=600)

os.system('clear')
print('Clustering Completed')
