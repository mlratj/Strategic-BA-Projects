import os.path
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt


dir_name = os.path.dirname(os.path.abspath(__file__))


# 1st subtask
def mean_median(filename):
    df = pd.read_csv(os.path.join(dir_name, filename), header = 0)
    print(df.info())
    print('Mean: ')
    print(df.mean(axis = 0))
    print('Median: ')
    print(df.median(axis = 0))


# 2nd subtask
def load_scaled(filename):
    scaler = MinMaxScaler()
    df = pd.read_csv(os.path.join(dir_name, filename), header = 0)
    df[['ADS', 'CV']] = scaler.fit_transform(df[['ADS', 'CV']])
    return df


def euclidean_cluster(df):
    cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')
    cluster.fit_predict(df[['ADS', 'CV']])
    return cluster


def draw_cluster(cluster, df):
    plt.scatter(df.iloc[:, 0].values, df.iloc[:, 1].values, c=cluster.labels_, cmap='rainbow')
    plt.show()
    return None


# 3rd subtask
def project_evaluation_plot(filename):
    df = pd.read_csv(os.path.join(dir_name, filename), header = 0)
    plt.scatter(df.iloc[:, 1], df.iloc[:, 2])
    plt.show()


# 4th subtask
def hr_cluster(filename):
    df = pd.read_csv(os.path.join(dir_name, filename), header=0)
    cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')
    cluster.fit_predict(df[['S', 'LPE', 'NP']])
    df['Cluster'] = cluster.labels_
    print(df.groupby('Cluster')['S'].median())


# 5th subtask
def telco_segmentation(filename):
    df = pd.read_csv(os.path.join(dir_name, filename), header=0)
    cluster = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
    cluster.fit_predict(df[['Age']])
    df['Cluster'] = cluster.labels_
    print(df.groupby('Cluster').mean())
    print(df.groupby('Cluster').count())


if __name__ == '__main__':
    # First subtask
    mean_median('DATA_2.01_SKU.csv')
    # Second subtask
    scalled_df = load_scaled('DATA_2.01_SKU.csv')
    cluster = euclidean_cluster(scalled_df)
    draw_cluster(cluster, scalled_df)
    # Third subtask
    project_evaluation_plot('DATA_2.02_HR.csv')
    # Fourth subtask
    hr_cluster('DATA_2.02_HR.csv')
    # Fifth subtask
    telco_segmentation('DATA_2.03_Telco.csv')
