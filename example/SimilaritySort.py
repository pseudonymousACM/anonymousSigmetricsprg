import numpy as np
from scipy.spatial.distance import euclidean

# سوپر کلاسترها
super_clusters = {
    "1": [17, 3, 12, 5, 9],
    "6": [19, 8, 5, 14, 2],
    "13": [14, 4, 11, 13, 2],
    "14": [3, 6, 18, 17, 5],
    "21": [11, 3, 14, 5, 8],
    "4": [20, 6, 17, 1, 10],
    "19": [13, 5, 19, 4, 17],
    "3": [8, 14, 2, 11, 18],
    "16": [5, 14, 9, 2, 12],
    "18": [15, 7, 2, 10, 18],
    "8": [7, 15, 12, 3, 16],
    "15": [10, 19, 1, 8, 16],
    "25": [9, 10, 14, 3, 8],
    "22": [6, 18, 2, 7, 20],
    "23": [20, 4, 1, 12, 19],
    "5": [13, 16, 4, 3, 12],
}

# فیچر وکتور ورودی
input_vector = [14, 4, 11, 13, 2]

# محاسبه فاصله اقلیدسی بین فیچر وکتور ورودی و هر سوپر کلاستر
distances = {cluster: euclidean(input_vector, np.array(vector)) for cluster, vector in super_clusters.items()}

# مرتب‌سازی بر اساس فاصله
sorted_distances = sorted(distances.items(), key=lambda x: x[1])

# چاپ نتایج
print("--- Distances (Sorted) ---")
for cluster, distance in sorted_distances:
    print(f"Cluster {cluster}: Distance = {distance:.2f}")

closest_cluster = sorted_distances[0][0]
print(f"\nThe closest cluster is: {closest_cluster}")
