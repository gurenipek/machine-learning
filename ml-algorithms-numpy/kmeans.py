import numpy as np

def initialize_centroids(points, k):
    """returns k centroids from the initial points"""
    centroids = points.copy()
    np.random.shuffle(centroids)
    return centroids[:k]

def euclidean(point_a, point_b):
    return np.linalg.norm(point_a - point_b)

def assign_clusters(data, cluster_centers):
    """
    Assigns every data point to its closest (in terms of Euclidean distance) cluster center.
    :param data: An (N, D) shaped numpy array where N is the number of examples
    and D is the dimension of the data
    :param cluster_centers: A (K, D) shaped numpy array where K is the number of clusters
    and D is the dimension of the data
    :return: An (N, ) shaped numpy array. At its index i, the index of the closest center
    resides to the ith data point.
    """
    result_array = []

    for example in data:
        i = 0
        min_dist = 999999
        correct_index = None
        for cluster in cluster_centers:
            dist = euclidean(example, cluster)
            if dist < min_dist:
                min_dist = dist
                correct_index = i
            i+=1
        result_array.append(correct_index)


def calc_new_center(data):
    x_sum = 0
    y_sum = 0
    i = 0
    for x in data:
        x_sum += x[0]
        y_sum += x[1]
        i += 1
    return [x_sum/i, y_sum/i]


def calculate_cluster_centers(data, assignments, cluster_centers, k):
    """
    Calculates cluster_centers such that their squared Euclidean distance to the data assigned to
    them will be lowest.
    If none of the data points belongs to some cluster center, then assign it to its previous value.
    :param data: An (N, D) shaped numpy array where N is the number of examples
    and D is the dimension of the data
    :param assignments: An (N, ) shaped numpy array with integers inside. They represent the cluster index
    every data assigned to.
    :param cluster_centers: A (K, D) shaped numpy array where K is the number of clusters
    and D is the dimension of the data
    :param k: Number of clusters
    :return: A (K, D) shaped numpy array that contains the newly calculated cluster centers.
    """
    # for a cluster, get all data thats assigned to a cluster
    # sort based on assignments -> 0,2,1,2,0 -> 0,0,1,2,2
    new_data = []
    data = data.tolist()
    assignments = assignments.tolist()
    cluster_centers = cluster_centers.tolist()
    for i in range(len(assignments)):
        new_data.append([data[i],assignments[i]])

    new_data = sorted(new_data, key=lambda x: x[1])

    values = set(map(lambda x:x[1], new_data))
    new_data = [[y[0] for y in new_data if y[1]==x] for x in values]
    result_array = []

    for group in new_data:
        new_center = calc_new_center(group)
        result_array.append(new_center)

    for i in range(k):
        if i not in assignments:
            result_array.insert(i, cluster_centers[i])

    return np.array(result_array)


def kmeans(data, initial_cluster_centers):
    """
    Applies k-means algorithm.
    :param data: An (N, D) shaped numpy array where N is the number of examples
    and D is the dimension of the data
    :param initial_cluster_centers: A (K, D) shaped numpy array where K is the number of clusters
    and D is the dimension of the data
    :return: cluster_centers, objective_function
    cluster_center.shape is (K, D).
    objective function is a float. It is calculated by summing the squared euclidean distance between
    data points and their cluster centers.
    """
    k = len(initial_cluster_centers)
    assignments = assign_clusters(data, initial_cluster_centers)
    cluster_centers = calculate_cluster_centers(data, assignments, initial_cluster_centers, k)
    max_iter = 50
    obj_func = 0
    i = 0
    while i < max_iter:
        length = len(assignments)
        arr = np.ones(length*2, dtype=np.int64)
        arr.reshape(length,2)
        assignments_before = copy.deepcopy(arr)
        assignments = assign_clusters(data, cluster_centers)
        cluster_centers = calculate_cluster_centers(data, assignments, cluster_centers, k)
        if(assignments_before == assignments):
            break
        i += 1

    for j in range(len(data)):
        #  summing the squared euclidean distance between data points and their cluster centers.
        cluster = assignments[j]
        obj_func += (euclidean(data[j], cluster_centers[cluster]))**2

    return cluster_centers, obj_func