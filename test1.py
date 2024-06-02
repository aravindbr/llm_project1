import numpy as np
import random
import time
from skimage import io
import imageio

def readjust_centroids(X, labels, k):
    # Initialize an array to store the new centroids
    centroids = np.zeros((k, X.shape[1]))
    
    # Loop over each cluster index from 0 to k-1
    for i in range(k):
        # Get all points assigned to the current cluster
        cluster_points = X[labels == i]
        
        # Check if the cluster has any points assigned to it
        if len(cluster_points) > 0:
            # Calculate the mean of the cluster points to get the new centroid
            centroids[i] = cluster_points.mean(axis=0)
        else:
            # If no points are assigned to the cluster, randomly pick a point from the data as the new centroid
            centroids[i] = X[random.randint(0, len(X) - 1)]
    
    return centroids


def kmeans(pixels, k, max_iters=100, tol=1e-4):
    """K-means clustering algorithm for image compression."""
    centroids = pixels[random.sample(range(len(pixels)), k)]
    iterations = 0

    for _ in range(max_iters):
            
        #Assign each pixel to the nearest centroid.
        distances = np.linalg.norm(pixels[:, np.newaxis] - centroids, axis=2)  # Squared-ℓ2 norm
        labels = np.argmin(distances, axis=1)

        new_centroids = readjust_centroids(pixels, labels, k)
        
        if np.all(np.linalg.norm(new_centroids - centroids, axis=1) < tol):
            break
        centroids = new_centroids
        iterations += 1
    return labels + 1, centroids, iterations  # Adjust cluster assignment to be 1-indexed

def run_kmeans(image_path, k_values):
    print("Processing run_kmeans]")
    image = io.imread(image_path)
    pixels = image.reshape(-1, 3)
    results = {}
    for k in k_values:
        start_time = time.time()
        labels, centroids, iterations = kmeans(pixels, k)
        elapsed_time = time.time() - start_time
        compressed_image = centroids[labels - 1].reshape(image.shape).astype(np.uint8)
        results[k] = {
            'labels': labels,
            'centroids': centroids,
            'compressed_image': compressed_image,
            'time': elapsed_time
        }
        log_file.write(f"k={k}, iterations={iterations}, elapsed_time={elapsed_time:.4f} seconds\n")
    return results

if __name__ == "__main__":
    image_paths = ["./data/football.bmp", "./data/Glockenbronze.png", "./data/nature.bmp"]
    k_values = [2, 5, 10, 20, 30]
    log_file_path = './output/kmeans_log.txt'

    with open(log_file_path, 'w') as log_file:
        for image_path in image_paths:
            log_file.write(f"Processing image: {image_path}\n")
            results = run_kmeans(image_path, k_values)
            for k, result in results.items():
                imageio.imwrite(f"./output/{image_path.split('/')[-1].split('.')[0]}_compressed_k{k}.png", result['compressed_image'])
                print(f"Image: {image_path}, k: {k}, Time: {result['time']}s, Centroids:\n{result['centroids']}")
            log_file.write("\n")
p