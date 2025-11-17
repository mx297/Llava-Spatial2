import kagglehub

# Download latest version
path = kagglehub.dataset_download("awsaf49/coco-2017-dataset",force_download=True)

print("Path to dataset files:", path)