import kagglehub

# Download latest version
path = kagglehub.dataset_download("nigelwilliams/ngsim-vehicle-trajectory-data-us-101")

print("Path to dataset files:", path)