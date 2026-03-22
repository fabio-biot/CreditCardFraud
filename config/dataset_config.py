import kagglehub

# Download latest version 
# WARNING THIS FILE ENABLES A DOWNLOADING

path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")

print("Path to dataset files:", path)
