import os

base_path = "denoising-dirty-documents"

train_path = os.path.join(base_path, "train")
test_path = os.path.join(base_path, "test")
cleaned_path = os.path.join(base_path, "train_cleaned")

features_path = "features.csv"
sample_prob = 0.02

model_path = "denoiser.pickle"