import os
# Path to the data directory
data_dir = '/content/CBAMDenseUNet/data'
os.makedirs(data_dir, exist_ok=True)
# Create __init__.py file
init_path = os.path.join(data_dir, '__init__.py')
with open(init_path, 'w') as f:
    f.write("# This makes 'data' a package\n")
print("__init__.py created at:", init_path)
