import h5py
import numpy as np

# Total number of chunks we expect
total_parts = 4
base_name = "single_sperm_flow_field_part_{:02d}.h5"
output_name = "single_sperm_flow_field_COMPLETE.h5"

print("Stitching HDF5 files together...")

# 1. Grab the static variables (time, X, Y) from the first file
with h5py.File(base_name.format(1), 'r') as f:
    time_array = f['time'][:]
    x_coords = f['x_coords'][:]
    y_coords = f['y_coords'][:]

# 2. Extract and concatenate the velocity fields in order
velocity_chunks = []
for i in range(1, total_parts + 1):
    file_name = base_name.format(i)
    print(f"Reading {file_name}...")
    with h5py.File(file_name, 'r') as f:
        velocity_chunks.append(f['velocity_field'][:])

# Stack them along the time axis (axis 0)
full_velocity_array = np.concatenate(velocity_chunks, axis=0)

print(f"All parts combined. Final shape: {full_velocity_array.shape}")

# 3. Write out the final master file
with h5py.File(output_name, 'w') as f:
    f.create_dataset('time', data=time_array)
    f.create_dataset('x_coords', data=x_coords)
    f.create_dataset('y_coords', data=y_coords)
    f.create_dataset('velocity_field', data=full_velocity_array, compression='gzip')

print(f"Success! Master file saved as {output_name}")