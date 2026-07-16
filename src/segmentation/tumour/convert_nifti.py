import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import os

# ==========================================

# ==========================================
# Point  to ONE of the original MRI scans from your dataset (e.g., patient 10's T2 scan)
original_mri_path = "E:/New folder (3)/HNTS-MRG24-UWLAIR/data/HNTSMRG24_train/10/preRT/10_preRT_T2.nii.gz"

# Point this to the corresponding prediction file your model just generated
prediction_path = "E:/New folder (3)/HNTS-MRG24-UWLAIR/Task1_preRT/prediction_testing_fold0/10_preRT_T2.nii.gz" # Update filename if different
# ==========================================

# Load the 3D data from both files
mri_img = nib.load(original_mri_path).get_fdata()
pred_img = nib.load(prediction_path).get_fdata()

# Find the best slice to visualize (the slice with the largest predicted tumor area)
# We sum the pixels along the X and Y axes to see which Z-slice has the most '1's and '2's
z_tumor_pixels = np.sum(pred_img > 0, axis=(0, 1))
best_slice_idx = np.argmax(z_tumor_pixels)

if z_tumor_pixels[best_slice_idx] == 0:
    print("Warning: The model did not predict any tumor in this volume. Defaulting to middle slice.")
    best_slice_idx = pred_img.shape[2] // 2

# Extract the 2D slices at that index
mri_slice = mri_img[:, :, best_slice_idx]
pred_slice = pred_img[:, :, best_slice_idx]

# Medical images are often saved sideways, this rotates it upright
mri_slice = np.rot90(mri_slice)
pred_slice = np.rot90(pred_slice)

# Set up the high-resolution plot
plt.figure(figsize=(10, 10))

# Plot the background MRI scan in grayscale
plt.imshow(mri_slice, cmap='gray')

# Overlay the contours (outlines) just like the GitHub repository
# Class 1 (GTVp) -> Red outline
if np.any(pred_slice == 1):
    plt.contour(pred_slice == 1, colors='red', linewidths=1.5, levels=[0.5])

# Class 2 (GTVn) -> Green outline
if np.any(pred_slice == 2):
    plt.contour(pred_slice == 2, colors='green', linewidths=1.5, levels=[0.5])

# Clean up the image (remove axes and borders)
plt.axis('off')

# Save the final image to your folder
output_filename = f"repo_style_overlay_slice_{best_slice_idx}.png"
plt.savefig(output_filename, bbox_inches='tight', pad_inches=0, dpi=300)
print(f"Successfully generated repo-style overlay: {output_filename}")

# Display it on screen
plt.show()