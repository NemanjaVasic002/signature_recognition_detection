from signatureDetection import signatureDetection
from datasetsCreation import resizeSignatures
from signaturePolymerization import combineMatchingImages
from training import training
from test import test

# === Step 1: Detect Signatures ===
image_path = r"path/to/input/signature.jpg"  # Full document or form
save_dir = r"path/to/save/extracted_signatures"
start_index = 0                             #number of image
preprocessed_output_dir = r"path/to/save/preprocessing_debug" #this is where you will save images in processing period so you can see why you arent getting wanted results

# Uncomment to extract individual signatures
# signatureDetection(image_path, save_dir, start_index, preprocessed_output_dir)


# === Step 2: Combine Name + Surname Signatures for training dataset ===
# folder11: original name images
# folder12: resized name images (for matching)
# folder21: original surname images
# folder22: resized surname images (for matching)

folder11 = r"path/to/name_original"
folder12 = r"path/to/name_resized"
folder21 = r"path/to/surname_original"
folder22 = r"path/to/surname_resized"
output_combined_folder = r"path/to/save/combined_signatures"

# Uncomment to combine name + surname into full signatures
# combineMatchingImages(folder11, folder12, folder21, folder22, output_combined_folder)


# === Step 3: Combine Name + Surname Signature for test dataset ===
# resizeSignatures(input_folder, output_folder, target_height=512, targetWidth=1044)
# folder11: original name images
# folder12: resized name images (for matching)
# folder21: original surname images
# folder22: resized surname images (for matching)

folderT11 = r"path/to/name_original"
folderT12 = r"path/to/name_resized"
folderT21 = r"path/to/surname_original"
folderT22 = r"path/to/surname_resized"
output_combined_folderT = r"path/to/save/combined_signatures"

# Uncomment to combine name + surname into full signatures
# combineMatchingImages(folderT11, folderT12, folderT21, folderT22, output_combined_folderT)

# === Step 4: Train the Signature CNN ===
train_csv = r"path/to/train_labels.csv"          # Format: image_name,label
train_images_dir = r"path/to/training_signatures"   #this path can be different than output_combine_path cuz you should add some non signature examples to your training set
epochs = 10

# Uncomment to train
# training(train_csv, train_images_dir, epochs)


# === Step 5: Evaluate the Model ===
test_csv = r"path/to/test_labels.csv"
test_images_dir = r"path/to/testing_signatures"

# Run evaluation
test(test_csv, test_images_dir)
