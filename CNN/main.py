from signatureDetection import signatureDetection
from datasetsCreation import resizeSignatures
from signaturePolymerization import combineMatchingImages
from trainining import training
from test import test



# === Step 1: Detect Signatures ===
image_path = r"D:\SignatureDetection\ProjekatSOS\SignatureLibrary\NV12.jpg"
save_dir = r"D:\SignatureDetection\ProjekatSOS\SingatureDetectedLibrary"
start_index = "NV12"
preprocessed_output_dir = r"D:\SignatureDetection\ProjekatSOS\ImagePreProcessing"

# Uncomment to extract individual signatures
#signatureDetection(image_path, save_dir, start_index, preprocessed_output_dir)


# === Step 2: Combine Name + Surname Signatures for training dataset ===
folder11 = r"D:\SignatureDetection\ProjekatSOS\SingatureDetectedLibrary\potpisi\Ime"
folder12 = r"D:\SignatureDetection\ProjekatSOS\SingatureDetectedLibrary\potpisi\ImeR"
folder21 = r"D:\SignatureDetection\ProjekatSOS\SingatureDetectedLibrary\potpisi\Prezime"
folder22 = r"D:\SignatureDetection\ProjekatSOS\SingatureDetectedLibrary\potpisi\PrezimeR"
#output_combined_folder = r"D:\SignatureDetection\ProjekatSOS\SingatureDetectedLibrary\potpisi\NV"

# Uncomment to combine name + surname into full signatures
#combineMatchingImages(folder11, folder12, folder21, folder22, output_combined_folder)


# === Step 3: Combine Name + Surname Signature for test dataset ===
# folderT11 = r"D:\SignatureDetection\ProjekatSOS\SingatureDetectedLibrary\testing\TestN"
# folderT12 = r"D:\SignatureDetection\ProjekatSOS\SingatureDetectedLibrary\testing\TestNR"
# folderT21 = r"D:\SignatureDetection\ProjekatSOS\SingatureDetectedLibrary\testing\TestV"
# folderT22 = r"D:\SignatureDetection\ProjekatSOS\SingatureDetectedLibrary\testing\TestVR"
# output_combined_folderT = r"D:\SignatureDetection\ProjekatSOS\SingatureDetectedLibrary\testing\TestNV"

# Uncomment to combine test signatures
# combineMatchingImages(folderT11, folderT12, folderT21, folderT22, output_combined_folderT)


# # === Step 4: Train the Signature CNN ===
train_csv = r"D:\SignatureDetection\ProjekatSOS\pythonCode\4thSetT.csv"
train_images_dir = r"D:\SignatureDetection\ProjekatSOS\verification\training4"
epochs = 26

# # Uncomment to train
training(train_csv, train_images_dir, epochs)
#
#
# # === Step 5: Evaluate the Model ===
test_csv = r"D:\SignatureDetection\ProjekatSOS\pythonCode\4thSetTest.csv"
test_images_dir = r"D:\SignatureDetection\ProjekatSOS\verification\test4"
#
# # Run evaluation
test(test_csv, test_images_dir)
