# signature_recognition_detection
This project combines classical image processing and a Convolutional Neural Network (CNN) to detect, extract, and classify signatures from images — distinguishing between original and forged signatures.
📌 Features

    ✔️ Signature Detection using color thresholds and contour analysis.

    ✨ Shadow Removal using Multi-Scale Retinex.

    🔍 Automatic Cropping of signature regions.

    🧠 CNN Model trained to classify signatures as genuine or fake.

    🛠️ Tools to combine and preprocess images, visualize histograms, and resize datasets.


    🔹 Step 1: Preprocess the Image

    Apply Multi-Scale Retinex to reduce shadows.

    Extract signature region via HSV thresholding.

🔹 Step 2: Extract Signature

    Label potential signature blobs.

    Filter using heuristics (area, pixel density).

    Crop and save the best candidate(s).

🔹 Step 3: Classification

    Train a CNN model on pre-labeled signature images.

    Classify new signature crops as original or forged.
