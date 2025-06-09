import cv2
import numpy as np
import matplotlib.pyplot as plt



def plot_highlighted_histogram(imagePath):
    # Read image in grayscale
    img = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print("Error: Image not loaded. Check the file path.")
        return

    # Calculate histogram
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])

    # Create figure with subplots
    plt.figure(figsize=(15, 6))

    # Full histogram
    plt.subplot(131)
    plt.plot(hist, color='black')
    plt.title('Full Histogram (0-255)')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.xlim([0, 255])

    # Highlighted region (200-255)
    plt.subplot(132)
    plt.plot(hist, color='black')
    plt.title('Highlighted Range (200-255)')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.xlim([100, 255])
    plt.axvspan(100, 255, color='yellow', alpha=0.3)  # Highlight area

    # Zoomed-in view (200-255)
    plt.subplot(133)
    plt.plot(hist[150:256], color='red')
    plt.title('Zoomed View (200-255)')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.xticks(np.arange(0, 56, 5), np.arange(150, 206, 5))  # Correct x-axis labels

    plt.tight_layout()
    plt.show()


