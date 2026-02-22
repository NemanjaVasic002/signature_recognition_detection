import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from nn import SignatureCNN  # Make sure nn.py is in the same directory or Python path
from datasetsCreation import SignatureDataset  # Your custom dataset class


def test(csvFile,datasetLoc):
    from datasetsCreation import resizeSignatures
    # Define transform
    transform = transforms.Compose([
        transforms.ToTensor()
    ])


    test_dataset = SignatureDataset(csv_file=csvFile, root_dir=datasetLoc, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Load model and weights ✅
    model = SignatureCNN()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load("model526.pth", map_location=device))  # ✅ Load saved weights
    model.to(device)
    model.eval()  # Set model to evaluation mode
    # Test loop
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)


            outputs = model(images)
            preds = (outputs >= 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            print("Predictions:", preds.view(-1).cpu().numpy())
            print("Ground Truth:", labels.view(-1).cpu().numpy())
    accuracy = 100 * correct / total
    print("Accuracy:", accuracy)