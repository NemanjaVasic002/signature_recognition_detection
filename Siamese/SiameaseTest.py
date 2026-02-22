import torch
from torch.utils.data import DataLoader
from SiameseNN import SiameseSignatureNet
from DataSet import SignatureDataset, transform


def test(csvFile, datasetLocation):
    testDataset = SignatureDataset(csvFile=csvFile, rootDirectory=datasetLocation, imageTransform=transform)
    testLoader = DataLoader(testDataset, batch_size=1, shuffle=False)

    siameseModel = SiameseSignatureNet()
    computingDevice = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    siameseModel.load_state_dict(torch.load("siamese_model.pth", map_location=computingDevice, weights_only=True))
    siameseModel.to(computingDevice)
    siameseModel.eval()

    correctPredictions = 0
    totalSamples = 0

    with torch.no_grad():
        for imageOne, imageTwo, groundTruthLabels in testLoader:
            imageOne, imageTwo, groundTruthLabels = imageOne.to(computingDevice), \
                                                     imageTwo.to(computingDevice), \
                                                     groundTruthLabels.to(computingDevice).unsqueeze(1)

            modelOutputs = siameseModel(imageOne, imageTwo)
            modelPredictions = (modelOutputs >= 0.5).float()

            correctPredictions += (modelPredictions == groundTruthLabels).sum().item()
            totalSamples += groundTruthLabels.size(0)

            print(f"Pred: {modelPredictions.item()} | GT: {groundTruthLabels.item()}")

    print(f"Accuracy: {100 * correctPredictions / totalSamples}%")