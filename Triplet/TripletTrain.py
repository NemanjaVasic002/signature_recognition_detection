import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from TripletDataSet import SignatureDataset, transform
from TripletNN import TripletSignatureNet


def training(csvFile, datasetLocation, totalEpochs):
    signatureDataset = SignatureDataset(csvFile=csvFile, rootDirectory=datasetLocation, imageTransform=transform)
    trainLoader = DataLoader(signatureDataset, batch_size=4, shuffle=True)

    siameseModel = TripletSignatureNet()
    computingDevice = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    siameseModel.to(computingDevice)

    lossCriterion = nn.TripletMarginLoss(margin=0.5)
    modelOptimizer = optim.Adam(siameseModel.parameters(), lr=0.001)

    lossHistory = []

    siameseModel.train()
    for epoch in range(totalEpochs):
        runningLoss = 0.0
        for anchor, positive, negative in trainLoader:
            anchor = anchor.to(computingDevice)
            positive = positive.to(computingDevice)
            negative = negative.to(computingDevice)

            modelOptimizer.zero_grad()

            anchorOut, positiveOut, negativeOut = siameseModel(anchor, positive, negative)

            currentLoss = lossCriterion(anchorOut, positiveOut, negativeOut)
            currentLoss.backward()
            modelOptimizer.step()

            runningLoss += currentLoss.item()

        averageLoss = runningLoss / len(trainLoader)
        lossHistory.append(averageLoss)
        print(f"Epoch {epoch + 1}/{totalEpochs}, Loss: {averageLoss}")

    with open("Triplet_Loss_History.csv", "w") as f:
        f.write("Epoha,Gubitak\n")
        for i, value in enumerate(lossHistory):
            f.write(f"{i + 1},{value}\n")

    torch.save(siameseModel.state_dict(), "triplet_model.pth")