from SiameaseTrain import training
from SiameaseTest import test

# Postavke
csvTrain = r"D:\Projekti\PythonProject\.venv\ADO\NoviNeuarlanArhitektura\csv\train5.csv"
csvTest =r"D:\Projekti\PythonProject\.venv\ADO\NoviNeuarlanArhitektura\csv\test5.csv"
testData = r"D:\Projekti\PythonProject\.venv\ADO\verification\test5"
trainData = r"D:\Projekti\PythonProject\.venv\ADO\verification\training5"
epochs = 26

if __name__ == "__main__":


    training(csvTrain, trainData, epochs)


    test(csvTest, testData)