package main

import (
	"fmt"
	"log"
	"os"

	prediction "main.go/Prediction"
)

func main() {
	fmt.Println("................. Multiple Linear Regression Model .................")

	Model := prediction.Beta{}
	// Read the csv file
	// NTPC
	getFile, err := Model.LoadCsv("Delhi.csv")
	if err != nil {
		log.Fatalf("Error while opening the file : %v", err)
		os.Exit(1)
	}

	Model.SummaryStats(getFile)

	// Split the dataset into Train and Test part
	trainRecords, testRecords := Model.SplitDataset(getFile, 10, 8)

	x1Train, x2Train, x3Train, yTrain := Model.PrepareData(trainRecords)

	x1Test, x2Test, x3Test, yTest := Model.PrepareData(testRecords)

	x1Train = Model.Normalize(x1Train)
	x2Train = Model.Normalize(x2Train)
	x3Train = Model.Normalize(x3Train)
	x1Test = Model.Normalize(x1Test)
	x2Test = Model.Normalize(x2Test)
	x3Test = Model.Normalize(x3Test)

	var trainData, testData prediction.Mat

	trainData.X1, trainData.X2, trainData.X3 = x1Train, x2Train, x3Train

	testData.X1, testData.X2, testData.X3 = x1Test, x2Test, x3Test

	fmt.Println()

	Model.FitModel(trainData, yTrain)

	PredictedModel, err := Model.PredictModel(testData, yTest)
	if err != nil {
		fmt.Println("Error while predicting")
	}

	Model.PlotGraph(yTest, yTrain, "Previous.png")

	Model.PlotGraph(yTest, PredictedModel, "prediction.png")
	Error1, Error2 := Model.Error(yTest, PredictedModel)

	r2_score, adj_R2_score, err := Model.Rsquare(yTest, PredictedModel)
	if err != nil {
		fmt.Println("Performance error", err)
	}
	if r2_score > 0 && adj_R2_score > 0 {
		fmt.Println("R2 Score( MODEL PERFORMANCE ) :", r2_score, "%", "\t\tAdjusted R2 score :", adj_R2_score)
	} else {
		fmt.Println("Poor Performance")
	}

	fmt.Println("RMSE :", Error1, "MSE :", Error2)
}
