package main

import (
	"encoding/csv"
	"fmt"
	"log"
	"os"
	"strconv"

	prediction "main.go/Prediction"
)

func main() {

	fmt.Println(" Multiple Linear Regression Model : ")

	// Read the csv file
	getFile, err := os.Open("Datasets/BPCL.csv")
	if err != nil {
		log.Fatal("Error while opening the file")
		os.Exit(1)
	}

	readFile := csv.NewReader(getFile)

	_, err = readFile.Read()
	if err != nil {
		log.Fatal("Error while reading the heading")
		os.Exit(2)
	}

	records, err := readFile.ReadAll()
	if err != nil {
		log.Fatal("Error while reading the Values")
		os.Exit(3)
	}

	x1 := make([]float64, 0)
	x2 := make([]float64, 0)
	x3 := make([]float64, 0)

	// Dependent Variable
	y := make([]float64, 0)

	for _, record := range records {
		x1Val, err := strconv.ParseFloat(record[4], 64)
		if err != nil {
			log.Fatal("Error while reading the 1 column")
			os.Exit(10)
		}
		x2Val, err := strconv.ParseFloat(record[5], 64)
		if err != nil {
			log.Fatal("Error while reading the 1 column")
			os.Exit(10)
		}
		x3Val, err := strconv.ParseFloat(record[6], 64)
		if err != nil {
			log.Fatal("Error while reading the 1 column")
			os.Exit(10)
		}
		yVal, err := strconv.ParseFloat(record[8], 64)
		if err != nil {
			log.Fatal("Error while reading the 1 column")
			os.Exit(10)
		}

		x1 = append(x1, x1Val)
		x2 = append(x2, x2Val)
		x3 = append(x3, x3Val)
		y = append(y, yVal)
	}

	// Declare a data of Mat Struct type
	var data prediction.Mat

	data.X1 = x1
	data.X2 = x2
	data.X3 = x3

	var ModelFit prediction.Beta

	// check is there any empty data in columns?
	ModelFit.FitModel(data, y)

	PredictedModel, err := ModelFit.Prediction(data, y)
	if err != nil {
		fmt.Println("Error while predicting")
	}

	for i := range PredictedModel {
		fmt.Println("Actual Value : ", y[i], " Predicted Value : ", PredictedModel[i])
	}

	Error1, Error2 := ModelFit.RootMeanSquare(y, PredictedModel)

	r2_score, adj_R2_score, err := ModelFit.Rsquare(y, PredictedModel)
	if err != nil {
		fmt.Println("Performance error", err)
	}
	fmt.Println("R2 Score( MODEL PERFORMANCE ) :", r2_score, "\t\tAdjusted R2 score :", adj_R2_score)

	fmt.Println("RMSE :", Error1, "MSE :", Error2)
}
