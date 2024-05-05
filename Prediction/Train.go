package prediction

import (
	"fmt"
	"log"
	"math"
)

type Beta struct {
	Intercept float64
	Slope     []float64
}

// Y = m1*x1 + m2*x2 + ....... + mn * xn + c + e

var M Beta

type Mat struct {
	X1 []float64
	X2 []float64
	X3 []float64
}

/*
InterCept Formula
β₀ = (1/n) * Σ(y - β₁x₁ - β₂x₂ - ... - βₚxₚ)

Coefficients (ββ):
β^=(XTX)−1XTyβ^=(XTX)−1XTy
*/
func (M *Beta) FitModel(x Mat, y []float64) {

	if len(x.X1) != len(x.X2) || len(x.X1) != len(y) {
		err := fmt.Errorf("Size of Independent and Dependent variable are not equal")
		panic(err)
	}

	// Below is coefficient
	X := MergeData(x)

	XTx, err := MulNM(Transpose(X), X)
	if err != nil {
		fmt.Println("Getting error while applying N*M MUltiplication method", err)
		return
	}

	inverseMatrix, err := LUInverse(XTx)
	if err != nil {
		log.Fatal(err)
		return
	}

	XTX_XT, err := MulNM(inverseMatrix, Transpose(X))
	if err != nil {
		return
	}

	slope, err := MulN(XTX_XT, y)
	if err != nil {
		fmt.Println("Getting error while applying mul mehtod", err)
		return
	}
	M.Slope = make([]float64, len(slope))

	for i := range slope {
		M.Slope[i] = slope[i]
	}

	for _, v := range M.Slope {
		fmt.Println("Slopes are ", v)
	}

	// Below is Intercept

	n := len(X)

	var intercept float64
	for i := 0; i < n; i++ {
		intercept += y[i]
		for j := 0; j < len(X[0]); j++ {
			//sumCoefficient += M.Slope[j] * X[i][j]
			intercept -= M.Slope[j] * X[i][j]
		}
	}
	M.Intercept = intercept / float64(n)

	fmt.Println("Intercept point is ", M.Intercept)
}

func (M *Beta) Prediction(x Mat, actual []float64) ([]float64, error) {

	// Y = m1*x1 + m2*x2 + ....... + mn * xn + c + e
	X := MergeData(x)
	n := len(X)

	y := make([]float64, n)
	residuals := make([]float64, n)
	for i := 0; i < n; i++ {
		rowPrediction := M.Intercept
		for j := 0; j < len(X[0]); j++ {
			rowPrediction += M.Slope[j] * X[i][j]
		}

		y[i] = rowPrediction
		residuals[i] = actual[i] - y[i]
	}

	return y, nil
}

func (M *Beta) Rsquare(actual []float64, predict []float64) (float64, float64, error) {

	if len(actual) != len(predict) {
		log.Fatal("error")
	}

	n := len(actual)

	var sse, sst float64

	for i := 0; i < n; i++ {
		sse += math.Pow(actual[i]-predict[i], 2)
		sst += math.Pow(actual[i]-M.Intercept, 2)
	}

	R := 1 - (sse / sst)

	adjR := 1 - (1-R)*float64(n-1)/float64(n-len(M.Slope)-1)

	return R, adjR, nil

}

func (M *Beta) RootMeanSquare(Actual []float64, Predicted []float64) (float64, float64) {
	if len(Actual) != len(Predicted) {
		log.Fatal("Size of actual and predicted not equal")
	}

	n := len(Actual)
	var sum float64
	for i, _ := range Actual {
		sum += math.Pow((Actual[i] - Predicted[i]), 2)
	}
	meanSquaredError := sum / float64(n)
	RMSE := math.Sqrt(sum / float64(n))

	return RMSE, meanSquaredError
}
