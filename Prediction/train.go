package prediction

import (
	"encoding/csv"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"strconv"
	"time"
)

type Beta struct {
	Intercept float64
	Slope     []float64
}

var M Beta

type Mat struct {
	// Independent Variables
	X1 []float64
	X2 []float64
	X3 []float64
}

func (M *Beta) FitModel(x Mat, y []float64) {

	if len(x.X1) != len(x.X2) || len(x.X1) != len(y) {
		err := fmt.Errorf("Size of Independent and Dependent variable are not equal")
		panic(err)
	}

	// Below is coefficient
	/*
		Coefficients (ββ):
		β^=(XTX)−1XTyβ^=(XTX)−1XTy
	*/

	// Pass Struct to MergeMat to convert 1D array to 2D array
	X := MergeMat(x)

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
	/*
		InterCept Formula
		β₀ = (1/n) * Σ(y - β₁x₁ - β₂x₂ - ... - βₚxₚ)
	*/

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

func (M *Beta) PredictModel(x Mat, actual []float64) ([]float64, error) {

	// Y = m1*x1 + m2*x2 + ....... + mn * xn + c + e
	X := MergeMat(x)
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

	meanActual := calculateMean(actual)

	var sse, sst float64

	for i := 0; i < n; i++ {
		// Sum Of Square of Errors
		sse = math.Pow(actual[i]-predict[i], 2)
		// Total Sum Of Squares
		sst = math.Pow(actual[i]-meanActual, 2)
	}

	R := 1 - (sse / sst)

	adjR := 1 - ((1 - R) * float64(n-1) / float64(n-len(M.Slope)-1))

	return R, adjR, nil

}

func (M *Beta) Error(Actual []float64, Predicted []float64) (float64, float64) {
	if len(Actual) != len(Predicted) {
		log.Fatal("Size of actual and predicted not equal")
	}

	n := len(Actual)
	var sum float64
	for i := range Actual {
		sum += math.Pow((Actual[i] - Predicted[i]), 2)
	}
	meanSquaredError := sum / float64(n)
	RMSE := math.Sqrt(sum / float64(n))

	return RMSE, meanSquaredError
}

func (M *Beta) SplitDataset(data [][]string, numFolds int, foldIndex int) ([][]string, [][]string) {

	folds := SplitIntoKFolds(data, numFolds)

	testFold := foldIndex % numFolds

	testSet := folds[testFold]

	rand.Seed(time.Now().UnixNano())
	rand.Shuffle(len(data), func(i, j int) { data[i], data[j] = data[j], data[i] })

	var trainSet [][]string
	for i, fold := range folds {
		if i != testFold {
			trainSet = append(trainSet, fold...)
		}
	}

	return trainSet, testSet
}

func SplitIntoKFolds(data [][]string, numFolds int) [][][]string {

	shuffleData := make([][]string, len(data))
	copy(shuffleData, data)

	rand.Seed(time.Now().UnixNano())
	rand.Shuffle(len(shuffleData), func(i, j int) { shuffleData[i], shuffleData[j] = shuffleData[j], shuffleData[i] })

	foldSize := len(data) / numFolds

	folds := make([][][]string, numFolds)

	for i := 0; i < numFolds; i++ {
		start := i * foldSize
		end := start + foldSize
		if i == numFolds-1 {
			end = len(data)
		}
		folds[i] = shuffleData[start:end]
	}

	return folds
}

func (M *Beta) PrepareData(records [][]string) ([]float64, []float64, []float64, []float64) {

	x1 := make([]float64, 0)
	x2 := make([]float64, 0)
	x3 := make([]float64, 0)
	y := make([]float64, 0)

	for _, record := range records {
		x1Val, _ := strconv.ParseFloat(record[1], 64)
		x2Val, _ := strconv.ParseFloat(record[3], 64)
		x3Val, _ := strconv.ParseFloat(record[4], 64)
		yVal, _ := strconv.ParseFloat(record[0], 64)

		x1 = append(x1, x1Val)
		x2 = append(x2, x2Val)
		x3 = append(x3, x3Val)
		y = append(y, yVal)
	}

	return x1, x2, x3, y
}

func (M *Beta) LoadCsv(fileName string) ([][]string, error) {

	file, err := os.Open(fileName)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	reader := csv.NewReader(file)

	records, err := reader.ReadAll()
	if err != nil {
		return nil, err
	}

	return records, nil
}

func (M *Beta) SummaryStats(records [][]string) {
	headers := records[0]
	data := records[1:]

	numCols := make(map[int]bool)
	for i, val := range headers {
		if _, err := strconv.ParseFloat(val, 64); err == nil {
			numCols[i] = true
		}
	}

	for i, col := range headers {
		if numCols[i] {
			values := make([]float64, len(data))

			for j, row := range data {
				val, _ := strconv.ParseFloat(row[i], 64)
				values[j] = val
			}

			mean, min, max := calculateStats(values)
			fmt.Printf("Column %s, Mean %2f  Min %.2f  Max %.2f\n", col, mean, min, max)
		}
	}
}

func calculateStats(data []float64) (mean, min, max float64) {
	n := len(data)

	if n == 0 {
		return 0, 0, 0
	}

	sum := 0.0
	min = data[0]
	max = data[0]

	for _, val := range data {
		sum += val
		if val < min {
			min = val
		}
		if val > max {
			max = val
		}
	}

	mean = sum / float64(n)
	return mean, min, max

}

func (M *Beta) Normalize(x []float64) []float64 {
	mean := calculateMean(x)
	stdDev := calculateStdDev(x, mean)

	normalized := make([]float64, len(x))
	for i, val := range x {
		normalized[i] = (val - mean) / stdDev
	}
	return normalized
}

// Function to calculate the mean of a feature vector
func calculateMean(x []float64) float64 {
	sum := 0.0
	for _, val := range x {
		sum += val
	}
	return sum / float64(len(x))
}

// Function to calculate the standard deviation of a feature vector
func calculateStdDev(x []float64, mean float64) float64 {
	sumSqDiff := 0.0
	for _, val := range x {
		diff := val - mean
		sumSqDiff += diff * diff
	}
	return math.Sqrt(sumSqDiff / float64(len(x)))
}
