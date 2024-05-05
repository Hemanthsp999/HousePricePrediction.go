package prediction

import (
	"fmt"
	"log"
)

func MergeData(x Mat) [][]float64 {

	// Get the Mat( x1, x2) Structure and Bind it to 2d array
	n := len(x.X1)
	MergeMat := make([][]float64, n)

	for i := range MergeMat {
		MergeMat[i] = []float64{x.X1[i], x.X2[i],x.X3[i]}
	}

	return MergeMat
}

func Transpose(x [][]float64) [][]float64 {

	n := len(x)
	m := len(x[0])

	// Each Columns in the Original Matrix should be equal to each Row in the Transpose Matrix
	TransposeMatrix := make([][]float64, m)

	for i := range TransposeMatrix {
		// Each row has lenth equal to number of row in original matrix
		TransposeMatrix[i] = make([]float64, n)
	}

	for i := 0; i < n; i++ {
		for j := 0; j < m; j++ {
			TransposeMatrix[j][i] = x[i][j]
		}
	}

	return TransposeMatrix

}

func MulNM(x [][]float64, y [][]float64) ([][]float64, error) {
	rowX := len(x)
	colX := len(x[0])
	colY := len(y[0])

	// Here column  of X matrix should be equal to row of Y matrix
	// Otherwise it violates the matrix multiplication rule
	if colX != len(y) {
		log.Print("Column is not equal to row and not following Matrix Mul rules")
	}

	Product := make([][]float64, rowX)
	for i := range Product {
		Product[i] = make([]float64, colY)
	}

	for i := 0; i < rowX; i++ {
		for j := 0; j < colY; j++ {
			for k := 0; k < colX; k++ {
				Product[i][j] += x[i][k] * y[k][j]
			}
		}
	}

	return Product, nil
}
func MulN(x [][]float64, y []float64) ([]float64, error) {

	Product := make([]float64, len(x))
	for i := 0; i < len(x); i++ {
		for j := 0; j < len(y); j++ {
			Product[i] += x[i][j] * y[j]
		}
	}

	return Product, nil
}

func LUInverse(matrix [][]float64) ([][]float64, error) {
	n := len(matrix)
	if n == 0 || len(matrix[0]) != n {
		return nil, fmt.Errorf("input matrix must be square")
	}

	// Initialize the identity matrix
	identity := make([][]float64, n)
	for i := range identity {
		identity[i] = make([]float64, n)
		identity[i][i] = 1
	}

	// Perform LU decomposition
	L, U, err := LUDecomposition(matrix)
	if err != nil {
		return nil, err
	}

	// Solve LY = I for Y
	Y := make([][]float64, n)
	for i := 0; i < n; i++ {
		Y[i] = make([]float64, n)
		for j := 0; j < n; j++ {
			if i == j {
				Y[i][j] = 1
			}
		}
	}
	for j := 0; j < n; j++ {
		for i := j + 1; i < n; i++ {
			for k := 0; k < n; k++ {
				Y[i][k] -= Y[j][k] * L[i][j]
			}
		}
	}

	// Solve UX = Y for X (the inverse)
	X := make([][]float64, n)
	for i := 0; i < n; i++ {
		X[i] = make([]float64, n)
	}
	for j := n - 1; j >= 0; j-- {
		for i := 0; i < n; i++ {
			X[i][j] = Y[i][j]
			for k := j + 1; k < n; k++ {
				X[i][j] -= X[i][k] * U[j][k]
			}
			X[i][j] /= U[j][j]
		}
	}

	return X, nil
}

func LUDecomposition(matrix [][]float64) ([][]float64, [][]float64, error) {
	n := len(matrix)
	L := make([][]float64, n)
	U := make([][]float64, n)
	for i := range L {
		L[i] = make([]float64, n)
		U[i] = make([]float64, n)
	}

	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			if j < i {
				L[j][i] = 0
				U[j][i] = matrix[j][i]
				for k := 0; k < j; k++ {
					U[j][i] -= L[j][k] * U[k][i]
				}
				U[j][i] /= L[j][j]
			} else if j == i {
				L[j][i] = 1
				U[j][i] = matrix[j][i]
				for k := 0; k < j; k++ {
					L[j][i] -= L[j][k] * U[k][i]
				}
			} else {
				L[j][i] = matrix[j][i]
				U[j][i] = 0
				for k := 0; k < i; k++ {
					L[j][i] -= L[j][k] * U[k][i]
				}
				L[j][i] /= U[i][i]
			}
		}
	}

	return L, U, nil
}
