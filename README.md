### Stock Price Prediction using Multiple Linear Regression in Go Lang

## No External Libraries are used for Model building

## Built from scratch.

### Once Cloned

do update the mod file

```bash
go mod tidy
```

### To choose Independent and Dependent variable

Go to "PrepareData func" in Prediction/train.go
![GitHub Image](images/PrepareData.png)
You will see like this, inside "for loop (Ex : x1[0],x2[0])"
" Here x represents Independent variable and Y is for Dependent variable"
Change indeces numbers with your acutal index numbers ( i.e Your Datasets variable index number)

### To run model?

```bash
go run .  or go run main.go
```
