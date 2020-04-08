library(caret)
library(ROCR)

set.seed(1)

dados = read_xlsx("C:/Users/Pedro/Desktop/ME_610/set.xlsx")

#splitando o banco em treino e teste

n = nrow(dados)
trainIdx = createDataPartition(y=dados$default, p=0.7, list=FALSE)

treino = dados[trainIdx, ]
teste = dados[-trainIdx, ]

#Padronização

preProcValues = preProcess(treino, method = c("center", "scale"))
train = predict(preProcValues, treino)
test = predict(preProcValues, teste)


#Validação Cruzada para o KNN

train.control = trainControl(method="cv", number=10, repeats=3)

#Ajustando KNN

kgrid = data.frame(k=round(seq(1, 50, by = 2)))
knnfit = train(default ~ ., data = train, method = "knn",
                preProcess=c("center", "scale"), tuneGrid=kgrid, 
                trControl = train.control)

# Número òtimo de vizinhos 

knn.fit$bestTune

# Validação do modelo na base de teste

knn.pred = predict(knnfit, newdata=test)
confusion.knn = table(ValorPredito=knn.pred, ValorReal=test$default)


# Validação Cruzada para a regressao logistica

train.control = trainControl(method = 'cv', classProbs = TRUE, 
                              summaryFunction = twoClassSummary, savePredictions = TRUE)

# Ajustando a regressão logística

lr.fit = train(default ~ ., data=train, method='glm', family='binomial', 
                trControl=train.control)

# Validação do modelo na base de teste

lr.pred = predict(lr.fit, newdata=test)
confusion.knn = table(ValorPredito=lr.pred, ValorReal=test$default)

#Validação Cruzada para a floresta aleatoria

control = trainControl(method="repeatedcv", number=10, repeats=3, search="grid")
tunegrid = expand.grid(.mtry=c(1:15))
floresta = train(default ~ ., data=train, method="rf", tuneGrid=tunegrid, trControl=control)



# Analisando a Curva ROC da regressão logistica

lr.pred.p = predict(lr.fit, newdata=test, type="prob")[,"Yes"]
ROCRPred = prediction(lr.pred.p, test$default)
ROCRPerf = performance(ROCRPred, "tpr", "fpr")
plot(ROCRPerf, colorize=TRUE, print.cutoffs.at=seq(0.1, by=0.2))