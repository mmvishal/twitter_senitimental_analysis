library(caret)
library(descr)
library(NLP)
library(tm)
library(SnowballC)

#reading the dataset
tweets <- read.csv("/Users/mallamehervishal/Downloads/tweets.csv")
#analysing the tweets dataset
#invokes the sprea-sheet for the object
View(tweets)
#retrieves the dimension of the object
dim(tweets)
#prints frequency of the object
freq(tweets)
#displays internal structure of th object
str(tweets)

#encoding tweets$labels into vector
tweets$label <- factor(tweets$label)
#cross classifies the object
table(tweets$label)

#converting tweets$twweet into documents for data cleaning
tweetscorpus <- Corpus(VectorSource(tweets$tweet))
#displays detailed information about CORPUS
inspect(tweetscorpus)

#converting all corpus data into LOWER CASE
tweetscorpus <- tm_map(tweetscorpus, tolower)

#removes stopwords from corpus
tweetscorpus <- tm_map(tweetscorpus, removeWords, stopwords("english"))

#removes punctuation
tweetscorpus <- tm_map(tweetscorpus, removePunctuation)

#removes stem words
tweetscorpus <- tm_map(tweetscorpus, stemDocument)

#removes numbers
tweetscorpus <- tm_map(tweetscorpus, removeNumbers)

#removes extra white spaces from the corpus
tweetscorpus <- tm_map(tweetscorpus, stripWhitespace)

#constructs term-document matrix 
tweetscorpus_dtm <- DocumentTermMatrix(tweetscorpus) #,control = list(tolower = TRUE, removeNumbers = TRUE, stopwords = TRUE, removePunctuation = TRUE,stemming = TRUE))

#Invoke a spreadsheet-style data viewer for tweetscorpus_dtm.
View(tweetscorpus_dtm)

#splitting the converted corpus dataset
#train dataset
traint <- tweetscorpus_dtm[1:20000,]
dim(traint)
View(traint)
#test dataset
testt <- tweetscorpus_dtm[20001:31962,]
dim(testt)
View(testt)

#trainlabels dataset splitting
traintlabels <- tweets[1:20000,]$label
#invokes spread sheet for traintlabels object
View(traintlabels)

#testlabels dataset splitting
testtlabels <- tweets[20001:31962,]$label
#invokes spread sheet for testtlabels object
View(testtlabels)

#removes non-zero elements from traint
tweetscorpus_dtm_freq_train<-removeSparseTerms(traint,0.999)
#removes non-zero elements from testt
tweetscorpus_dtm_freq_test<-removeSparseTerms(testt,0.999)

#finds frequent terms which is repeated 3 or more times from the traint document
tweets_freq_words <- findFreqTerms(tweetscorpus_dtm_freq_train,3)
tweets_freq_words

#finds frequent terms which is repeated 3 or more times from the testt document
tweets_freq_words2 <- findFreqTerms(tweetscorpus_dtm_freq_test,3)
tweets_freq_words2

tweetscorpus_dtm_freq_train <- tweetscorpus_dtm_freq_train[,tweets_freq_words]
tweetscorpus_dtm_freq_test <- tweetscorpus_dtm_freq_test[,tweets_freq_words2]

#user defined function which counts the tems in the document
convert_counts<-function(x){
  x<-ifelse(x>0,"Yes","No")
}

#applies user-defined function to the margin to tweetscorpus_dtm_freq_train
tweets_train<-apply(tweetscorpus_dtm_freq_train, MARGIN=2, convert_counts)
View(tweets_train)
#applies user-defined function to the margin to tweetscorpus_dtm_freq_test
tweets_test<-apply(tweetscorpus_dtm_freq_test, MARGIN=2, convert_counts)
View(tweets_test)

#this library is used to import package "NAIVE BAYES"
library(e1071)
#creating model for naive bayes and laplace is used for smoothing
tweetnb <- naiveBayes(tweets_train,traintlabels, laplace = 1)
#predicting the label values from created model
tweetpred <- predict(tweetnb, tweets_test)

#confusion matrix between predicted and test labels
CrossTable(tweetpred, testtlabels, prop.chisq = FALSE, prop.t = FALSE, prop.r = FALSE, dnn = c('predicated','actual'))

#gives the accuracy of the predicted values
confusionMatrix(tweetpred, testtlabels)$overall
