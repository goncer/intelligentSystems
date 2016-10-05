setwd("/home/gonzo/workspace/R/intelligent/intelligentSystems/unit2")
library(tm)
library(ggplot2)
library(wordcloud)
library(RWeka)
library(reshape2)
library(SnowballC)
options(mc.cores=1)
source.pos = DirSource("/home/gonzo/workspace/R/intelligent/intelligentSystems/unit2/txt_sentoken/pos", encoding = "UTF-8")
corpus = Corpus(source.pos)
length(corpus)
summary(corpus[1:3])
corpus[1]
corpus[[1]]
meta(corpus[[1]])
meta(corpus[[1]])$id
tdm = TermDocumentMatrix(corpus)
tdm
inspect(tdm[2000:2003,100:103])
length(dimnames(tdm)$Terms)
head(dimnames(tdm)$Terms,10)
tail(dimnames(tdm)$Terms,10)
freq=rowSums(as.matrix(tdm))
head(freq,10)
tail(freq,10)
plot(sort(freq, decreasing = T),col="blue",main="Word frequencies", xlab="Frequency-based rank", ylab = "Frequency")
tail(sort(freq),n=10)
sum(freq == 1)
getTransformations()
doc=corpus[1]
doc[[1]]$content[1]
stopwords()

doc = tm_map(doc,removeWords,stopwords())
doc[[1]]$content[1]
doc = tm_map(corpus[1],removePunctuation)
doc[[1]]$content[1]

doc = tm_map(doc,removeNumbers)
doc[[1]]$content[1]

doc = tm_map(doc,stripWhitespace)
doc[[1]]$content[1]

doc = tm_map(doc,stemDocument)
doc[[1]]$content[1]

# transformation!
tdm = TermDocumentMatrix(corpus,control=list(stopwords = T,removePunctuation = T,removeNumbers = T,stemming = T))
tdm

inspect(tdm[2000:2003,100:103])
length(dimnames(tdm)$Terms)
head(dimnames(tdm)$Terms,10)
tail(dimnames(tdm)$Terms,10)
#frequencies!!
freq=rowSums(as.matrix(tdm))
head(freq,10)
tail(freq,10)
plot(sort(freq, decreasing = T),col="blue",main="Word frequencies", xlab="Frequency-based rank", ylab = "Frequency")
tail(sort(freq),n=10)
sum(freq == 1)


#####Create TDM with transformations and custom stopwords###
doc = corpus[1]
doc[[1]]$content[1]
myStopwords = c(stopwords(),"film","films","movie","movies")
doc = tm_map(corpus[1],removeWords,myStopwords)
doc[[1]]$content[1]
tdm = TermDocumentMatrix(corpus,
                         control=list(stopwords = myStopwords,
                                      removePunctuation = T, 
                                      removeNumbers = T,
                                      stemming = T))


tdm
inspect(tdm[2000:2003,100:103])
length(dimnames(tdm)$Terms)
head(dimnames(tdm)$Terms,10)
tail(dimnames(tdm)$Terms,10)
freq=rowSums(as.matrix(tdm))
head(freq,10)
tail(freq,10)
plot(sort(freq, decreasing = T),col="blue",main="Word frequencies", xlab="Frequency-based rank", ylab = "Frequency")
# Ten most frequent terms
tail(sort(freq),n=10)
sum(freq == 1)

####barplot@@
high.freq=tail(sort(freq),n=10)
hfp.df=as.data.frame(sort(high.freq))
hfp.df$names <- rownames(hfp.df) 
ggplot(hfp.df, aes(reorder(names,high.freq), high.freq)) +
  geom_bar(stat="identity") + coord_flip() + 
  xlab("Terms") + ylab("Frequency") +
  ggtitle("Term frequencies")


####Create a TDM with TF-IDF weights
tdm.tfidf = TermDocumentMatrix(corpus,
                               control = list(weighting = weightTfIdf,
                                              stopwords = myStopwords, 
                                              removePunctuation = T,
                                              removeNumbers = T,
                                              stemming = T))
tdm.tfidf
freq=rowSums(as.matrix(tdm.tfidf))

plot(sort(freq, decreasing = T),col="blue",main="Word TF-IDF frequencies", xlab="TF-IDF-based rank", ylab = "TF-IDF")
tail(sort(freq),n=10)


#####Make an association analysis####3
asoc.star = as.data.frame(findAssocs(tdm,"star", 0.5))
asoc.star$names <- rownames(asoc.star) 
asoc.star
ggplot(asoc.star, aes(reorder(names,star), star)) +   
  geom_bar(stat="identity") + coord_flip() + 
  xlab("Terms") + ylab("Correlation") +
  ggtitle("\"star\" associations")


asoc.indi = as.data.frame(findAssocs(tdm,"indiana", 0.5))
asoc.indi$names <- rownames(asoc.indi) 
asoc.indi
ggplot(asoc.indi, aes(reorder(names,indiana), indiana)) +   
  geom_bar(stat="identity") + coord_flip() + 
  xlab("Terms") + ylab("Correlation") +
  ggtitle("\"indiana\" associations")

####Create a word-document frequency graph

tdm.small = removeSparseTerms(tdm,0.5)
dim(tdm.small)
tdm.small
inspect(tdm.small[1:4,1:4])
matrix.tdm = melt(as.matrix(tdm.small), value.name = "count")
head(matrix.tdm)

ggplot(matrix.tdm, aes(x = Docs, y = Terms, fill = log10(count))) +
  geom_tile(colour = "white") +
  scale_fill_gradient(high="#FF0000" , low="#FFFFFF")+
  ylab("Terms") +
  theme(panel.background = element_blank()) +
  theme(axis.text.x = element_blank(), axis.ticks.x = element_blank())

####Create a word cloud

pal=brewer.pal(8,"Blues")
pal=pal[-(1:3)]

freq = sort(rowSums(as.matrix(tdm)), decreasing = T)
set.seed(1234)
word.cloud=wordcloud(words=names(freq), freq=freq,
                     min.freq=400, random.order=F, colors=pal)


###Create a bigram wordcloud
corpus.ng = tm_map(corpus,removeWords,c(stopwords(),"s","ve"))
corpus.ng = tm_map(corpus.ng,removePunctuation)
corpus.ng = tm_map(corpus.ng,removeNumbers)

BigramTokenizer <- function(x) NGramTokenizer(x, Weka_control(min = 2, max = 2))
tdm.bigram = TermDocumentMatrix(corpus.ng,
                                control = list(tokenize = BigramTokenizer))

freq = sort(rowSums(as.matrix(tdm.bigram)),decreasing = TRUE)
freq.df = data.frame(word=names(freq), freq=freq)
head(freq.df, 20)
wordcloud(freq.df$word,freq.df$freq,max.words=100,random.order = F, colors=pal)


ggplot(head(freq.df,15), aes(reorder(word,freq), freq)) +   
  geom_bar(stat="identity") + coord_flip() + 
  xlab("Bigrams") + ylab("Frequency") +
  ggtitle("Most frequent bigrams")


####Create a trigram wordcloud
TrigramTokenizer <- function(x) NGramTokenizer(x, Weka_control(min = 3, max = 3))
tdm.trigram = TermDocumentMatrix(corpus.ng,
                                 control = list(tokenize = TrigramTokenizer))

freq = sort(rowSums(as.matrix(tdm.trigram)),decreasing = TRUE)
freq.df = data.frame(word=names(freq), freq=freq)
head(freq.df, 20)
wordcloud(freq.df$word,freq.df$freq,max.words=100,random.order = F, colors=pal)
ggplot(head(freq.df,15), aes(reorder(word,freq), freq)) +   
  geom_bar(stat="identity") + coord_flip() + 
  xlab("Trigrams") + ylab("Frequency") +
  ggtitle("Most frequent trigrams")
