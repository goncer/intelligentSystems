getwd()
setwd("/home/gonzo/workspace/R/intelligent/intelligentSystems/unit2")
library(rJava)
.jinit(parameters="-Xmx4g")
# If there are more memory problems, invoke gc() after the POS tagging

library(openNLP) 
#install.packages("http://datacube.wu.ac.at/src/contrib/openNLPmodels.en_1.5-1.tar.gz",repos=NULL, type="source")
library(openNLPmodels.en)
library(tm)

##Functions
getAnnotationsFromDocument = function(doc){
  x=as.String(doc)
  sent_token_annotator <- Maxent_Sent_Token_Annotator()
  word_token_annotator <- Maxent_Word_Token_Annotator()
  pos_tag_annotator <- Maxent_POS_Tag_Annotator()
  y1 <- annotate(x, list(sent_token_annotator, word_token_annotator))
  y2 <- annotate(x, pos_tag_annotator, y1)
  parse_annotator <- Parse_Annotator()
  y3 <- annotate(x, parse_annotator, y2)
  return(y3)  
} 

getAnnotatedMergedDocument = function(doc,annotations){
  x=as.String(doc)
  y2w <- subset(annotations, type == "word")
  tags <- sapply(y2w$features, '[[', "POS")
  r1 <- sprintf("%s/%s", x[y2w], tags)
  r2 <- paste(r1, collapse = " ")
  return(r2)  
} 

getAnnotatedPlainTextDocument = function(doc,annotations){
  x=as.String(doc)
  a = AnnotatedPlainTextDocument(x,annotations)
  return(a)  
}

##Only  GET: CERRO MORERA GONZALO DEL  : cv420_* - cv429_*
source.pos = DirSource("/home/gonzo/workspace/R/intelligent/intelligentSystems/unit2/txt_sentoken/pos",
                       encoding = "UTF-8", pattern="(cv42[0-9]_[0-9]+.\\w*)")
corpus = Corpus(source.pos)
corpus[[1]]
annotations = lapply(corpus, getAnnotationsFromDocument)

head(annotations[[1]])
tail(annotations[[1]])


corpus.tagged = Map(getAnnotatedPlainTextDocument, corpus, annotations)
corpus.tagged[[1]] 

corpus.taggedText = Map(getAnnotatedMergedDocument, corpus, annotations)
corpus.taggedText[[1]] 

#Geet the first document
doc = corpus.tagged[[1]] 
doc

strsplit(as.character(doc), "#", fixed = FALSE, perl = FALSE, useBytes = FALSE)[1]

head(words(doc),2)
sentences = 2
head(sents(doc),sentences)
head(tagged_words(doc))
head(tagged_sents(doc),2)[1]
head(parsed_sents(doc),sentences)
asd = tagged_sents(doc)[2]
length (asd)
slices <- c(58 ,4)
lbls <- c("True positive ", "False Positive")
pct <- round(slices/sum(slices)*100)
lbls <- paste(lbls, pct) # add percents to labels
lbls <- paste(lbls,"%",sep="") # ad % to labels 
pie(slices, labels = lbls, main="Pie Chart of Countries")

precision = 58 / (58 + 4)
precision = 58 / (58 + 4) 
precision
