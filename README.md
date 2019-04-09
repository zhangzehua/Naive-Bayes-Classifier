# Naive-Bayes-Classifier
Using Naive Bayes Classifier to distinguish article of different author or different language
## Bernoulli Naive Bayes Classfier
Using Bernoulli Naive Bayes Classfier to distinguish article of different author, using stop words.  
Input: 
  * Train and test set
  * stop words 
  * ground truth table
  
Output:
  * Accuarcy
  * Confusion Matrix
  * Feature Ranking of top 20 by their class-conditional entropy
  * Feature Curve 
  
Run like this: python3 Bernoulli_Naive_Bayes.py problemA/

## Multinomial
Using Multinomial Naive Bayes Classfier to distinguish article of different language. Using both bigram and trigram.  
Input: 
  * Train and test set 

Output: 
  * Accuarcy
  * Confusion Matrix 
  
Run like this: python3 Multinomial_Naive_Bayes.py extra-test/  
  
NOTE:The dataset we use is a subset of the Ad-hoc Authorship Attribution Competition(AAAC) Dataset, publicly available at http://www.mathcs.duq.edu/~juola/authorship_materials2.html.
