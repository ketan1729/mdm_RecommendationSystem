# Introduction

In recent years, digital technologies have become faster and easily
accessible. Few people have the patience to spend exhaustive time
looking through the internet for a specific piece of information, which
pressures services to provide some sort of system to assist users in
this task. One common approach to providing this is the development of
recommender systems. This can be found on today’s popular applications
such as Netflix and YouTube.  
We have designed and implemented such a system to address two primary
objectives. First, to predict what rating a user would give a movie that
the user hasn’t watched yet and second, using that rating to recommend
the user ten movies that the user would enjoy, based on what similar
users have rated highly.  
We have used the MovieLens dataset in which we processed One million
records and used the attributes - IDs of movies and users, rating out of
five and the timestamp. We performed some analysis by plotting
histograms to see in which years, the concentration of most reviews is.
We have also found the users who have reviewed very few movies and
excluded them from our analysis.  
We have used User-Based CF as a baseline and performed analysis on it
using algorithms KNN, SVD and SVDpp. In addition to performance and
accuracy metrics, we also recorded the training time for the models to
see which models were easy to train and which models were heavy on the
resources.  
We found that SVD and KNN significantly out performed the User-Based CF
algorithm. One of the reasons for this is that users reviewing movies
can depend on their current mindset, the environment in which they
watched it, etc. which influences their rating thus adding noise and
compromising accuracy and performance.  

![Year-wise count of movies rated](MDM%20Project/year-wise.jpg)

# Related Work

In this section, we will go through the research done by people in the
field of Recommendation systems and Collaborative Filtering.

Badrul in \[12\] describes some of the key reasons why the old
Collaborative Filtering is not the efficient way for recommender
systems. They came up with their own system which combines Item based CF
system with KNN algorithm. They have used MovieLens dataset which has
43000 users and they have evaluated on 100,000 ratings. Experimental
results show that for a large dataset, the system proves to be better in
performance than the basic Item based CF system. They have compared
different similarity computation techniques like Cosine-base Similarity,
Adjusted Cosine Similarity, Correlation-based Similarity. Evaluation
metrics like MAE and RMSE are used to prove the results and conclude
that there is a need of smarter recommender systems in the future that
can precisely and correctly give better recommendations to the user.

Authors in \[17\] proposed matrix factorization technique called
singular value decomposition (SVD) in Item based CF system. They have
used SVD to reduce the dimensionality of user-item utility matrix and
came up with an algorithm which uses this lesser dimension information
to predict the ratings and give recommendations. Experimental results
show that this model with SVD gives better results than the original
Item based CF system.

Authors in \[16\] came up with their novel system where the predictions
from User based CF and Item based CF are combined together for a given
user and then they are passed through multiple linear regression and
support vector regression models. They evaluated the data from multiple
datasets (combined and processed together to get around 855,000 ratings)
using MAE and mean absolute percentage error(MAPE) and mean squared
error(MSE).

# Dataset and Preprocessing

We have used Movie Lens dataset having 100,000 reviews given by about
650 users. Each user and movie has a unique id (userId, movieId) and
every user can rate a movie between 0-5. We are using the ratings.csv to
get the userId, movieId and rating fields. After that, we split the data
into 80-20 split for each user. This means that for every user, 20
percent of the ratings are there in the testing dataset and 80 percent
are in the training dataset. We first start with the training dataset
and extract the necessary fields of information. We are creating a
utility matrix that will have ratings for each movie per user. This
matrix will help us to apply different techniques on the data. We have
used similarity matrix using centered-cosine similarity (also known as
Pearson correlation coefficient). Using this similarity allows us to
make sure that any 2 users are correctly identified to be dissimilar.
The user-user value will be positive if those two users are similar and
negative value if they are dissimilar. We have explicitly set the value
for same user-user in the matrix to be -1.

# Problem Formalization

The goal of this project is to find out which technique is the best
suited to generate predicted ratings for movies and accordingly
recommend top-N movies for a particular user. For the purposes of this
project, we have chosen the value of N as ten. We shall evaluate our
prediction and recommendation models by calculating metrics such as RMSE
and MAE for ratings model and Precision, Recall, F-Measure and NDCG for
the recommender model. The models explored are:

## User-Based Collaborative Filtering

This model uses techniques to calculate similarity between users. The
most commonly used measure is the cosine similarity. We have made a
small modification and used centered cosine instead.  
Subtracting mean from every term : \([x[i] = x[i] - \bar{x}]\)  
Cosine similarity :
\(cos(\pmb x, \pmb y) = \frac {\pmb x \cdot \pmb y}{||\pmb x|| \cdot ||\pmb y||}\)  
Pearson Correlation Coefficient:
\(r = \frac{{}\sum_{i=1}^{n} (x_i - \overline{x})(y_i - \overline{y})}
{\sqrt{\sum_{i=1}^{n} (x_i - \overline{x})^2(y_i - \overline{y})^2}}\)  

## SVD

Singular value decomposition (SVD) is a matrix factorization method that
generalizes the eigen decomposition of a square matrix \((n x n)\) to
any matrix \((n x m)\). Formula of SVD is:

\(M=U \Sigma V^T\)

where,  
M - is original matrix we want to decompose  
U - is left singular matrix (columns are left singular vectors). U
columns contain eigenvectors of matrix \(MM^T\)  
\(\Sigma\) - is a diagonal matrix containing singular (eigen)values  
V - is right singular matrix (columns are right singular vectors). V
columns contain eigenvectors of matrix \(M^TM\)  

## K-Nearest Neighbors

This model finds similar users or the ’neighbors’ of a particular users
by calculating a distance metric between two data points, in our case,
users. The most commonly used distance formulae are:  
Manhattan Distance \(\left(\sum_{i=1}^n |x_i-y_i|^p\right)^{1/p}\)  
  
Euclidean Distance \(\sqrt{\sum_{i=1}^n (x_i-y_i)^2}\)  

# The Proposed Model

As a baseline, we have implemented the User-Based Collaborative
filtering model. This technique finds similar users and then using their
highly rated movies, recommends a movie to the other similar users. It
can be implemented by first pre-processing the data to remove users with
a smaller number of reviews, padding missing values with local means,
and considering a time-frame where most number of reviews are
concentrated. A similarity matrix of users is generated using the
centered-cosine similarity. The ratings for movies for a particular user
are calculated taking the weighted average of the ratings given by the
top-K similar users to those movies. However, this technique yielded a
very high error and a very low precision for recommendation. Moreover,
the training and testing time for this system was very large as it took
hours to give the results.  
To find a more accurate and efficient technique, we used algorithms
implemented by the ‘Surprise’ package in Python. The K-Nearest Neighbor
algorithm finds users or items that closely resemble the relevant
entity. We have used KNN with Means model which is a collaborative
filtering method that considers the mean ratings of each user. To
improve its performance, we have used Pearson coefficient for the
calculating similarity and made it item based instead of user based.
After trying multiple combinations, we got the best results when
learning rate is 0.005 and regularization factor is 0.002 with 20
epochs. This provided us with a much-reduced RMSE and a far better
precision for recommendation.  
The best performance and accuracy were obtained by using the SVD
algorithm which is an in-built recommender tool. SVD performs Principal
Component Analysis to form matrices for users (U) and movies(M) by
performing dimensionality reduction. These generated matrices are used
as factors of the ratings matrix that we will generate to provide us
with the predicted ratings. (\(R = M. S. U^T\)) We have also used SVD++
which includes the "implicit" information while training and hence, will
give better results in some cases. This model gives better results for
new users for which there are not enough ratings to correctly recommend
the movies.  

# Experiments

As seen in the figure 2, KNN, SVD and SVD++ significantly outperforms
the baseline model, that is, User-Based CF. Within these, SVD and SVD++
are extremely efficient models that also has small training and testing
time. Interesting note: There was a Netflix competition for the best
recommender model. The prize winner was a highly optimized model of
SVD++.

![Metric comparisons among different
algorithms](MDM%20Project/Evaluations.jpg)

# Conclusions and Future Work

In this paper, we have used User based Collaborative Filtering which
doesn’t prove to be quite efficient in predicting rating for the movies
given this is a huge dataset. Time taken for training and testing by the
original CF system is also significantly higher. Different Matrix
factorization and Nearest Neighbor techniques are proposed which
performs better than the baseline model. KNN and SVD are proposed which
are based on Item based CF and Centered-cosine Similarity techniques. We
have compared the different algorithms based on the prediction ratings
using the metrics MAE and RMSE. Moreover, we are calculating the top-10
recommendations for each user in the testing set and evaluating the
performance using Precision, Recall, F-measure and NDCG.

In future, we can include the other attributes from the dataset to make
better predictions like recommendations based on Genre or time frame. We
can also add Natural Language Processing techniques like sentiment
analysis to understand the reviews given by the user, based on that we
can predict a rating and compare it with the rating given by the user.
Based on the trends seen in the dataset, most people are inclined
towards giving a higher rating which may not be true for every
individual. Using NLP, we can try to find out if a particular movie
can/can not be recommended to a person based on the content rather than
just ratings. We can also aim to implement Deep Learning models like
Convolutional Neural Network which will classify the movie into fixed
genre based classes using the posters(images) and trailers(videos) of
those movies.
