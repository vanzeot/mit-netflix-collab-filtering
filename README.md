# MIT - Netflix Collaborative Filtering (Machine Learning)
A solution to a collaborative filtering problem from MIT's Machine Learning course, using Netflix's dataset.

Problematic:

"Your task is to build a mixture model for collaborative filtering. You are given a data matrix containing movie ratings made by users where the matrix is extracted from a much larger Netflix database. Any particular user has rated only a small fraction of the movies so the data matrix is only partially filled. The goal is to predict all the remaining entries of the matrix.

You will use mixtures of Gaussians to solve this problem. The model assumes that each user's rating profile is a sample from a mixture model. In other words, we have  possible types of users and, in the context of each user, we must sample a user type and then the rating profile from the Gaussian distribution associated with the type. We will use the Expectation Maximization (EM) algorithm to estimate such a mixture from a partially observed rating matrix. The EM algorithm proceeds by iteratively assigning (softly) users to types (E-step) and subsequently re-estimating the Gaussians associated with each type (M-step). Once we have the mixture, we can use it to predict values for all the missing entries in the data matrix."

<p align="center">
  <img src="https://github.com/user-attachments/assets/e8673f2e-ef33-4b2b-9fe6-ab4957cc6466" alt="image description" width="800"/>
</p>

