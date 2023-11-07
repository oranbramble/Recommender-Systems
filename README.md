# Recommender Systems

With this project, I developed 2 different recommender systems. The aim of both is to predict a list of ratings for usersv on items they have not rated yet, based on their previous ratings for other items. This is a classic problem for streaming services such as Netflix and Amazon Video, as they want to always be able to recommend the most relevant content for each individual to maximise sales. To then recommend content based on the list of ratings, you would sort the list of ratings and take the top items for each user and recommend these. The two systems I implemented are:

- Cosine Similarity
- Matrix Factorisation

Note both these systems were implemented relatively manually, meaning they do not use libraries to perform all the steps. The main library used is `numpy` for vector multiplication. Other than that, it relies on the equations and techniques below to calculated the predicted ratings.

## Cosine Similarity

</br>

The cosine similarity algorithm relies on two main equations. The first calculates the similarity between each and every item. This forms a **similarity matrix** containing similarities between each and every item. 

</br>

<p align="center">
  <img src="https://github.com/oranbramble/Recommender-Systems/assets/56357864/3425bd18-2367-40c9-b428-5d4032d756c7">
</p>

</br>

Following this. the similarities are used to then calculate the predicted ratings for each user. A predicted rating is a rating we predict a user will give for a certain item. In the example of Netflix, this would be predicting what a user will probably rate for a new tv show they have not watched yet.

</br>

<p align="center">
  <img src="https://github.com/oranbramble/Recommender-Systems/assets/56357864/3e6d92cb-361d-4979-aa1d-e535339a6ea6">
</p>

</br>

The **neighbourhood** above relates to the items we want to compare to in order to calculate the predicted rating. For exmaple, might have a neighbourhood of 5 most similar items. This equation leaves us with a list of predicted ratings for users and items. This is what this program outputs to a file, the list of predicted ratings. 

</br> 

## Matrix Factorisation

This is a more complicated method. It 
