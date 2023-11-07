# Recommender Systems

With this project, I developed 2 different recommender systems. The aim of both is to predict a list of ratings for usersv on items they have not rated yet, based on their previous ratings for other items. This is a classic problem for streaming services such as Netflix and Amazon Video, as they want to always be able to recommend the most relevant content for each individual to maximise sales. To then recommend content based on the list of ratings, you would sort the list of ratings and take the top items for each user and recommend these. The two systems I implemented are:

- Cosine Similarity
- Matrix Factorisation

## Cosine Similarity

</br>

The cosine similarity algorithm relies on two main equations. The first calculates the similarity between each and every item. This forms a **similarity matrix** containing similarities between each and every item. 

![cosine sim](https://github.com/oranbramble/Recommender-Systems/assets/56357864/3425bd18-2367-40c9-b428-5d4032d756c7)

Following this. the similarities are used to then calculate the predicted ratings for each user. A predicted rating is a rating we predict a user will give for a certain item. In the example of Netflix, this would be predicting what a user will probably rate for a new tv show they have not watched yet.

![prediction eq](https://github.com/oranbramble/Recommender-Systems/assets/56357864/3e6d92cb-361d-4979-aa1d-e535339a6ea6)
