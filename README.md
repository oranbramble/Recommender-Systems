# Recommender Systems

With this project, I developed 2 different recommender systems. The aim of both is to predict a list of ratings for usersv on items they have not rated yet, based on their previous ratings for other items. This is a classic problem for streaming services such as Netflix and Amazon Video, as they want to always be able to recommend the most relevant content for each individual to maximise sales. To then recommend content based on the list of ratings, you would sort the list of ratings and take the top items for each user and recommend these. The two systems I implemented are:

- Cosine Similarity
- Matrix Factorisation

## Cosine Similarity

![cosine sim](https://github.com/oranbramble/Recommender-Systems/assets/56357864/3425bd18-2367-40c9-b428-5d4032d756c7)

The Cosine Similarity algorith works using the equation above to calculate predicted ratings for each user.
