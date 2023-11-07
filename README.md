# Recommender Systems

With this project, I developed 2 different recommender systems. The aim of both is to predict a list of ratings for usersv on items they have not rated yet, based on their previous ratings for other items. This is a classic problem for streaming services such as Netflix and Amazon Video, as they want to always be able to recommend the most relevant content for each individual to maximise sales. To then recommend content based on the list of ratings, you would sort the list of ratings and take the top items for each user and recommend these. The two systems I implemented are:

- Cosine Similarity
- Matrix Factorisation

Note both these systems were implemented relatively manually, meaning they do not use libraries to perform all the steps. The main library used is `numpy` for vector multiplication. Other than that, it relies on the equations and techniques below to calculated the predicted ratings.

</br>

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

This method relies on the decomposition and reformation of a ratings matrix. It starts with a matrix containing all known ratings for each user on all the items they have rated. It contains both the ratings and **holes**, where a user has not rated an item yet.

</br>

<p align="center">
  <img src="https://github.com/oranbramble/Recommender-Systems/assets/56357864/aab47861-04a3-4555-bc7e-1678e2faa6ac">
</p>

</br>

Following this, it splits this into the **latent factors**. In this case, the latent factors are the users and the items. Basically, it splits the ratings matrix into 2 smaller matrices which, when multiplied together form the original ratings matrix. To find the latent factors, a **stochastic gradient descent** algorithm is used. This slowly builds up the latent factor matrices by randomly initialising the latent factor matrices, and the adjusting them using individual gradient descent steps. 

</br>

<p align="center">
  <img src="https://github.com/oranbramble/Recommender-Systems/assets/56357864/ab7b2f47-c92c-4ace-9f09-d0e93713a4d2">
</p>

</br>

These latent factors can then be multiplied back together to provide predicted ratings for items that a user has not rated. This works as we initially factorised the ratings matrix, which contained holes (zeros) where a user had not rated an item, into the latents, and then when they are mutliplied back together, it creates a new rating matrix with the holes filled with predicted ratings. These predicted ratings are then taken and saved into a csv file. 

</br>

## How to run

To be able to run either of the systems above, first the repository must be cloned using the Command Line interface line below:

```
git clone https://github.com/oranbramble/Recommender-Systems.git
```

These programs were written fully in Python, so if you do not have this installed, please install the latest version from [here](https://www.python.org/downloads/). It also requires teh `numpy` library to be installed, which can be done in the command line using the following command (after installing Python):

```
python -m pip install numpy
```

</br>

### Cosine Similarity Program

To run this program, navigate to the `Cosine Similarity` directory, and then execute the following command:

```
python main.py
```

This will output the following result on the command line, showing the Mean Average Error (MAE) and Root Mean Squared Error (RMSE), which captures the accuracy of the system. 

</br>

<p align="center">
  <img src="https://github.com/oranbramble/Recommender-Systems/assets/56357864/8eb7c06e-cced-4fc5-b9d2-74a463b44001">
</p>

</br>



