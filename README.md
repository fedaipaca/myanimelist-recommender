# Anime Recommendation System with Apache Spark

A collaborative filtering-based anime recommendation system built using Apache Spark's ALS (Alternating Least Squares) algorithm. This project analyzes anime ratings from MyAnimeList dataset to provide personalized recommendations and discovers similar content through matrix factorization techniques.

## Team Members

- **211805076** - Mehmet ÖZCAN
- **211805036** - Ahmet Muhammed AYDIN  
- **211805008** - Fedai PAÇA

## Table of Contents

- [Installation](#installation)
- [Dataset Information](#dataset-information)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Data Preprocessing](#data-preprocessing)
- [Model Training](#model-training)
- [Results and Analysis](#results-and-analysis)
- [Recommendations](#recommendations)
- [Conclusions](#conclusions)
- [Usage](#usage)

## Installation

To set up the environment and run the project, follow these steps:

### Prerequisites

1. **Java 21**
   Install Java Development Kit (JDK) 21 from [here](https://www.oracle.com/tr/java/technologies/downloads/#java21).
    ```bash
    # Check Java version
    java -version
    ```


2. **Python 3.12.10+**
   ```bash
   # Check Python version
   python --version
   ```

   Project contains a `.python-version` file to ensure compatibility with Python 3.12.10. You can use tools like `pyenv` to manage Python versions easily.

### Required Python Packages

You will find the `requirements.txt` file in the project root directory. Install the required packages using pip:

```bash
pip install -r requirements.txt
```

### Main packages and Libraries are used in this project:
- `pyspark`: For distributed data processing and machine learning
- `pandas`: For data manipulation and analysis
- `numpy`: For numerical operations
- `matplotlib`: For data visualization
- `seaborn`: For statistical data visualization
- `jupyter`: For interactive notebooks
- `findspark`: For Spark session management in Jupyter notebooks
- `ipykernel`: For Jupyter kernel management
- `pyarrow`: For saving and loading Parquet files


## Dataset Information

The project uses three main datasets from MyAnimeList.
Original dataset can be found in [Kaggle MyAnimeList Dataset.](https://www.kaggle.com/datasets/azathoth42/myanimelist)

Original dataset are in csv format, but we converted them to Parquet format for better performance and efficiency in Spark.

About Parquet format:
Parquet is a columnar storage file format optimized for use with big data processing frameworks like Apache Spark. It provides efficient data compression and encoding schemes, which significantly reduce the amount of disk space required and improve query performance. Parquet files are designed to work well with complex nested data structures, making them ideal for analytical workloads.

Original dataset is around 3gb. But after saving them in Parquet format, the size is reduced to around 800mb.

### 1. Users Dataset (`users.parquet`)
**Size**: 108711 users

**Columns**:
- **username**: Display name of the user (nullable = true)
- **user_id**: Unique identifier for the user (nullable = true)
- **user_watching**: Number of anime the user is currently watching (nullable = true)
- **user_completed**: Number of anime the user has completed (nullable = true)
- **user_onhold**: Number of anime the user has put on hold (nullable = true)
- **user_dropped**: Number of anime the user has dropped (nullable = true)
- **user_plantowatch**: Number of anime the user plans to watch (nullable = true)
- **user_days_spent_watching**: Total days the user has spent watching anime (nullable = true)
- **gender**: Gender of the user (nullable = true)
- **location**: Location of the user (nullable = true)
- **birth_date**: Birth date of the user (nullable = true)
- **access_rank**: User's access rank on the platform (nullable = true)
- **join_date**: Date the user joined the platform (nullable = true)
- **last_online**: Date the user was last online (nullable = true)
- **stats_mean_score**: Mean score of all ratings given by the user (nullable = true)
- **stats_rewatched**: Number of anime the user has rewatched (nullable = true)
- **stats_episodes**: Total number of episodes watched by the user (nullable = true)

### 2. Anime Dataset (`animes.parquet`)
**Size**: 6668 anime entries

**Columns**:
- **anime_id**: Unique identifier for each anime (nullable = true)
- **title**: Main title of the anime (nullable = true)
- **title_english**: English title of the anime (nullable = true)
- **title_japanese**: Japanese title of the anime (nullable = true)
- **title_synonyms**: Alternative titles or synonyms for the anime (nullable = true)
- **image_url**: URL of the anime's cover image (nullable = true)
- **type**: Type of the anime (e.g., TV, Movie, OVA) (nullable = true)
- **source**: Source material for the anime (e.g., Manga, Light Novel) (nullable = true)
- **episodes**: Total number of episodes (nullable = true)
- **status**: Current status of the anime (e.g., Finished Airing, Currently Airing) (nullable = true)
- **airing**: Boolean indicating if the anime is currently airing (nullable = true)
- **aired_string**: Human-readable airing period (nullable = true)
- **aired**: Detailed airing period in structured format (nullable = true)
- **duration**: Duration of each episode (nullable = true)
- **rating**: Content rating of the anime (e.g., PG-13, R) (nullable = true)
- **score**: Average user score for the anime (nullable = true)
- **scored_by**: Number of users who scored the anime (nullable = true)
- **rank**: Rank of the anime based on score (nullable = true)
- **popularity**: Popularity rank of the anime (nullable = true)
- **members**: Number of members who added the anime to their list (nullable = true)
- **favorites**: Number of users who marked the anime as a favorite (nullable = true)
- **background**: Background information or trivia about the anime (nullable = true)
- **premiered**: Season and year the anime premiered (nullable = true)
- **broadcast**: Scheduled broadcast time (nullable = true)
- **related**: Information about related anime (e.g., sequels, spin-offs) (nullable = true)
- **producer**: Production company or entity responsible for the anime (nullable = true)
- **licensor**: Licensing company for the anime (nullable = true)
- **studio**: Studio responsible for the anime's animation (nullable = true)
- **genre**: Genres associated with the anime (nullable = true)
- **opening_theme**: Opening theme song(s) of the anime (nullable = true)
- **ending_theme**: Ending theme song(s) of the anime (nullable = true)
- **duration_min**: Duration of each episode in minutes (nullable = true)
- **aired_from_year**: Year the anime started airing (nullable = true)

### 3. User-Anime Ratings Dataset (`users_animes.parquet`)
**Size**: 31284030 ratings

**Columns**:
- **username**: Display name of the user (nullable = true)
- **anime_id**: Unique identifier for the anime (nullable = true)
- **my_score**: Rating given by the user to the anime (nullable = true)

## Project Structure

```
anime-recommender-spark-main/
├── notebooks/
│   └── 211805076_211805036_211805008.ipynb
├── data/
│   ├── users.parquet
│   ├── animes.parquet
│   ├── users_animes.parquet
│   └── test_data_single.parquet (generated)
├── trained_models/
│   └── best_model/ (generated)
├── results_df.json (generated)
└── README.md
```

notebooks/ contains the Jupyter notebook with the complete project code and analysis.
data/ contains the Parquet files for users, anime, and user-anime ratings.
trained_models/ contains the saved best performing ALS model.
results_df.json contains the final results of the model predictions and recommendations.

## Requirements for this project
- Present/show your computer info (with code) IP address, name and configuration of your PC etc.
- Use spark dataframes.
- Make some EDA (Exploratory Data Analysis) on your dataset.
- Use at least one map() and reduce() (derivatives i.e. reduceByKey) functions.
- Use 70% of dataset for training, 30% for testing.
- Use ALS (Alternating Least Squares) for training recommendation model with last 4 digit of your student number as a “seed” value. 
- Also change the parameters of ALS re-run the algorithm for parameters “rank” (10, 50, 200), “iteration” (10, 50, 200) and “lambda” (0.01, 0.1). This means 18 different model will be created using specified rank-iteration-lambda values.
- Run ALS with different parameters and visualize the comparative performance result of different parameters with a plot in the program you wrote (i.e. pyspark - Jupiter notebook).
- Find and present MSE (Mean Squared Error), RMSE (Root Mean Squared Error) for performance evaluation of each model and explain them, indicate best model for your dataset, explain why. Plot a graph of all RMSE values according to changing iteration or rank values.
- Make prediction with ALS and compare it with the original values (with real values) side by side.
- Cosine similarity between ALS model and the products should be found then, 10 users who will like product X the most should be determined.

## Methodology

Our recommendation system follows a systematic approach using collaborative filtering:

### 1. **Collaborative Filtering with ALS**
- Uses matrix factorization to decompose the user-item rating matrix
- Learns latent (hidden, generated) features for both users and anime

**What is Collaborative Filtering?**
Collaborative filtering is a technique used in recommendation systems to predict a user's interests by collecting preferences from many users. It assumes that if two users agree on one issue, they are likely to agree on others as well. Collaborative filtering can be user-based (finding similar users) or item-based (finding similar items).
A basic example is Netflix recommending movies based on what similar users have watched and rated highly.

**What is ALS?**
In a user-movie ratings table, some users may have rated some movies, while others may not have rated those movies at all. ALS fills in these missing ratings by learning patterns from the existing ratings. It does this by breaking down the large user-movie matrix into two smaller matrices: one representing users and another representing movies. Each user and movie is represented by a set of latent features (hidden characteristics).


### 2. **Cosine Similarity Analysis**
- Calculates similarity between anime based on learned features
- Enables content-based recommendations

### 3. **Paramaters We Used**
- **Rank**: Number of latent factors (10, 50, 200)
What is Rank?
Rank in the context of ALS (Alternating Least Squares) refers to the number of latent factors or dimensions used to represent users and items (anime in this case). A higher rank allows the model to capture more complex relationships between users and items, but it also increases the risk of overfitting if the dataset is sparse. In our project, we tested ranks of 10, 50, and 200 to find the optimal balance between model complexity and performance.

- **Iterations**: Number of training iterations (10, 50, 200)
What is Iteration?
Iteration in the context of ALS (Alternating Least Squares) refers to the number of times the algorithm updates the user and item latent factor matrices during training. Each iteration refines the model's predictions by minimizing the error between predicted and actual ratings. More iterations can lead to better convergence and improved model performance, but they also increase computational time. In our project, we tested iterations of 10, 50, and 200 to evaluate their impact on model accuracy.

- **Regularization (Lambda)**: Controls overfitting (0.01, 0.1)
What is Regularization (or Lambda)?
Regularization in the context of ALS (Alternating Least Squares) is a technique used to prevent overfitting by adding a penalty term to the loss function. It discourages the model from fitting too closely to the training data, which can lead to poor generalization on unseen data. The regularization parameter (lambda) controls the strength of this penalty. A higher lambda value increases the penalty, while a lower value allows more flexibility in fitting the data. In our project, we tested regularization values of 0.01 and 0.1 to find the optimal balance between model complexity and performance.

## Data Preprocessing

### Cleaning Steps

1. **Anime Filtering**:
   - Removed anime with ≤1 episode (movies excluded for simplicity)
   - Filtered anime with <5,000 members (popularity threshold)
   - Removed anime with score <6.0 (quality threshold)
   - Filtered anime with <1,000 ratings (reliability threshold)

2. **User Filtering**:
   - Removed users with <5 completed anime (experience threshold)
   - Ensured users exist in the ratings dataset

3. **Rating Filtering**:
   - Removed ratings <5 (focusing on positive recommendations)
   - Filtered null values and inconsistent data

### Training and Testing Split
- **70% Training Set**: Used for model training
Training data size: 9749653 ratings 
- **30% Testing Set**: Used for model evaluation
Testing data size: 4179419 ratings



## Models Training with different parameters

### Training Process


### Saving Best Model and Results


## Results and Analysis

### Best Model Performance

**Optimal Parameters**:
- **Rank**: 200 (latent factors)
- **Iterations**: 200
- **Regularization**: 0.10
- **RMSE**: 1.0153
- **MSE**: 1.0309
- **Training Time**: ~15 minutes

### Key Findings

1. **Parameter Impact**:
   - Higher rank (200) consistently outperformed lower values
   - More iterations (200) improved performance, especially with higher rank
   - Stronger regularization (λ=0.1) prevented overfitting better than λ=0.01

2. **Performance Metrics**:
   - Average prediction error: ~1.02 rating points on 5-10 scale
   - Model explains significant variance in user preferences
   - Prediction distribution closely matches actual rating distribution

3. **Error Analysis**:
   - Mean prediction error: ~0.003 (nearly unbiased)
   - Standard deviation: ~1.01
   - Error distribution approximately normal (good sign)

### Comparing Predictions with Actual Ratings

**Prediction Quality Examples**:
```
User: MishaMisha
Anime: "Attack on Titan" | Actual: 9.0 | Predicted: 8.7
Anime: "Death Note" | Actual: 8.0 | Predicted: 8.2
Anime: "One Piece" | Actual: 7.0 | Predicted: 7.1
```

## Recommendations

### Sample User Recommendations

For user "MishaMisha" (active user with 150+ ratings):

**Top 5 Recommended Anime**:
1. Steins;Gate (Predicted: 9.2)
2. Fullmetal Alchemist: Brotherhood (Predicted: 9.1) 
3. Hunter x Hunter (2011) (Predicted: 8.9)
4. Code Geass (Predicted: 8.8)
5. Monster (Predicted: 8.7)

### Similar Animes to "One Punch Man"


### Users Who Will Like "One Punch Man"


## Conclusions


### Limitations and Future Work
