# 1. Data Collection

### Data Source
The source of the data is `TMDB (The Movie Database)`, available for download on `kaggle`.

It consists of the following `csv` files:
1. tmdb_5000_credits.csv
2. tmdb_5000_movies.csv

**tmdb_5000_credits.csv**
This file contains the following columns:
- `movie_id`
- `title`
- `cast`
- `crew`

**tmdb_5000_movies.csv**
This file contains the following columns:
- `budget`
- `genres`
- `homepage`
- `id`
- `keywords`
- `original_language`
- `original_title`
- `overview`
- `popularity`
- `production_companies`
- `production_countries`
- `release_date`
- `revenue`
- `runtime`
- `spoken_languages`
- `status`
- `tagline`
- `title`
- `vote_average`
- `vote_count`



# 2. Data Preprocessing

Data preprocessing is necessary to provide clean and useful data for computer-dirven decision-making.

It includes, but is not limited to, the following operations:
1. **Feature Selection:** Selecting the most influential columns for decision-making.
2. **Handling Missing and Duplicate Data:** Removing rows or columns with null values or imputing them with appropriate values. Duplicate rows and columns should also be removed if present.
3. **Feature Extraction:** Deriving new, meaningful features from existing columns.

**For this project, I prepared the data as follows:**
1. Merged both the DataFrames, movies and credits into a single one based on movie id which was present in both of them.
2. Selected only the appropriate columns from the merged DataFrame. Following is the list of selected columns:
	- `genres`
	- `keywords`
	- `overview`
	- `cast`
	- `crew`
	- `movie_id` (for fetching movie poster via TMDB API and showing on the frontend)
	- `title`
3. Dropped rows having null values.
4. Extracted only the genre names from the `genres` column which consisted other data as well which was redundant for our purpose. Such as genre id. Also removed the space between genre name such as "Science Fiction" became "ScienceFiction".
5. Same was done with the `keywords` column as well.
6. Extracted first 3 cast names from the `cast` column.
7. Extracted director name from the `crew` column.
8. Split words based on space character from `overview` column in order to merge it with `genres`, `keywords`, `cast` and `crew` columns.
9. Named the merged column `tags`.
10. Joined the words in `tags` column with space character and made them lowercase.
11. Selected only the `movie_id`, `title` and `tags` columns for the final DataFrame.
12. Applied tokenization using regex pattern in order to extract words/tokens from the `tags` column. The pattern ignores the punctuation as they are not words.
13. Ignored stop words and Applied stemming to the tokens.
14. Applied `CountVectorizer` (BoW) to convert each list of tokens into a vector for each movie.


# 3. Similarity Score Calculation

The final goal/task is to calculate the similarity score between each pair of vectors. A higher similarity score between two vectors indicates that the corresponding movies are highly similar.
We used `cosine_similarity` function from `scikit-learn` module, passing the array of vectors as input. The function returned a similarity matrix containing pairwise similarity scores between all vector pairs.
Each row of the similarity matrix contains similarity scores between the movie in that row and all other movies in the DataFrame.


# 4. Movie Recommendation

We created a function that takes a movie name as input and returns the top 5 most similar movies.
The function works as follows:
1. Accepts a movie name as input.
2. Finds the index of the DataFrame row where the 'title' column matches the input movie name and stores it in movie_index.
3. Selects the row present at movie_index in the similarity matrix.
4. Pairs each similarity score in the selected row with its corresponding movie index.
5. Sorts the row based on similarity score in descending order and selects top 5 similarity scores along with movie indices.
6. Uses these movie indices to retrieve the movie names from the DataFrame and prints them.