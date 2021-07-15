# Sentiment Tracker

**BUSINESS CASE**

Sentiment analysis is critical for understanding how customers, investors, and the general public feel about a companies brand. Companies are operating in an environment where anything less than a pristine image is detrimental. I have built a dashboard that shows companies how their public sentiment is changing on an hourly basis. They can use this data to make informed decisions on how to alter their public persona. 

**ROADMAP**

The goal of this project is to track the sentiment of major technology companies based on global news articles. I used the newsapi (link) to source articles from around the world and filtered out those that did not provide meaningful content. Next, I used a Kaggle financial news headlines (link) dataset to determine the best combination of unsupervised sentiment analysis models and landed on a combination of VADER and TextBlob. Then, I used this rules based model to approach to extract sentiment from the news headlines that I sourced and confidentially assigned each article a positive, neutral, or negative score. Finally, I used a Random Forest model to determine which specific words were most impactful in driving sentiment by extracting feature importance. Last, I built a dashboard on streamlit to display my findings. 

**GOALS**

1. Import news articles with proper queries and filtering
2. Implement various unsupervised sentiment analysis models (VADER, TextBlob, FLAIR)
3. Utilize classification models like Random Forests to create inferences
4. Deploy to a webapp using streamlit

---------
# Extract, Transform, Load

ETL was performed using Python Requests package and the newsapi Python client. I decided to focus on FAANG companies + Microsoft. Newsapi has strict throttling limits and only allowed for 100 requests over a 24 hour period which prevented me from being able to stream live data. Additionally, results only span one month so I decided to focus on 2 weeks of data for 6 companies. 

Querried results by relevancy and manually sorted out 'bad sources.' Removed duplicates based on content and source and dropped null values. 'Content' of the article is limited to 200 characters so concatenated 'title', 'description' (summary of article) and 'content' so that unsupervised models would have more data to extract sentiment. 

-------
# Opinion Mining

Used VADER and TextBlob to extract sentiment:

> VADER: Lexicon and rule based model for sentiment analysis typically used to extract social media sentiment. Trained on social media content and retuns a score between -1 and 1 based on positivity vs. negativity

> TextBlob: Rules based sentiment analysis model which uses NaiveBayesAnalyzer and is trained on a movie reviews corpus

Performed sentiment analysis on news articles using VADER and TextBlob. When both models had the same prediction, they were 67% accurate (predicting positive, negative, neutral sentiment) and 42% accurate (TextBlob) and 47% (VADER) accurate when they differed.

When VADER & TextBlob had the same result, I agreed with consensus and took the conclusion as accurate. When they differed, I created a rules based approach to reconcile. When their results differed, I looked at how much they differed by (0,2) and chose the VADER or TextBlob result based on how far off from each other they were. When they were very different, I opted for VADER. This improved accuracy from 56% to 60%.


https://www.researchgate.net/publication/275828927_VADER_A_Parsimonious_Rule-based_Model_for_Sentiment_Analysis_of_Social_Media_Text

https://textblob.readthedocs.io/en/dev/advanced_usage.html#sentiment-analyzers

------
# NLP Using TF-IDF and Random Forest

Used sklearn TF-IDF vectorizer to pre-process text for classifcation models. Did not use CountVectorizer because TF-IDF is a more nuanced approach. Used a pipeline to alter the following:

> 1. Stop words: List of words eliminated because they have litte to no meaning for classification (ie prepositions)
> 2. Ngram Range: Number of n-grams to incorporate into TF-IDF Vectorizer (used unigrams and bigrams)
> 3. Max_df: Ignores terms that are present accross maximum threshold of documents
> 4. Min_df: Ignores terms that are present accross minimum threshold of documents
> 5. Tokenizer: Utilized NLTK RegExp tokenizer to handle contractions
> 6. Normalizer: Applied L1 or L2 normalization to reduce noise

Tested Random Forest and Logistic Regression to classify sentiment as positive, negative, or neutral (1, -1, 0). Dummy model had 40% accuracy and Random Forest Models had a test accuracy of 64%. Logistic regression performed slightly poorer at 62%. Negative recall was the most difficult type of sentiment to predict across the board. Positive recall was the most succesfull across the board. Pruned random forest model to reduce overfitting using: 

> 1. Max Depth: Maximum depth of tree
> 2. Minimum Sample Leaf: Minimum number of samples to to split an internal node


**FUTURE WORK**
1. Topic modeling using Latent Dirichlet Allocation to improve insights
2. Live stream news articles (would need to get around API request limits or use different API)
3. Extract opinions from social media sitess like Reddit, Twitter, & Facebook
4. Create API to add sentiment as an overlay for technical analysis for quantitative traders
5. Reduce overfit on Random Forest models