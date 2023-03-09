from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import joblib

app = Flask(__name__)

# Route for home page
@app.route("/")
def index():
    return render_template("index.html")

# Route for the NLP model to predict the number of stars associated with a text review
# The function loads the model and uses the review inserted into the form and gives a
# prediction and displays it back onto the homepage.

# the variable 'model' is the model being loaded, and it predicts number of stars associated with the review
# the variable 'review' is the review the user inputs into the form
# the variable 'prediction' is the predicted number of stars from an inputted review
@app.route("/predict", methods=['POST'])
def predict():
    model = joblib.load('yelp_review_model.pkl')
    review = list(request.form.values())
    print(review)
    prediction = model.predict(review)

    return render_template("index.html", prediction_text=f"{review[0]}: {prediction[0]} stars")


# Route for the business page where a user will input the name of a business in order to
# find the predicted average star rating for the business (based off the business's text reviews)
# as well as to find the keywords associated with a business's rating.
@app.route("/business")
def open_business():
    return render_template('business.html')

# This route renders all of the available locations for the user inputted location.
# For example there may be many locations for a Walmart and the user only wants to know the reviews and
# keywords for only one location.

# This function first reads in the Yelp business dataset and filters it based on the name of the business
# the user entered.
# Then it displays all of the available locations to the user.

# The variable 'df' is the dataframe that represents the Yelp business dataset
# The variable 'business' what the user inputted into the form on the business page
# the variable 'name' is the name of the business the user inputted
# the variable 'business_df' is a dataframe that represents the business dataset in which
#   all of the entries are the same business as the user inputted, just different locations of the same
#   business
# the variable 'locations' is a dataframe which contains the information that will be displayed to the user so
#   that they may select a specific location for the business
#   information includes: the name, the address, and the location's business id.

@app.route("/business_search", methods=['POST'])
def search_businesses():
    df = pd.read_json('../DATA/archive (1)/yelp_academic_dataset_business.json', lines=True)
    df['name'] = df['name'].apply(lambda x: x.lower())   # makes it so the inputted business is not case-sensitive
    business = list(request.form.values())
    name = business[0].lower()
    business_df = df[df['name'] == name]
    locations = business_df[['name', 'address', 'business_id']]
    # return render_template('business.html')
    return render_template('business.html', column_names=locations.columns.values, row_data=list(
        locations.values.tolist()), link_column='business_id', zip=zip)


# A function that gets the reviews of a specific business location (via its business id)
# Returns a dataframe where all the entries are the reviews for the specific business location

# The parameters are 'df' and 'id'.
#   'df' is the dataframe of the Yelp reviews dataset
#   'id' is the business id that we want to filter for
# The variable 'br_df' stands for 'business review dataframe' where it is a dataframe that has the reviews for a
# particular business location
def get_reviews(df, id):
    br_df = df[df['business_id'] == id]
    return br_df

# A function that adds a column to a dataframe. This column is the predicted star rating for a text review.
# This function uses the NLP model to predict the star rating for a text review.

# The parameter is 'br_df'.
#   'br_df' is the business review dataframe which is the relevant dataframe that has text reviews, star ratings and
#   other useful information for the desired business.
# The variable 'stars_model' is the model
# br_df['pred_stars'] is the new column
def get_pred_stars(br_df):
    stars_model = joblib.load('yelp_review_model.pkl')
    br_df['pred_stars'] = br_df['text'].apply(lambda x: stars_model.predict([x]))
    # avg_pred_stars = br_df['pred_stars'].mean()
    return br_df


# This functions gets the average star rating for the business location
# The parameter 'pred_stars_df' is the dataframe that has the review information for the desired business including
# the predicted star rating for each text review.
def get_avg_pred_stars(pred_stars_df):
    return round(pred_stars_df['pred_stars'].mean()[0])


# This function gets the bag of words associated with the predicted average star rating.
# If the predicted average for a business location was 4 stars, then it would get the top 30 words (if there are 30
# words) that contributed to that rating.

# It has the parameters 'stars' and 'pred_stars_df' where 'stars' is the average star rating that was predicted using the
# model and 'pred_stars_df' is the dataframe that is used as the Corpus for count vectorization

# The variable 'c_vec' is the Count Vectorizer
# The variable 'matrix_stars' is the resulting matrix from count vectorization of the corpus
# The variable 'freqs' is a two-tuple that has the word and its frequency
def get_star_BOW(stars, pred_stars_df):
    c_vec = CountVectorizer(stop_words='english')
    c_vec.fit(pred_stars_df['text'])
    matrix_stars = c_vec.transform(pred_stars_df[pred_stars_df['pred_stars'] == stars]['text'])
    freqs = zip(c_vec.get_feature_names_out(), matrix_stars.sum(axis=0).tolist()[0])
    stars_top_30 = sorted(freqs, key=lambda x: -x[1])[:30]
    return stars_top_30


# Routes to the resulting page after the user selects a specific business that are curious about.
# The page will display the words associated with the business's predicted average rating as well as whether
# the business received mainly positive or negative reviews.

# A post request is received in which the user selected a specific business. (by selecting a business id)
# The business id is then used to filter out the Yelp business dataset and the Yelp reviews dataset so that
# only the relevant data remains (that being the reviews for the user selected business)

# The variable 'business_id' is the business id the user selected on the business page
# The variable 'df' is the dataframe that represents the Yelp reviews dataset
# The variable 'br_df' is the dataframe that is the result of calling the get_reviews() function
# The variable 'pred_stars_df' is the dataframe that is the result of calling the get_pred_stars() function
# The variable 'avg_pred_stars' is the average star rating from the get_avg_pred_stars() function
# The variable 'top_30' is the list of the top 30 words associated with the rating of the business which is the result
#   of calling the get_star_BOW() function

@app.route("/business_reviews", methods=['POST'])
def get_business_reviews():
    business_id = request.form['business_id']
    df = pd.read_csv('../DATA/archive (1)/yelp_review.csv')
    br_df = get_reviews(df, business_id)
    pred_stars_df = get_pred_stars(br_df)
    avg_pred_stars = get_avg_pred_stars(pred_stars_df)
    top_30 = get_star_BOW(avg_pred_stars, pred_stars_df)
    sentiment = "POS"
    if avg_pred_stars >= 4:
        sentiment = "POS"
    else:
        sentiment = "NEG"
    return render_template('business_reviews.html', top_30=top_30, avg_review=avg_pred_stars, sentiment = sentiment)


# Routes to the sentiment page where a review is inputted and a predicted sentiment of that review is returned back
# onto the page.
@app.route("/sentiment")
def open_sentiment():
    return render_template('sentiment.html')


# This route handles receiving the post request and return the prediction of the text review sentiment

# The variable 'model' is the NLP model that predicts the sentiment of the text review (either pos or neg)
# The variable 'review' is the text review that was inputted by the user
# The variable 'prediction' is the predicted sentiment
@app.route("/predictSent", methods=['POST'])
def predict_sent():
    model = joblib.load('yelp_review_sentiment_model.pkl')
    review = list(request.form.values())
    prediction = model.predict(review)

    return render_template("sentiment.html", sentiment_text=f"{review[0]}: Sentiment: {prediction[0]}")


