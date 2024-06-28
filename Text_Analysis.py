import requests
import pandas as pd
from bs4 import BeautifulSoup
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


POSITIVE_WORDS_FILE = "positive-words.txt"
NEGATIVE_WORDS_FILE = "negative-words.txt"


# Ensure NLTK data is downloaded
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('cmudict')
from nltk.corpus import cmudict
import re

# Initialize CMU Pronouncing Dictionary
d = cmudict.dict()

def count_syllables(word):
    """
    Count the number of syllables in a word.
    """
    if word.lower() in d:
        return [len(list(y for y in x if y[-1].isdigit())) for x in d[word.lower()]][0]
    else:
        return sum([1 for char in word if char in 'aeiou'])

def extract_article(URL):
    """
    Extracts article title and text from a URL using Beautiful Soup.
    """
    try:
        response = requests.get(URL)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')

        title = soup.find('title')
        if not title:
            return None

        article_body = None
        common_tags = ['article', 'main']
        common_classes = ['content', 'post', 'article-body', 'entry-content', 'post-content']

        for tag in common_tags:
            article_body = soup.find(tag)
            if article_body:
                break

        if not article_body:
            for class_name in common_classes:
                article_body = soup.find('div', class_=class_name)
                if article_body:
                    break

        if not article_body:
            return None

        for unwanted_tag in article_body.find_all(['header', 'footer', 'nav', 'aside']):
            unwanted_tag.decompose()

        text = article_body.get_text(strip=True, separator='\n')

        return {
            'URL': URL,
            'title': title.text.strip(),
            'text': text
        }

    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        return None

def load_stopwords(filename):
    """Loads stop words from a text file.

    Args:
        filename (str): Path to the stop words file.

    Returns:
        list: A list of stop words.
    """
    with open(filename, "r") as f:
        stopwords_list = f.read().splitlines()
    return stopwords_list
    
def load_sentiment_words(filename):
    """Loads positive or negative words from a text file.

    Args:
        filename (str): Path to the sentiment words file.

    Returns:
        set: A set of sentiment words.
    """
    with open(filename, "r") as f:
        sentiment_words = set(f.read().splitlines())
    return sentiment_words


def clean_text(text):
    """Prepares the text for sentiment analysis.

    Args:
        text (str): The text to be cleaned.

    Returns:
        str: The cleaned text.
    """

    # Load stop words
    stop_words = load_stopwords()

    # Lowercase the text and remove punctuation
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)

    # Tokenize the text and remove stop words
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]

    # Join the tokens back into a string
    cleaned_text = " ".join(tokens)
    return cleaned_text

def analyze_article(text):
    """
    Analyze the article text and calculate various metrics.
    """
    blob = TextBlob(text)
    sentences = blob.sentences
    words = blob.words

    # Word and sentence counts
    word_count = len(words)
    sentence_count = len(sentences)
    
    # Positive, negative, polarity, and subjectivity scores
    positive_words = load_sentiment_words(POSITIVE_WORDS_FILE)
    negative_words = load_sentiment_words(NEGATIVE_WORDS_FILE)

    # Perform sentiment analysis with TextBlob
    textblob_obj = TextBlob(text)

    # Sentiment scores
    positive_score = sum(1 for word in text.split() if word in positive_words)
    negative_score = -1 * sum(1 for word in text.split() if word in negative_words)

    # Polarity score
    polarity_score = (positive_score - negative_score) / (
        max(positive_score + negative_score, 0.000001)
    )

    # Subjectivity score
    subjectivity_score = (positive_score + negative_score) / (
        len(text.split()) + 0.000001
    )

    # Additional metrics
    sentence_count = len(re.split(r'[.!?]', text))
    word_count = len(text.split())

    # Complex words (more than two syllables)
    complex_word_count = sum(
        1
        for word in text.split()
        if count_syllables(word) > 2
        and not (word.endswith("es") or word.endswith("ed"))
    )

    # Average sentence length
    avg_sentence_length = word_count / (sentence_count + 0.000001)

    # Percentage of complex words
    percentage_complex_words = complex_word_count / (word_count + 0.000001)

    # Fog Index
    fog_index = 0.4 * (avg_sentence_length + percentage_complex_words)
    
    syllables_per_word = sum(count_syllables(word) for word in text.split()) / word_count

    avg_word_length = sum(len(word) for word in text.split()) / word_count

    avg_words_per_sentence = word_count / sentence_count

    

    # Personal pronouns (excluding country name "US")
    personal_pronouns = sum(
        1
        for word in text.split()
        if word in ("i", "we", "my", "ours", "us")
        and word.lower() != "us"  # Exclude "US"
    )

    
   


    return {
        'POSITIVE SCORE': positive_score,
        'NEGATIVE SCORE': negative_score,
        'POLARITY SCORE': polarity_score,
        'SUBJECTIVITY SCORE': subjectivity_score,
        'AVG SENTENCE LENGTH': avg_sentence_length,
        'PERCENTAGE OF COMPLEX WORDS': percentage_complex_words,
        'FOG INDEX': fog_index,
        'AVG NUMBER OF WORDS PER SENTENCE': avg_words_per_sentence,
        'COMPLEX WORD COUNT': complex_word_count,
        'WORD COUNT': word_count,
        'SYLLABLE PER WORD': syllables_per_word,
        'PERSONAL PRONOUNS': personal_pronouns,
        'AVG WORD LENGTH': avg_word_length
    }

# List of 100 URL 
URL = ["https://insights.blackcoffer.com/rising-it-cities-and-its-impact-on-the-economy-environment-infrastructure-and-city-life-by-the-year-2040-2/", "https://insights.blackcoffer.com/rising-it-cities-and-their-impact-on-the-economy-environment-infrastructure-and-city-life-in-future/", "https://insights.blackcoffer.com/internet-demands-evolution-communication-impact-and-2035s-alternative-pathways/", "https://insights.blackcoffer.com/rise-of-cybercrime-and-its-effect-in-upcoming-future/", "https://insights.blackcoffer.com/ott-platform-and-its-impact-on-the-entertainment-industry-in-future/", "https://insights.blackcoffer.com/the-rise-of-the-ott-platform-and-its-impact-on-the-entertainment-industry-by-2040/", "https://insights.blackcoffer.com/rise-of-cyber-crime-and-its-effects/", "https://insights.blackcoffer.com/rise-of-internet-demand-and-its-impact-on-communications-and-alternatives-by-the-year-2035-2/", "https://insights.blackcoffer.com/rise-of-cybercrime-and-its-effect-by-the-year-2040-2/", "https://insights.blackcoffer.com/rise-of-cybercrime-and-its-effect-by-the-year-2040/", "https://insights.blackcoffer.com/rise-of-internet-demand-and-its-impact-on-communications-and-alternatives-by-the-year-2035/", "https://insights.blackcoffer.com/rise-of-telemedicine-and-its-impact-on-livelihood-by-2040-3-2/", "https://insights.blackcoffer.com/rise-of-e-health-and-its-impact-on-humans-by-the-year-2030/", "https://insights.blackcoffer.com/rise-of-e-health-and-its-imapct-on-humans-by-the-year-2030-2/", "https://insights.blackcoffer.com/rise-of-telemedicine-and-its-impact-on-livelihood-by-2040-2/", "https://insights.blackcoffer.com/rise-of-telemedicine-and-its-impact-on-livelihood-by-2040-2-2/", "https://insights.blackcoffer.com/rise-of-chatbots-and-its-impact-on-customer-support-by-the-year-2040/", "https://insights.blackcoffer.com/rise-of-e-health-and-its-imapct-on-humans-by-the-year-2030/", "https://insights.blackcoffer.com/how-does-marketing-influence-businesses-and-consumers/", "https://insights.blackcoffer.com/how-advertisement-increase-your-market-value/", "https://insights.blackcoffer.com/negative-effects-of-marketing-on-society/", "https://insights.blackcoffer.com/how-advertisement-marketing-affects-business/", "https://insights.blackcoffer.com/rising-it-cities-will-impact-the-economy-environment-infrastructure-and-city-life-by-the-year-2035/", "https://insights.blackcoffer.com/rise-of-ott-platform-and-its-impact-on-entertainment-industry-by-the-year-2030/", "https://insights.blackcoffer.com/rise-of-electric-vehicles-and-its-impact-on-livelihood-by-2040/", "https://insights.blackcoffer.com/rise-of-electric-vehicle-and-its-impact-on-livelihood-by-the-year-2040/", "https://insights.blackcoffer.com/oil-prices-by-the-year-2040-and-how-it-will-impact-the-world-economy/", "https://insights.blackcoffer.com/an-outlook-of-healthcare-by-the-year-2040-and-how-it-will-impact-human-lives/", "https://insights.blackcoffer.com/ai-in-healthcare-to-improve-patient-outcomes/", "https://insights.blackcoffer.com/what-if-the-creation-is-taking-over-the-creator/", "https://insights.blackcoffer.com/what-jobs-will-robots-take-from-humans-in-the-future/", "https://insights.blackcoffer.com/will-machine-replace-the-human-in-the-future-of-work/", "https://insights.blackcoffer.com/will-ai-replace-us-or-work-with-us/", "https://insights.blackcoffer.com/man-and-machines-together-machines-are-more-diligent-than-humans-blackcoffe/", "https://insights.blackcoffer.com/in-future-or-in-upcoming-years-humans-and-machines-are-going-to-work-together-in-every-field-of-work/", "https://insights.blackcoffer.com/how-machine-learning-will-affect-your-business/", "https://insights.blackcoffer.com/deep-learning-impact-on-areas-of-e-learning/", "https://insights.blackcoffer.com/how-to-protect-future-data-and-its-privacy-blackcoffer/", "https://insights.blackcoffer.com/how-machines-ai-automations-and-robo-human-are-effective-in-finance-and-banking/", "https://insights.blackcoffer.com/ai-human-robotics-machine-future-planet-blackcoffer-thinking-jobs-workplace/", "https://insights.blackcoffer.com/how-ai-will-change-the-world-blackcoffer/", "https://insights.blackcoffer.com/future-of-work-how-ai-has-entered-the-workplace/", "https://insights.blackcoffer.com/ai-tool-alexa-google-assistant-finance-banking-tool-future/", "https://insights.blackcoffer.com/ai-healthcare-revolution-ml-technology-algorithm-google-analytics-industrialrevolution/", "https://insights.blackcoffer.com/all-you-need-to-know-about-online-marketing/", "https://insights.blackcoffer.com/evolution-of-advertising-industry/", "https://insights.blackcoffer.com/how-data-analytics-can-help-your-business-respond-to-the-impact-of-covid-19/", "https://insights.blackcoffer.com/environmental-impact-of-the-covid-19-pandemic-lesson-for-the-future/", "https://insights.blackcoffer.com/how-data-analytics-and-ai-are-used-to-halt-the-covid-19-pandemic/", "https://insights.blackcoffer.com/difference-between-artificial-intelligence-machine-learning-statistics-and-data-mining/", "https://insights.blackcoffer.com/how-python-became-the-first-choice-for-data-science/", "https://insights.blackcoffer.com/how-google-fit-measure-heart-and-respiratory-rates-using-a-phone/", "https://insights.blackcoffer.com/what-is-the-future-of-mobile-apps/", "https://insights.blackcoffer.com/impact-of-ai-in-health-and-medicine/", "https://insights.blackcoffer.com/telemedicine-what-patients-like-and-dislike-about-it/", "https://insights.blackcoffer.com/how-we-forecast-future-technologies/", "https://insights.blackcoffer.com/can-robots-tackle-late-life-loneliness/", "https://insights.blackcoffer.com/embedding-care-robots-into-society-socio-technical-considerations/", "https://insights.blackcoffer.com/management-challenges-for-future-digitalization-of-healthcare-services/", "https://insights.blackcoffer.com/are-we-any-closer-to-preventing-a-nuclear-holocaust/", "https://insights.blackcoffer.com/will-technology-eliminate-the-need-for-animal-testing-in-drug-development/", "https://insights.blackcoffer.com/will-we-ever-understand-the-nature-of-consciousness/", "https://insights.blackcoffer.com/will-we-ever-colonize-outer-space/", "https://insights.blackcoffer.com/what-is-the-chance-homo-sapiens-will-survive-for-the-next-500-years/", "https://insights.blackcoffer.com/why-does-your-business-need-a-chatbot/", "https://insights.blackcoffer.com/how-you-lead-a-project-or-a-team-without-any-technical-expertise/", "https://insights.blackcoffer.com/can-you-be-great-leader-without-technical-expertise/", "https://insights.blackcoffer.com/how-does-artificial-intelligence-affect-the-environment/", "https://insights.blackcoffer.com/how-to-overcome-your-fear-of-making-mistakes-2/", "https://insights.blackcoffer.com/is-perfection-the-greatest-enemy-of-productivity/", "https://insights.blackcoffer.com/global-financial-crisis-2008-causes-effects-and-its-solution/", "https://insights.blackcoffer.com/gender-diversity-and-equality-in-the-tech-industry/", "https://insights.blackcoffer.com/how-to-overcome-your-fear-of-making-mistakes/", "https://insights.blackcoffer.com/how-small-business-can-survive-the-coronavirus-crisis/", "https://insights.blackcoffer.com/impacts-of-covid-19-on-vegetable-vendors-and-food-stalls/", "https://insights.blackcoffer.com/impacts-of-covid-19-on-vegetable-vendors/", "https://insights.blackcoffer.com/impact-of-covid-19-pandemic-on-tourism-aviation-industries/", "https://insights.blackcoffer.com/impact-of-covid-19-pandemic-on-sports-events-around-the-world/", "https://insights.blackcoffer.com/changing-landscape-and-emerging-trends-in-the-indian-it-ites-industry/", "https://insights.blackcoffer.com/online-gaming-adolescent-online-gaming-effects-demotivated-depression-musculoskeletal-and-psychosomatic-symptoms/", "https://insights.blackcoffer.com/human-rights-outlook/", "https://insights.blackcoffer.com/how-voice-search-makes-your-business-a-successful-business/", "https://insights.blackcoffer.com/how-the-covid-19-crisis-is-redefining-jobs-and-services/", "https://insights.blackcoffer.com/how-to-increase-social-media-engagement-for-marketers/", "https://insights.blackcoffer.com/impacts-of-covid-19-on-streets-sides-food-stalls/", "https://insights.blackcoffer.com/coronavirus-impact-on-energy-markets-2/", "https://insights.blackcoffer.com/coronavirus-impact-on-the-hospitality-industry-5/", "https://insights.blackcoffer.com/lessons-from-the-past-some-key-learnings-relevant-to-the-coronavirus-crisis-4/", "https://insights.blackcoffer.com/estimating-the-impact-of-covid-19-on-the-world-of-work-2/", "https://insights.blackcoffer.com/estimating-the-impact-of-covid-19-on-the-world-of-work-3/", "https://insights.blackcoffer.com/travel-and-tourism-outlook/", "https://insights.blackcoffer.com/gaming-disorder-and-effects-of-gaming-on-health/", "https://insights.blackcoffer.com/what-is-the-repercussion-of-the-environment-due-to-the-covid-19-pandemic-situation/", "https://insights.blackcoffer.com/what-is-the-repercussion-of-the-environment-due-to-the-covid-19-pandemic-situation-2/", "https://insights.blackcoffer.com/impact-of-covid-19-pandemic-on-office-space-and-co-working-industries/", "https://insights.blackcoffer.com/contribution-of-handicrafts-visual-arts-literature-in-the-indian-economy/", "https://insights.blackcoffer.com/how-covid-19-is-impacting-payment-preferences/", "https://insights.blackcoffer.com/how-will-covid-19-affect-the-world-of-work-2/"]   

# Loop through each URL and extract data
extracted_data = []
for URL in URL:
    article_data = extract_article(URL)
    if article_data:
        
        analysis = analyze_article(article_data['text'])
    
    
        article_data['URL_ID'] = URL
        article_data.update(analysis)
        
        # Save extracted data to a text file
        filename = f"blackassign{URL.index(URL) + 1}.txt"  # Modify filename format if needed
        with open(filename, 'w', encoding='utf-8') as file:
            file.write(f"URL: {article_data['URL']}\n")
            file.write(f"Title: {article_data['title']}\n\n")
            file.write(article_data['text'] + "\n\n")
            for key, value in analysis.items():
                file.write(f"{key.replace('_', ' ').title()}: {value}\n")

        extracted_data.append(article_data)
    else:
        print(f"Failed to extract data for: {URL}")

# Handle results
if extracted_data:
  # Create pandas DataFrame from list of dictionaries
  df = pd.DataFrame(extracted_data)

  # Select desired columns for output
  output_columns = ['URL_ID', 'URL', 'POSITIVE SCORE', 'NEGATIVE SCORE', 'POLARITY SCORE',
                    'SUBJECTIVITY SCORE', 'AVG SENTENCE LENGTH', 'PERCENTAGE OF COMPLEX WORDS',
                    'FOG INDEX', 'AVG NUMBER OF WORDS PER SENTENCE', 'COMPLEX WORD COUNT', 'WORD COUNT',
                    'SYLLABLE PER WORD', 'PERSONAL PRONOUNS', 'AVG WORD LENGTH']
  df_filtered = df[output_columns]  # Filter relevant columns

  # Save as CSV or XLSX (modify filename as needed)
  df_filtered.to_csv("extracted_data.csv", index=False)  # Save as CSV
  # df_filtered.to_excel("extracted_data.xlsx", index=False)  # Save as XLSX (uncomment to use)
  print(f"{len(extracted_data)} articles extracted and saved successfully!")
else:
    print("No articles extracted.")


