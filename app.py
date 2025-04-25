import os
from flask import Flask, render_template, request, jsonify, redirect
import requests
from datetime import datetime, timedelta, date
from pandas_datareader import data as pdr
import plotly.graph_objs as go
from transformers import pipeline
import json
from dotenv import load_dotenv
import warnings
import logging
import openai
from inference import get_enhanced_predictions, get_all_sentiment_classes
import markdown

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Load environment variables
load_dotenv()

# Initialize OpenAI client
openai.api_key = os.getenv('OPENAI_API_KEY')

app = Flask(__name__)

# Initialize sentiment analyzer
sentiment_analyzer = None

# Load fine-tuned responses
def load_fine_tuned_responses():
    responses = {}
    try:
        print("\nAttempting to load fine-tuned responses...")
        file_path = os.path.join(os.path.dirname(__file__), 'fine_tuning.jsonl')
        print(f"Looking for file at: {file_path}")
        
        if not os.path.exists(file_path):
            print(f"Error: fine_tuning.jsonl not found at {file_path}")
            return {}
            
        with open(file_path, 'r', encoding='utf-8') as f:
            line_count = 0
            for line in f:
                line_count += 1
                if line.strip():  # Skip empty lines
                    try:
                        data = json.loads(line)
                        if 'messages' in data and len(data['messages']) >= 3:
                            # Get the user's question and assistant's response
                            user_msg = data['messages'][1]['content'].lower().strip()
                            assistant_msg = data['messages'][2]['content'].strip()
                            
                            # Store both the exact question and keywords
                            responses[user_msg] = {
                                'response': assistant_msg,
                                'keywords': set(user_msg.split())
                            }
                            print(f"Loaded response for: {user_msg[:50]}...")
                    except json.JSONDecodeError as e:
                        print(f"Error parsing line {line_count}: {e}")
                        continue
                    except Exception as e:
                        print(f"Error processing line {line_count}: {e}")
                        continue
                        
        print(f"\nSuccessfully loaded {len(responses)} fine-tuned responses")
        print("Sample questions loaded:")
        for key in list(responses.keys())[:3]:
            print(f"- {key}")
        return responses
    except Exception as e:
        print(f"Error loading fine-tuned responses: {str(e)}")
        import traceback
        traceback.print_exc()
        return {}

# Load responses at startup
print("\nInitializing fine-tuned responses...")
fine_tuned_responses = load_fine_tuned_responses()
print(f"Loaded {len(fine_tuned_responses)} responses")

def get_sentiment_analyzer():
    global sentiment_analyzer
    if sentiment_analyzer is None:
        sentiment_analyzer = pipeline("sentiment-analysis", model="yiyanghkust/finbert-tone")
    return sentiment_analyzer

def get_enhanced_system_prompt():
    return """You are a Stock Analysis Assistant specializing in Pfizer (PFE) and Johnson & Johnson (JNJ).
Your responses should be:
1. Detailed and comprehensive, covering multiple aspects of the company
2. Current and up-to-date, including recent developments
3. Balanced, presenting both positive and negative aspects
4. Professional yet conversational in tone
5. Focused on long-term investment perspectives
6. Based on factual data and market analysis

Format your responses with clear sections using the following structure:
[Section Title]
- Content for this section
- Additional points if needed

When discussing:
- Business models: Cover core operations, revenue streams, and competitive advantages
- Management: Evaluate leadership effectiveness and strategic decisions
- Financials: Analyze key metrics, growth potential, and risk factors
- Investment safety: Consider dividend history, financial stability, and market position
- Acquisitions: Assess strategic fit and integration challenges
- Long-term potential: Evaluate growth drivers and industry trends"""

def get_stock_trend(ticker, days=7):
    """Get stock price trend from Polygon API"""
    print(f"Getting stock trend for {ticker}")
    
    end = datetime.now()
    start = end - timedelta(days=days)

    url = (
        f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/1/day/"
        f"{start.strftime('%Y-%m-%d')}/{end.strftime('%Y-%m-%d')}"
        f"?adjusted=true&sort=asc&limit={days}&apiKey={os.getenv('POLYGON_API_KEY')}"
    )

    try:
        print(f"Making API request to: {url}")
        res = requests.get(url)
        data = res.json()

        if res.status_code != 200:
            print(f"API error: {res.status_code} - {data}")
            return None
            
        if "results" not in data or len(data["results"]) < 2:
            print(f"Not enough data points for {ticker}")
            return None

        start_price = data["results"][0]["c"]
        end_price = data["results"][-1]["c"]
        percent_change = ((end_price - start_price) / start_price) * 100
        direction = "increased" if percent_change > 0 else "decreased"
        
        result = {
            "summary": f"{ticker} stock has {direction} by {abs(percent_change):.2f}% over the last {days} days.",
            "start_price": start_price,
            "end_price": end_price,
            "percent_change": percent_change,
            "direction": direction
        }
        print(f"Generated trend data: {result}")
        return result
        
    except Exception as e:
        print(f"Error fetching price trend for {ticker}: {str(e)}")
        return None

def get_recent_headlines(symbol):
    """Get recent news headlines with sentiment analysis"""
    url = f"https://api.polygon.io/v2/reference/news?ticker={symbol}&limit=10&apiKey={os.getenv('POLYGON_API_KEY')}"
    
    try:
        res = requests.get(url)
        if res.status_code != 200:
            return []

        articles = res.json().get("results", [])
        headlines = []
        analyzer = get_sentiment_analyzer()
        
        for article in articles:
            text = article.get("description", "") or article.get("title", "")
            title = article.get("title", "").lower()
            
            relevant = False
            if symbol == "PFE":
                relevant = "pfizer" in title or "pfe" in title
            elif symbol == "JNJ":
                relevant = any(term in title for term in ["johnson", "jnj", "j&j"])
                
            if relevant:
                sentiment = analyzer(text[:512])[0]['label']
                headlines.append({
                    "title": article["title"],
                    "summary": text,
                    "sentiment": sentiment
                })
        return headlines
    except Exception as e:
        print(f"Error fetching headlines: {str(e)}")
        return []

def search_news_articles(news_list, query):
    """Search articles related to a user query"""
    query_lower = query.lower()
    return [article for article in news_list if query_lower in article['title'].lower() or query_lower in article['summary'].lower()]

def explain_news_topic(news_articles, query, company):
    """Explain news on a specific topic using GPT"""
    if not news_articles:
        return f"I couldn't find any recent news about {company} related to '{query}'."

    summaries = "\n".join(
        f"- {a['title']} ({a['sentiment']}): {a['summary']}" for a in news_articles[:3]
    )

    prompt = f"""
You are a financial assistant helping users understand company-related news.

Here are recent headlines about {company} related to '{query}':
{summaries}

Please summarize this topic clearly and conversationally.
"""

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a stock assistant that summarizes company news topics with clarity and accuracy."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1000,
        temperature=0.7
    )
    return response.choices[0].message.content

def get_sentiment_summary(news):
    """Summarize sentiment from news articles"""
    if not news:
        return "No recent news available."

    scores = {"Positive": 0, "Neutral": 0, "Negative": 0}
    examples = {"Positive": [], "Neutral": [], "Negative": []}

    for article in news:
        sentiment = article['sentiment']
        scores[sentiment] += 1
        examples[sentiment].append(article['title'])

    summary_parts = []
    if scores["Positive"]:
        summary_parts.append(f"Positive sentiment was reflected in {scores['Positive']} article(s), such as: {', '.join(examples['Positive'][:2])}.")
    if scores["Neutral"]:
        summary_parts.append(f"There were {scores['Neutral']} neutral updates, including: {', '.join(examples['Neutral'][:2])}.")
    if scores["Negative"]:
        summary_parts.append(f"Concerns appeared in {scores['Negative']} article(s), like: {', '.join(examples['Negative'][:2])}.")

    return "Recent news coverage reflects the following:\n" + "\n".join(summary_parts)

def extract_stocks_from_input(user_input):
    """Extract company mentions from user input"""
    print(f"Extracting stocks from input: {user_input}")
    
    stock_map = {
        "pfizer": ("Pfizer", "PFE"),
        "pfe": ("Pfizer", "PFE"),
        "johnson & johnson": ("Johnson & Johnson", "JNJ"),
        "johnson and johnson": ("Johnson & Johnson", "JNJ"),
        "jnj": ("Johnson & Johnson", "JNJ"),
        "j&j": ("Johnson & Johnson", "JNJ")
    }

    input_lower = user_input.lower().strip()
    found = []
    
    # First, try exact matches
    for key, val in stock_map.items():
        if key in input_lower and val not in found:
            found.append(val)
            print(f"Found stock: {val}")

    # If no exact matches, try more flexible matching
    if not found:
        if any(word in input_lower for word in ["pfe", "pfizer"]):
            found.append(("Pfizer", "PFE"))
            print("Found Pfizer through flexible matching")
        if any(word in input_lower for word in ["jnj", "johnson"]):
            found.append(("Johnson & Johnson", "JNJ"))
            print("Found J&J through flexible matching")

    print(f"Extracted stocks: {found}")
    return found

def get_fine_tuned_response(user_input):
    """Get matching response from fine-tuned examples"""
    try:
        print(f"\nSearching for response to: {user_input}")
        user_input_lower = user_input.lower()
        best_match = None
        best_score = 0
        
        # Read and process each line of the JSONL file
        with open('fine_tuning.jsonl', 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():  # Skip empty lines
                    continue
                    
                try:
                    # Parse each line as a separate JSON object
                    data = json.loads(line)
                    
                    if 'messages' in data and len(data['messages']) >= 3:
                        example_question = data['messages'][1]['content'].lower()
                        example_answer = data['messages'][2]['content']
                        
                        # Calculate match score based on word overlap and key phrases
                        user_words = set(user_input_lower.split())
                        example_words = set(example_question.split())
                        common_words = user_words & example_words
                        
                        # Calculate base score from word overlap
                        score = len(common_words)
                        
                        # Boost score for key phrases and company names
                        key_phrases = {
                            'business model': 3,
                            'management': 3,
                            'financials': 3,
                            'earnings': 3,
                            'dividend': 3,
                            'acquisition': 3,
                            'pipeline': 3,
                            'performance': 2,
                            'guidance': 2,
                            'outlook': 2,
                            'growth': 2,
                            'risk': 2,
                            'valuation': 2,
                            'pfizer': 2,
                            'pfe': 2,
                            'johnson': 2,
                            'jnj': 2,
                            'j&j': 2
                        }
                        
                        # Add bonus points for matching key phrases
                        for phrase, bonus in key_phrases.items():
                            if phrase in user_input_lower and phrase in example_question:
                                score += bonus
                                print(f"Matched key phrase '{phrase}' (+{bonus} points)")
                        
                        # If this is the best match so far, update it
                        if score > best_score:
                            best_score = score
                            best_match = example_answer
                            print(f"New best match (score {score}):")
                            print(f"Q: {example_question[:100]}...")
                            print(f"A: {example_answer[:100]}...")
                            
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON line: {str(e)}")
                    continue
                    
        # Return the best match if it has a reasonable score
        if best_score >= 3:
            print(f"\nUsing best match with score {best_score}")
            return best_match
        else:
            print("\nNo good match found")
            return None
            
    except Exception as e:
        print(f"Error in get_fine_tuned_response: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def generate_response(user_input):
    """Generate a response using fine-tuned model and ChatGPT"""
    try:
        # Get fine-tuned response
        fine_tuned_response = get_fine_tuned_response(user_input)
        
        # Get ChatGPT response
        chatgpt_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a Stock Analysis Assistant that helps users understand information about Pfizer (PFE) and Johnson & Johnson (JNJ). You provide insightful, up-to-date, and conversational answers. Include the current date in your response."},
                {"role": "user", "content": user_input}
            ],
            max_tokens=1000,
            temperature=0.7
        ).choices[0].message.content

        # Get stocks mentioned in the query
        stocks = extract_stocks_from_input(user_input)
        
        # Build the response
        response_parts = []
        
        # Add expert analysis from fine-tuned response
        if fine_tuned_response:
            response_parts.append("**Assistant's Response:**")
            response_parts.append(fine_tuned_response)
            response_parts.append("")  # Add spacing
        
        # Add ChatGPT's perspective
        response_parts.append("**ChatGPT's Perspective:**")
        response_parts.append(chatgpt_response)
        response_parts.append("")  # Add spacing
        
        # Add real-time data for mentioned stocks
        for company_name, ticker in stocks:
            # Get stock trend
            trend_data = get_stock_trend(ticker)
            if trend_data:
                response_parts.append(f"**[{company_name} Stock Performance]**")
                response_parts.append(trend_data['summary'])
                
                if trend_data['direction'] == 'increased':
                    response_parts.append(f"• Current Price: ${trend_data['end_price']:.2f} (↑)")
                    response_parts.append(f"• Change: +${abs(trend_data['end_price'] - trend_data['start_price']):.2f} (+{abs(trend_data['percent_change']):.2f}%)")
                else:
                    response_parts.append(f"• Current Price: ${trend_data['end_price']:.2f} (↓)")
                    response_parts.append(f"• Change: -${abs(trend_data['end_price'] - trend_data['start_price']):.2f} (-{abs(trend_data['percent_change']):.2f}%)")
                response_parts.append("")  # Add spacing
            
            # Get news
            news = get_recent_headlines(ticker)
            if news:
                sentiment_summary = get_sentiment_summary(news)
                response_parts.append(f"**[Recent {company_name} News]**")
                response_parts.append(sentiment_summary)
                response_parts.append("")  # Add spacing
        
        # If no response was generated, provide help message
        if not response_parts:
            return """Please ask me about Pfizer (PFE) or Johnson & Johnson (JNJ).
You can ask about:
• Company analysis and insights
• Current stock performance
• Recent news and developments
• Investment considerations"""
        
        return "\n".join(response_parts)

    except Exception as e:
        print(f"Error generating response: {str(e)}")
        return "I apologize, but I encountered an error while processing your request. Please try again."

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chatbot")
def chatbot():
    return render_template("chatbot.html")

@app.route("/predictions")
def predictions():
    results = get_enhanced_predictions()
    dates = [datetime.now().date() + timedelta(days=i+1) for i in range(7)]
    dates_str = [d.strftime("%b %d") for d in dates]

    graphs = []
    for stock in ["pfe", "jnj"]:
        base = results[stock]["base_predictions"]
        adjusted = results[stock]["predictions"]
        historical = get_historical_prices(stock.upper())[-15:]
        historical_dates = [(datetime.now() - timedelta(days=len(historical)-i)).strftime("%b %d") for i in range(len(historical))]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=historical_dates, y=historical, mode='lines+markers', name='Historical'))
        fig.add_trace(go.Scatter(x=dates_str, y=adjusted, mode='lines+markers', name='Predicted'))
        fig.update_layout(title=stock.upper() + " Analysis", xaxis_title='Date', yaxis_title='Price ($)', height=400)

        graphs.append(fig.to_html(full_html=False))

    return render_template("predictions.html", graphs=graphs)

def get_historical_prices(ticker):
    end = datetime.now()
    start = end - timedelta(days=30)
    df = pdr.DataReader(ticker, "stooq", start, end)
    return df['Close'].values[::-1]

@app.route("/chat", methods=["POST"])
def chat():
    try:
        print("Received chat request")
        
        if not request.is_json:
            print("Error: Request is not JSON")
            return jsonify({
                "response": "Invalid request format. Please send a JSON request."
            }), 400
            
        message = request.json.get("message", "")
        print(f"Received message: {message}")
        
        if not message:
            print("Error: Empty message")
            return jsonify({
                "response": "Please provide a message."
            }), 400
        
        # Generate response
        try:
            response = generate_response(message)
            print(f"Generated response: {response[:100]}...")
            return jsonify({"response": response})
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return jsonify({
                "response": "I apologize, but I encountered an error while processing your request. Please try again."
            }), 500
            
    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        return jsonify({
            "response": "An unexpected error occurred. Please try again."
        }), 500

if __name__ == "__main__":
    print("\n" + "="*50)
    print("\nStock Analysis Dashboard is running!")
    print("\nAccess the application at: http://127.0.0.1:5000")
    print("\n" + "="*50 + "\n")
    
    app.run(debug=True)