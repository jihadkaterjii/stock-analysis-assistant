<<<<<<< HEAD
# Stock Analysis and Prediction Application

A Flask-based web application that provides stock analysis, predictions, and sentiment analysis using machine learning models and OpenAI's GPT.

## Features

- Real-time stock data analysis
- Stock price predictions using machine learning
- Sentiment analysis of news articles
- Interactive chatbot for stock-related queries
- Historical price visualization
- News article summarization and analysis

## Prerequisites

- Python 3.9 or higher
- Docker (optional, for containerized deployment)
- OpenAI API key
- Required Python packages (listed in requirements.txt)

## Installation

### Local Development Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd stock-analysis-app
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the root directory with your API keys:
```
OPENAI_API_KEY=your_openai_api_key
```

5. Run the application:
```bash
flask run
```

### Docker Deployment

1. Build the Docker image:
```bash
docker build -t stock-prediction-app .
```

2. Run the container:
```bash
docker run -p 5000:5000 --env-file .env stock-prediction-app
```

## Project Structure

```
.
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── Dockerfile            # Docker configuration
├── .dockerignore         # Docker ignore rules
├── templates/            # HTML templates
├── model/               # Machine learning models
├── inference.py         # Prediction logic
├── train_models.py      # Model training scripts
└── fine_tuning.jsonl    # Fine-tuning data
```

## API Endpoints

- `/` - Home page
- `/chatbot` - Chatbot interface
- `/predictions` - Stock predictions
- `/chat` - Chat API endpoint

## Environment Variables

- `FLASK_APP`: Set to `app.py`
- `FLASK_ENV`: Set to `production` or `development`
- `OPENAI_API_KEY`: Your OpenAI API key

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAI for the GPT models
- Pandas and NumPy for data processing
- Plotly for visualizations
- Flask for the web framework 
=======
# stockanalyst
>>>>>>> ad7c1c612bc5fafe769e2133ccc6e3a395bea13c
