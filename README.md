# Below Common Sense Floor Checking Interface

A Streamlit-based interface for reviewing and correcting moderation results of our below common sense floor layer

## Features

- Review and correct classification results for multimodal content
- View images and associated text content with automatic translation
- Evaluate classification results and data collection quality
- Support for multiple languages with automatic translation
- Integration with BigQuery for data querying
- Search by Artifact ID or date range
- Export corrections to CSV

## Prerequisites

- Python 3.9 or higher (recommended: Python 3.11)
- Google Cloud credentials configured
- Access to BigQuery and Google Cloud Storage
- Docker and Docker Compose installed

## Installation

### Option 1: Local Installation

1. Clone the repository:
```bash
git clone https://github.com/scope3data/bcsf-monitoring-interface.git
cd bcsf-monitoring-interface
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
# Using pip
pip install -r requirements.txt

# Or using uv for faster installs
uv pip install -r requirements.txt
```

4. Create a .env file with your configuration:
```bash
# Project configuration
PROJECT_ID=scope3-dev
ENVIRONMENT=development

# Bucket configurations
RESEARCH_BUCKET=your-research-bucket-name
META_BRAND_SAFETY_OUTPUT_BUCKET=your-meta-brand-safety-bucket-name
CHECKING_BUCKET=your-checking-bucket-name
```

5. Set up Google Cloud credentials:
```bash
export GOOGLE_APPLICATION_CREDENTIALS="path/to/your/credentials.json"
```

### Option 2: Docker Installation (Recommended)

1. Clone the repository:
```bash
git clone https://github.com/scope3data/bcsf-monitoring-interface.git
cd bcsf-monitoring-interface
```

2. Create a .env file with your configuration (same as above)

3. Download your Google Cloud service account credentials:
   - Go to Google Cloud Console > IAM & Admin > Service Accounts
   - Create or select a service account
   - Create a new key (JSON format)
   - Download and rename to `credentials.json`
   - Place it in the project root directory

4. Run the application using Docker:
```bash
# Using the provided script (recommended)
./reload_docker.sh

# Or manually with docker-compose
docker-compose up --build
```

5. Access the application at `http://localhost:80`

## Usage

### Local Development

1. Set up your Google Cloud credentials:
```bash
export GOOGLE_APPLICATION_CREDENTIALS="path/to/your/credentials.json"
```

2. Run the application:
```bash
streamlit run app.py
```

3. Open your browser and navigate to `http://localhost:8501`

### Docker Deployment

The application is configured to run on port 80 when using Docker. Simply run:

```bash
./reload_docker.sh
```

Or manually:

```bash
# Build and start (runs in foreground)
docker-compose up --build

# View logs (in another terminal)
docker-compose logs -f

# Stop the application (Ctrl+C or in another terminal)
docker-compose down

# Restart
docker-compose restart
```

## Features

### Search Options
- **Date-based Search**: Query content by date, limit, and classification value
- **Artifact ID Search**: Direct lookup of specific content by Artifact ID
- **Classification Filtering**: Filter by Safe (0), Below Common Sense Floor (1), or Any

### Content Review
- **Image Display**: View images from multiple bucket sources
- **Text Content**: Display original text with automatic translation for non-English content
- **Image Captions**: Show image captions (English only)
- **Metadata**: View artifact ID, language, classification value, and timestamps

### Correction System
- **Data Collection Quality**: Evaluate data collection quality (Good/Bad)
- **Classification Correction**: Correct classifications (Floor/Not Floor)
- **Export Functionality**: Download corrections as CSV

## Project Structure

```
bcsf-monitoring-interface/
├── app.py              # Main application file
├── requirements.txt    # Python dependencies
├── logger.py          # Logging configuration
├── Dockerfile         # Docker configuration
├── docker-compose.yml # Docker Compose configuration
├── reload_docker.sh   # Docker runner script
├── .dockerignore      # Docker ignore file
├── README.md          # Project documentation
├── .env               # Environment variables (create this)
├── credentials.json   # Google Cloud credentials (create this)
└── .gitignore         # Git ignore file
```

## Configuration

The application requires:
- **BigQuery Access**: For querying classification data
- **Google Cloud Storage**: For accessing images and assets
- **Environment Variables**: Configured in `.env` file
- **Google Cloud Credentials**: `credentials.json` file in project root

## Docker Configuration

The Docker setup includes:
- **Port 80**: Application exposed on port 80
- **Health Checks**: Automatic health monitoring
- **Environment Variables**: Passed from `.env` file
- **Google Cloud Credentials**: Mounted from `credentials.json` in project root
- **Non-root User**: Security best practices
- **Foreground Mode**: Runs in foreground for better logging and debugging


## Local Development

This project is designed for local development only. To run:

1. Ensure you have the required Google Cloud permissions
2. Set up your Google Cloud credentials
3. Run `streamlit run app.py`
4. Access the interface at `http://localhost:8501`

## License

This is a private project. All rights reserved.
