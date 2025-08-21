#!/bin/bash

# BCSF Monitoring Interface Docker Runner
# This script helps you run the application in Docker

set -e

echo "BCSF Monitoring Interface Docker Setup"
echo "======================================"

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "ERROR: .env file not found!"
    echo "Please create a .env file with the following variables:"
    echo ""
    echo "PROJECT_ID=scope3-dev"
    echo "ENVIRONMENT=development"
    echo "RESEARCH_BUCKET=your-research-bucket-name"
    echo "META_BRAND_SAFETY_OUTPUT_BUCKET=your-meta-brand-safety-bucket-name"
    echo "CHECKING_BUCKET=your-checking-bucket-name"
    echo ""
    echo "You can copy the example from README.md"
    exit 1
fi

# Check if credentials.json file exists
if [ ! -f "credentials.json" ]; then
    echo "ERROR: credentials.json file not found!"
    echo "Please place your Google Cloud service account credentials file at the root of the project."
    echo "The file should be named 'credentials.json' and contain your service account key."
    echo ""
    echo "You can download it from Google Cloud Console:"
    echo "1. Go to IAM & Admin > Service Accounts"
    echo "2. Create or select a service account"
    echo "3. Create a new key (JSON format)"
    echo "4. Download and rename to 'credentials.json'"
    echo "5. Place it in the project root directory"
    exit 1
fi

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "ERROR: Docker is not running. Please start Docker and try again."
    exit 1
fi

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null; then
    echo "ERROR: docker-compose is not installed. Please install it and try again."
    exit 1
fi

echo "SUCCESS: Environment check passed"
echo "SUCCESS: credentials.json found"
echo ""

# Stop existing containers if running
echo "Stopping existing containers..."
docker-compose down

# Build and run the application
echo "Building and starting the application..."
docker-compose up --build

echo ""
echo "SUCCESS: Application is starting up!"
echo "Access the application at: http://localhost:80"
echo ""
echo "Useful commands:"
echo "  - View logs: docker-compose logs -f"
echo "  - Stop app: docker-compose down"
echo "  - Restart app: docker-compose restart"
echo "  - Rebuild: docker-compose up --build"
echo ""
echo "Waiting for application to be ready..."
echo "   (This may take a few moments on first run)"

# Wait for the application to be ready
attempts=0
max_attempts=30
while [ $attempts -lt $max_attempts ]; do
    if curl -f http://localhost:80/_stcore/health > /dev/null 2>&1; then
        echo "SUCCESS: Application is ready!"
        echo "Open your browser and go to: http://localhost:80"
        break
    fi
    echo "Waiting for application to start... (attempt $((attempts + 1))/$max_attempts)"
    sleep 2
    attempts=$((attempts + 1))
done

if [ $attempts -eq $max_attempts ]; then
    echo "WARNING: Application may still be starting up. Please check:"
    echo "   - http://localhost:80"
    echo "   - docker-compose logs -f"
fi