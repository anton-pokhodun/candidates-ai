#!/bin/bash

set -e

echo "================================================"
echo "Candidates AI - Setup Script"
echo "================================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if .env exists
if [ ! -f .env ]; then
  echo -e "${YELLOW}Creating .env file...${NC}"
  cat >.env <<EOF
OPENAI_API_KEY=your-api-key-here
EOF
  echo -e "${RED}⚠️  Please update .env with your actual OpenAI API key${NC}"
  echo ""
  read -p "Press enter to continue after updating .env file..."
fi

# Check if OPENAI_API_KEY is set
source .env
if [ "$OPENAI_API_KEY" = "your-api-key-here" ]; then
  echo -e "${RED}Error: Please set your OPENAI_API_KEY in .env file${NC}"
  exit 1
fi

# Create necessary directories
echo "Creating directories..."
mkdir -p chroma_db data

echo -e "${GREEN}Setting up local Python environment...${NC}"

# Install dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Run data ingestion
echo "Running data ingestion..."
python persist.py

echo ""
echo -e "${GREEN}✓ Setup complete!${NC}"
echo ""
echo "To start the application:"
echo ""
echo "1. Start the backend server:"
echo "   make run"
echo "   or"
echo "   fastapi dev backend.py"
echo ""
echo "2. In a separate terminal, start the frontend server:"
echo "   python -m http.server 8080"
echo ""
echo "3. Open your browser and navigate to:"
echo "   http://localhost:8080"
echo ""
