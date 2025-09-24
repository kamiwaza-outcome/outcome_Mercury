#!/bin/bash

# Docker Build Script for Mercury Blue ALLY
# This script builds all Docker images for the application

set -e

echo "🚀 Building Mercury Blue ALLY Docker Images..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if docker-compose is installed
if ! command -v docker-compose &> /dev/null; then
    print_error "Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Copy environment file if it doesn't exist
if [ ! -f .env ]; then
    print_warning ".env file not found. Creating from .env.docker..."
    cp .env.docker .env
    print_status "Created .env file. Please update it with your settings."
fi

# Build images
print_status "Building Docker images..."

# Build with docker-compose
if [ "$1" = "dev" ]; then
    print_status "Building development images..."
    docker-compose -f docker-compose.yml -f docker-compose.dev.yml build --parallel
else
    print_status "Building production images..."
    docker-compose build --parallel
fi

print_status "Docker images built successfully!"

echo ""
echo "📋 Next steps:"
echo "1. Update the .env file with your Kamiwaza and other configurations"
echo "2. Run './docker-run.sh' to start the application"
echo "3. Access the application at http://localhost:3003"

# Show built images
echo ""
echo "Built images:"
docker images | grep mercury-