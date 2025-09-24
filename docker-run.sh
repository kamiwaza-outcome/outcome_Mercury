#!/bin/bash

# Docker Run Script for Mercury Blue ALLY
# This script starts the Docker containers for the application

set -e

echo "🚀 Starting Mercury Blue ALLY with Docker..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
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

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

# Check if Docker is running
if ! docker info &> /dev/null; then
    print_error "Docker is not running. Please start Docker Desktop."
    exit 1
fi

# Check if .env file exists
if [ ! -f .env ]; then
    print_error ".env file not found. Please run './docker-build.sh' first."
    exit 1
fi

# Parse command line arguments
MODE="prod"
if [ "$1" = "dev" ]; then
    MODE="dev"
elif [ "$1" = "down" ]; then
    print_status "Stopping all containers..."
    docker-compose down
    print_status "All containers stopped."
    exit 0
elif [ "$1" = "logs" ]; then
    docker-compose logs -f
    exit 0
elif [ "$1" = "clean" ]; then
    print_warning "This will remove all containers, volumes, and networks."
    read -p "Are you sure? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        docker-compose down -v --remove-orphans
        print_status "Cleanup complete."
    fi
    exit 0
fi

# Start containers based on mode
if [ "$MODE" = "dev" ]; then
    print_info "Starting in DEVELOPMENT mode..."
    print_warning "Make sure Kamiwaza is running on your host machine at port 7777"
    docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d
else
    print_info "Starting in PRODUCTION mode..."
    docker-compose up -d
fi

# Wait for services to be healthy
print_info "Waiting for services to be ready..."
sleep 5

# Check service health
echo ""
print_status "Service Status:"

# Check backend
if curl -f http://localhost:8003/api/health &> /dev/null; then
    print_status "Backend: Running at http://localhost:8003"
else
    print_warning "Backend: Not responding (may still be starting)"
fi

# Check frontend
if curl -f http://localhost:3003 &> /dev/null; then
    print_status "Frontend: Running at http://localhost:3003"
else
    print_warning "Frontend: Not responding (may still be starting)"
fi

# Check Milvus
if curl -f http://localhost:9091/healthz &> /dev/null; then
    print_status "Milvus: Running at http://localhost:19530"
else
    print_warning "Milvus: Not responding (may still be starting)"
fi

# Show running containers
echo ""
print_info "Running containers:"
docker-compose ps

echo ""
echo "📋 Application URLs:"
echo "   Frontend:    http://localhost:3003"
echo "   Backend API: http://localhost:8003"
echo "   Minio Console: http://localhost:9001 (minioadmin/minioadmin)"
echo ""
echo "📝 Commands:"
echo "   View logs:    ./docker-run.sh logs"
echo "   Stop:         ./docker-run.sh down"
echo "   Clean all:    ./docker-run.sh clean"
echo "   Dev mode:     ./docker-run.sh dev"

# Check Kamiwaza connection
echo ""
print_info "Checking Kamiwaza connection..."
HEALTH_RESPONSE=$(curl -s http://localhost:8003/api/health 2>/dev/null || echo "{}")
if echo "$HEALTH_RESPONSE" | grep -q '"healthy":true'; then
    print_status "Kamiwaza: Connected"
else
    print_warning "Kamiwaza: Not connected - Make sure Kamiwaza is running"
    print_info "To use Kamiwaza on host machine, update KAMIWAZA_ENDPOINT in .env"
fi