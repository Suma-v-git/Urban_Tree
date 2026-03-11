#!/bin/bash

# Urban Tree Segmentation Deployment Script

echo "🌳 Urban Tree Segmentation - Deployment Script"
echo "=============================================="

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Create docker directory if it doesn't exist
mkdir -p docker

# Build and run the application
echo "🔨 Building Docker image..."
docker-compose -f docker/docker-compose.yml build

echo "🚀 Starting application..."
docker-compose -f docker/docker-compose.yml up -d

echo "⏳ Waiting for application to start..."
sleep 30

# Check if the application is running
if curl -f http://localhost:8501/_stcore/health &> /dev/null; then
    echo "✅ Application is running successfully!"
    echo "🌐 Open your browser and go to: http://localhost:8501"
else
    echo "❌ Application failed to start. Check logs with:"
    echo "   docker-compose -f docker/docker-compose.yml logs"
fi

echo ""
echo "📋 Useful commands:"
echo "  View logs: docker-compose -f docker/docker-compose.yml logs -f"
echo "  Stop app: docker-compose -f docker/docker-compose.yml down"
echo "  Restart app: docker-compose -f docker/docker-compose.yml restart"
echo "  Access container: docker exec -it urban-tree-app bash"
