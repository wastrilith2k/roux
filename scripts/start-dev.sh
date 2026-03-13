#!/bin/bash
# Start the companion in development mode (local environment)

set -e

echo "🚀 Starting Companion in DEVELOPMENT mode..."

# Use base docker-compose with dev Caddyfile
export CADDYFILE=Caddyfile.dev

# Start services
docker compose -f docker-compose.yml up -d

echo "✅ Development environment started!"
echo ""
echo "📍 Access points:"
echo "   Frontend:        http://localhost:3000"
echo "   Backend API:     http://localhost:5000"
echo "   RabbitMQ Admin:  http://localhost:15672 (user: companion, pass: see .env)"
echo "   Neo4j Browser:   http://localhost:7474"
echo ""
echo "💡 Note: Image consumer may need manual restart if RabbitMQ isn't ready:"
echo "   docker compose up -d image-generation-consumer"
