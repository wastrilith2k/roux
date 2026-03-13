#!/bin/bash
# Start the companion in production mode

set -e

echo "🚀 Starting Companion in PRODUCTION mode..."

# Use production docker-compose with prod Caddyfile
export CADDYFILE=Caddyfile.prod

# Start services with production config
docker compose -f docker-compose.prod.yml up -d

echo "✅ Production environment started!"
echo ""
echo "📍 Access points:"
echo "   Main site:       https://${DOMAIN:-localhost}"
echo "   n8n workflows:   https://n8n.${DOMAIN:-localhost}"
echo ""
echo "💡 Note: Image consumer may need manual restart if RabbitMQ isn't ready:"
echo "   docker compose -f docker-compose.prod.yml up -d image-generation-consumer"
