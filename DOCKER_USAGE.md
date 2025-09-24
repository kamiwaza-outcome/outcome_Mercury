# Mercury Blue ALLY - Docker Usage Guide

## Overview
Mercury Blue ALLY is fully configured to run in Docker containers, providing a consistent deployment environment across all platforms.

## Architecture
The application consists of the following services:
- **Frontend**: Next.js application (Port 3003)
- **Backend**: FastAPI application with Kamiwaza integration (Port 8003)
- **Milvus**: Vector database for RAG operations (Port 19530)
- **Redis**: Cache service (Port 6379)
- **Supporting Services**: etcd, MinIO for Milvus storage

## Quick Start

### 1. Prerequisites
- Docker Desktop installed and running
- Kamiwaza running on host machine (port 7777) or configured endpoint

### 2. Build and Run

```bash
# Make scripts executable (first time only)
chmod +x docker-build.sh docker-run.sh

# Build Docker images
./docker-build.sh

# Start all services
./docker-run.sh
```

### 3. Access the Application
- Frontend: http://localhost:3003
- Backend API: http://localhost:8003
- API Documentation: http://localhost:8003/docs
- MinIO Console: http://localhost:9001 (minioadmin/minioadmin)

## Configuration

### Environment Variables
Copy `.env.docker` to `.env` and configure:

```bash
cp .env.docker .env
```

Key configurations:
- `KAMIWAZA_ENDPOINT`: URL to Kamiwaza service
- `KAMIWAZA_DEFAULT_MODEL`: Default AI model to use
- `MILVUS_HOST`: Vector database host

### Development Mode
For local development with hot-reload:

```bash
./docker-run.sh dev
```

This mode:
- Mounts source code as volumes
- Enables hot-reload for both frontend and backend
- Assumes Kamiwaza runs on host machine

## Docker Commands

### Service Management
```bash
# Start services
./docker-run.sh

# Stop services
./docker-run.sh down

# View logs
./docker-run.sh logs

# Clean everything (removes volumes)
./docker-run.sh clean
```

### Manual Docker Compose Commands
```bash
# Start services
docker-compose up -d

# Stop services
docker-compose down

# View logs
docker-compose logs -f

# Rebuild images
docker-compose build

# Development mode
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up
```

## Service Health Checks

All services include health checks:
- Backend: `http://localhost:8003/api/health`
- Frontend: `http://localhost:3003`
- Milvus: `http://localhost:9091/healthz`

## Volumes

Persistent data is stored in Docker volumes:
- `backend-data`: Application data
- `milvus-data`: Vector database storage
- `redis-data`: Cache storage
- `kamiwaza-models`: AI model cache

## Troubleshooting

### Kamiwaza Connection Issues
If Kamiwaza is running on your host machine:
- Use `host.docker.internal:7777` in `.env`
- Ensure Kamiwaza is running before starting Docker containers

### Port Conflicts
If ports are already in use, modify the port mappings in `docker-compose.yml`:
```yaml
ports:
  - "3003:3000"  # Change 3003 to another port
```

### Build Failures
- Check `requirements.txt` for dependency conflicts
- Ensure Docker has sufficient resources allocated
- Try cleaning and rebuilding: `./docker-run.sh clean && ./docker-build.sh`

### Performance Issues
- Allocate more resources to Docker Desktop
- Use production builds for better performance
- Consider using external Milvus/Redis instances for production

## Production Deployment

For production deployment:
1. Update `.env` with production values
2. Use external databases (Milvus, Redis)
3. Configure proper SSL/TLS
4. Set up monitoring and logging
5. Use container orchestration (Kubernetes, Docker Swarm)

## Development Workflow

1. Make code changes
2. If using dev mode, changes auto-reload
3. If using production mode, rebuild: `docker-compose build`
4. Test changes at http://localhost:3003

## Security Notes

- Change default passwords in production
- Use secrets management for sensitive data
- Configure firewall rules for exposed ports
- Regular security updates for base images

## Support

For issues or questions:
- Check application logs: `docker-compose logs [service-name]`
- Verify service health endpoints
- Ensure all required environment variables are set
- Check Kamiwaza connection and model availability