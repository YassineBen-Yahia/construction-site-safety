# Use official Redis image as base
FROM redis:7-alpine

# Expose Redis default port
EXPOSE 6379

# Set default command to start Redis server
CMD ["redis-server", "--appendonly", "yes"]
