# syntax=docker/dockerfile:1
# =============================================================================
# SyntheticDataSuite Frontend
# =============================================================================
# Next.js frontend with React Flow for workflow visualization
#
# Build:
#   eval $(minikube docker-env)
#   docker build -t syntheticdata-frontend:latest \
#     --build-arg NEXT_PUBLIC_API_BASE_URL=http://syntheticdata-backend:8000 \
#     ./frontend
#
# Run locally (for testing):
#   docker run -p 3000:3000 syntheticdata-frontend:latest
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Install dependencies
# -----------------------------------------------------------------------------
FROM node:20-alpine AS deps

WORKDIR /app

# Copy package files
COPY package.json package-lock.json ./

# Install dependencies
RUN npm ci --only=production=false

# -----------------------------------------------------------------------------
# Stage 2: Build the application
# -----------------------------------------------------------------------------
FROM node:20-alpine AS builder

WORKDIR /app

# Copy dependencies from deps stage
COPY --from=deps /app/node_modules ./node_modules
COPY . .

# Build-time environment variable for API URL
# This gets embedded into the JavaScript bundle during build
ARG NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
ENV NEXT_PUBLIC_API_BASE_URL=$NEXT_PUBLIC_API_BASE_URL

# Build the Next.js application
RUN npm run build

# -----------------------------------------------------------------------------
# Stage 3: Production runner
# -----------------------------------------------------------------------------
FROM node:20-alpine AS runner

WORKDIR /app

# Set production environment
ENV NODE_ENV=production

# Create non-root user for security
RUN addgroup --system --gid 1001 nodejs \
    && adduser --system --uid 1001 nextjs

# Copy the standalone output and static assets
COPY --from=builder /app/public ./public
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static

# Switch to non-root user
USER nextjs

# Expose the application port
EXPOSE 3000

# Set the hostname to listen on all interfaces
ENV HOSTNAME="0.0.0.0"
ENV PORT=3000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:3000 || exit 1

# Start the Next.js server
CMD ["node", "server.js"]
