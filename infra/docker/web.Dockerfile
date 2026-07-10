FROM node:22.16.0-bookworm-slim AS build

WORKDIR /app

COPY apps/web/package.json apps/web/package-lock.json apps/web/.npmrc ./
RUN npm ci --no-audit --no-fund

COPY apps/web/ ./
RUN npm run build

FROM nginx:1.27-alpine AS runtime

ENV NGINX_CLIENT_MAX_BODY_SIZE=250m

COPY infra/docker/web.nginx.conf.template /etc/nginx/templates/default.conf.template
COPY --from=build /app/dist/indexer-core-web /usr/share/nginx/html

EXPOSE 80
