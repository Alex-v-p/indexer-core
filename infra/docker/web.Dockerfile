FROM node:22-alpine AS build

WORKDIR /app

COPY apps/web/package.json ./package.json
RUN npm install

COPY apps/web/ ./
RUN npm run build

FROM nginx:1.27-alpine AS runtime

ENV NGINX_CLIENT_MAX_BODY_SIZE=250m

COPY infra/docker/web.nginx.conf.template /etc/nginx/templates/default.conf.template
COPY --from=build /app/dist/indexer-core-web /usr/share/nginx/html

EXPOSE 80
