FROM node:22-alpine AS build

WORKDIR /app

COPY apps/web/package.json ./package.json
RUN npm install

COPY apps/web/ ./
RUN npm run build

FROM nginx:1.27-alpine AS runtime

COPY infra/docker/web.nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist/indexer-core-web /usr/share/nginx/html

EXPOSE 80
