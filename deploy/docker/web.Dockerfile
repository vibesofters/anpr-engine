FROM node:22-bookworm-slim@sha256:48e4b67d85f87bd551df43704e24d252f56cc5f8e9718841aace50f19948f0f9 AS builder
ENV NEXT_TELEMETRY_DISABLED=1
WORKDIR /build/apps/web
COPY apps/web/package.json apps/web/package-lock.json ./
RUN npm ci --no-audit --no-fund
WORKDIR /build
COPY apps/web/next.config.ts apps/web/next-env.d.ts apps/web/tsconfig.json apps/web/
COPY apps/web/src apps/web/src
COPY packages/api-contract/generated/typescript/contracts.ts packages/api-contract/generated/typescript/contracts.ts
COPY packages/api-contract/fixtures/valid/processing-notice.json packages/api-contract/fixtures/valid/processing-notice.json
WORKDIR /build/apps/web
RUN npm run build
WORKDIR /build
COPY deploy/docker/create_web_notices.mjs deploy/docker/create_web_notices.mjs
COPY deploy/docker/THIRD-PARTY-NOTICES.sharp-libvips-v1.3.3.md deploy/docker/THIRD-PARTY-NOTICES.sharp-libvips-v1.3.3.md
RUN node deploy/docker/create_web_notices.mjs

FROM node:22-bookworm-slim@sha256:48e4b67d85f87bd551df43704e24d252f56cc5f8e9718841aace50f19948f0f9 AS runtime
ENV NODE_ENV=production NEXT_TELEMETRY_DISABLED=1 PORT=3000 HOSTNAME=0.0.0.0 HOME=/tmp
WORKDIR /app
COPY --from=builder --chown=10001:10001 /build/apps/web/.next/standalone/ ./
COPY --from=builder --chown=10001:10001 /build/apps/web/.next/static/ apps/web/.next/static/
COPY --from=builder --chown=10001:10001 /build/licenses/ /licenses/
RUN mkdir -p /licenses/base && \
    dpkg-query -W -f='${Package}\t${Version}\n' > /licenses/base/debian-packages.tsv && \
    find /usr/share/doc -name copyright | while read -r file; do cp --parents "$file" /licenses/base/; done && \
    find /usr/share/doc -maxdepth 1 -type l -printf '%f -> %l\n' > /licenses/base/debian-doc-aliases.txt && \
    node -e 'fetch("https://raw.githubusercontent.com/nodejs/node/v22.23.2/LICENSE").then(async r=>{if(!r.ok) throw Error("node_license_fetch_failed"); require("fs").writeFileSync("/licenses/base/NODE-LICENSE",await r.text())}).catch(e=>{console.error(e);process.exit(1)})' && \
    node -e 'const f=require("fs"),c=require("crypto");let p="/licenses/base/NODE-LICENSE";f.writeFileSync("/licenses/base/node-license.sha256",c.createHash("sha256").update(f.readFileSync(p)).digest("hex")+"  NODE-LICENSE\n")'
RUN mkdir -p /quota && chown 10001:10001 /quota
USER 10001:10001
EXPOSE 3000
CMD ["node", "apps/web/server.js"]
