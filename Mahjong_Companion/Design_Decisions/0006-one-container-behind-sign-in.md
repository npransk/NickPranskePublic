# 6. One container behind Microsoft sign-in, image in GitHub's registry

Date: 2026-10-23 · Status: accepted

## Context
The site must be private. With the page on Static Web Apps (Free) and the API on a Container App, sign-in would protect only the page: the API's address stays open. Linking the two behind one sign-in needs the paid Static Web Apps tier. The API's image also needs a registry, and Azure Container Registry costs about $5 a month.

## Decision
- The API container also serves the built web page, so the app has one address.
- Container Apps' built-in authentication (Microsoft sign-in) protects everything in it.
- The image lives in GitHub's container registry. The Container App pulls it with a read-only GitHub token stored in Key Vault.

## Consequences
- No extra cost, and no cross-origin setup between page and API.
- The Static Web App comes out of `infra/main.bicep` when the deploy work lands.
- The GitHub token has to be renewed if it expires.
