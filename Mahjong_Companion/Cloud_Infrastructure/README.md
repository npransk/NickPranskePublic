# Cloud infrastructure

The source behind [Part 3 of the write-up](../README.md#part-3-cloud-infrastructure). These are copies from the private repository, there to be read.

| File | What it does | Worth a look |
| --- | --- | --- |
| [`main.bicep`](main.bicep) | Every Azure resource, in one file: Container App, managed identity, Key Vault, storage, Speech, logging, sign-in and a budget. | The `secrets` block: each app secret is a Key Vault reference read by the managed identity. The `auth` resource: sign-in in front of everything, one user allowed. `scale`: zero to one replica. |
| [`Dockerfile`](Dockerfile) | Builds one image in two stages: Node builds the web page, then a slim Python image serves the API and that page. | One image means one address and one sign-in for both. |
| [`ci.yml`](ci.yml) | Runs on every pull request: lint, format check, tests, web build, Bicep build. | |
| [`deploy.yml`](deploy.yml) | Runs on every merge to `main`: build, push to GitHub's registry, sign in to Azure with OIDC, update the Container App. | `id-token: write` and `vars.*` instead of `secrets.*`: no Azure secret is stored in GitHub. The `concurrency` group runs one deploy at a time. |
| [`store.py`](store.py) | Where games, cards and photos are kept: a local folder on my laptop, blob storage in Azure. | `BlobStore` signs in with `DefaultAzureCredential`, so there is no connection string anywhere. |
| [`speech.py`](speech.py) | Trades the server's Speech key for a 10-minute token for the phone. | The key never leaves the server, and audio never reaches it. |

## Why these files are safe to publish

- **No secret values.** The Bicep file names secrets (`anthropic-api-key`, `langfuse-secret-key` and so on) and points at the vault that holds them. The values exist only in Key Vault.
- **No account identifiers.** Subscription, tenant and client IDs are passed in when deploying or held as repository variables in the private repository. The two GUIDs in `main.bicep` are Azure's public IDs for the built-in roles "Key Vault Secrets User" and "Storage Blob Data Contributor", the same in every subscription.
- **The workflows can't run from here.** GitHub only runs workflows from `.github/workflows`, and these sit outside it. Even moved there, the deploy could not sign in: Azure trusts tokens issued for the private repository and its `dev` environment, and no other.
