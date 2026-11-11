// Showcase copy of infra/main.bicep from the private mahjong-companion repo. Here to be read, not run.

// Foundations for Mahjong Companion. Deploy into an existing resource group:
//   az deployment group create -g rg-mahjong-dev -f infra/main.bicep -p env=dev
// Postgres and Azure AI Speech are added in later phases to keep idle cost near zero.

targetScope = 'resourceGroup'

@allowed(['dev', 'prod'])
param env string = 'dev'
param location string = resourceGroup().location
@description('Container image for the API. The placeholder runs until CI pushes a real image.')
param apiImage string = 'mcr.microsoft.com/k8se/quickstart:latest'
@description('Monthly budget in USD for this resource group.')
param budgetUsd int = 30
@description('Email for budget alerts.')
param alertEmail string
@description('Client ID of the sign-in app registration. Empty leaves sign-in off (first deploy only).')
param authClientId string = ''
@description('Object ID of the one user allowed to sign in.')
param allowedUserObjectId string = ''
@description('GitHub user that owns the image in ghcr.io. Empty means a public image.')
param registryUser string = ''
@description('Turn on the strategist once anthropic-api-key, langfuse-public-key and langfuse-secret-key are in the vault.')
param enableStrategist bool = false
@description('Turn on listening: a free-tier (F0) Azure Speech resource for the phone microphone.')
param enableSpeech bool = false
@description('Turn on Jev for unclear table talk once typesafe-api-key is in the vault.')
param enableJev bool = false
@description('Langfuse region URL, e.g. https://cloud.langfuse.com or https://us.cloud.langfuse.com.')
param langfuseBaseUrl string = 'https://cloud.langfuse.com'

var name = 'mahjong-${env}'
var suffix = uniqueString(resourceGroup().id)
var useAuth = !empty(authClientId)
var useRegistry = !empty(registryUser)
// Vault secret name -> app setting, for the strategist (Phase 4).
var strategistSecrets = [
  { secret: 'anthropic-api-key', setting: 'ANTHROPIC_API_KEY' }
  { secret: 'langfuse-public-key', setting: 'LANGFUSE_PUBLIC_KEY' }
  { secret: 'langfuse-secret-key', setting: 'LANGFUSE_SECRET_KEY' }
]

resource logs 'Microsoft.OperationalInsights/workspaces@2023-09-01' = {
  name: 'log-${name}'
  location: location
  properties: {
    sku: { name: 'PerGB2018' }
    retentionInDays: 30
  }
}

resource appInsights 'Microsoft.Insights/components@2020-02-02' = {
  name: 'appi-${name}'
  location: location
  kind: 'web'
  properties: {
    Application_Type: 'web'
    WorkspaceResourceId: logs.id
  }
}

resource identity 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' = {
  name: 'id-${name}'
  location: location
}

// Holds the Anthropic, TypeSafe, Langfuse and Speech keys.
resource vault 'Microsoft.KeyVault/vaults@2023-07-01' = {
  name: 'kv-${take(name, 10)}-${take(suffix, 8)}'
  location: location
  properties: {
    tenantId: subscription().tenantId
    sku: { family: 'A', name: 'standard' }
    enableRbacAuthorization: true
    enableSoftDelete: true
    softDeleteRetentionInDays: 7
  }
}

// Private card data, game logs and photos. Never committed to git.
resource storage 'Microsoft.Storage/storageAccounts@2023-05-01' = {
  name: 'st${replace(name, '-', '')}${take(suffix, 8)}'
  location: location
  sku: { name: 'Standard_LRS' }
  kind: 'StorageV2'
  properties: {
    allowBlobPublicAccess: false
    minimumTlsVersion: 'TLS1_2'
  }
}

resource blobService 'Microsoft.Storage/storageAccounts/blobServices@2023-05-01' = {
  parent: storage
  name: 'default'
}

resource containers 'Microsoft.Storage/storageAccounts/blobServices/containers@2023-05-01' = [
  for c in ['cards', 'games', 'tile-photos', 'hand-photos']: {
    parent: blobService
    name: c
  }
]

// Built-in role IDs.
var keyVaultSecretsUser = '4633458b-17de-408a-b874-0445c86b69e6'
var storageBlobDataContributor = 'ba92f5b4-2d11-453d-a403-e96b0029c9fe'

resource vaultAccess 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  scope: vault
  name: guid(vault.id, identity.id, keyVaultSecretsUser)
  properties: {
    principalId: identity.properties.principalId
    principalType: 'ServicePrincipal'
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', keyVaultSecretsUser)
  }
}

resource storageAccess 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  scope: storage
  name: guid(storage.id, identity.id, storageBlobDataContributor)
  properties: {
    principalId: identity.properties.principalId
    principalType: 'ServicePrincipal'
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', storageBlobDataContributor)
  }
}

// Speech-to-text for the phone's microphone (Phase 6). F0 is the free tier: 5 audio hours a
// month, and it stops rather than bills beyond that. One F0 Speech resource per subscription.
resource speech 'Microsoft.CognitiveServices/accounts@2024-10-01' = if (enableSpeech) {
  name: 'speech-${name}-${take(suffix, 6)}'
  location: location
  kind: 'SpeechServices'
  sku: { name: 'F0' }
  properties: {
    publicNetworkAccess: 'Enabled'
  }
}

resource appEnv 'Microsoft.App/managedEnvironments@2024-03-01' = {
  name: 'cae-${name}'
  location: location
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: logs.properties.customerId
        sharedKey: logs.listKeys().primarySharedKey
      }
    }
  }
}

// Serves the API and the web page from one image (ADR 6). Its secrets are read from Key Vault.
resource api 'Microsoft.App/containerApps@2024-03-01' = {
  name: 'ca-${name}-api'
  location: location
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: { '${identity.id}': {} }
  }
  dependsOn: [vaultAccess]
  properties: {
    managedEnvironmentId: appEnv.id
    configuration: {
      ingress: {
        external: true
        targetPort: 8000
      }
      secrets: concat(
        useRegistry
          ? [{ name: 'ghcr-token', keyVaultUrl: '${vault.properties.vaultUri}secrets/ghcr-token', identity: identity.id }]
          : [],
        useAuth
          ? [{ name: 'auth-client-secret', keyVaultUrl: '${vault.properties.vaultUri}secrets/auth-client-secret', identity: identity.id }]
          : [],
        enableStrategist
          ? map(strategistSecrets, s => { name: s.secret, keyVaultUrl: '${vault.properties.vaultUri}secrets/${s.secret}', identity: identity.id })
          : [],
        enableSpeech ? [{ name: 'speech-key', value: speech!.listKeys().key1 }] : [],
        enableJev
          ? [{ name: 'typesafe-api-key', keyVaultUrl: '${vault.properties.vaultUri}secrets/typesafe-api-key', identity: identity.id }]
          : []
      )
      registries: useRegistry ? [{ server: 'ghcr.io', username: registryUser, passwordSecretRef: 'ghcr-token' }] : []
    }
    template: {
      containers: [
        {
          name: 'api'
          image: apiImage
          resources: { cpu: json('0.5'), memory: '1Gi' }
          env: concat(
            [
              { name: 'APPLICATIONINSIGHTS_CONNECTION_STRING', value: appInsights.properties.ConnectionString }
              { name: 'AZURE_CLIENT_ID', value: identity.properties.clientId }
              { name: 'KEY_VAULT_URI', value: vault.properties.vaultUri }
              { name: 'STORAGE_ACCOUNT_URL', value: storage.properties.primaryEndpoints.blob }
              { name: 'CARD_NAME', value: 'cards/2024.toml' }
            ],
            enableStrategist
              ? concat(
                  map(strategistSecrets, s => { name: s.setting, secretRef: s.secret }),
                  [{ name: 'LANGFUSE_BASE_URL', value: langfuseBaseUrl }]
                )
              : [],
            enableSpeech
              ? [
                  { name: 'AZURE_SPEECH_KEY', secretRef: 'speech-key' }
                  { name: 'AZURE_SPEECH_REGION', value: location }
                ]
              : [],
            enableJev ? [{ name: 'TYPESAFE_API_KEY', secretRef: 'typesafe-api-key' }] : []
          )
        }
      ]
      // Scale to zero between games.
      scale: { minReplicas: 0, maxReplicas: 1 }
    }
  }
}

// Built-in Microsoft sign-in in front of everything, admitting one user.
resource auth 'Microsoft.App/containerApps/authConfigs@2024-03-01' = if (useAuth) {
  parent: api
  name: 'current'
  properties: {
    platform: { enabled: true }
    globalValidation: {
      unauthenticatedClientAction: 'RedirectToLoginPage'
      redirectToProvider: 'azureactivedirectory'
    }
    identityProviders: {
      azureActiveDirectory: {
        enabled: true
        registration: {
          clientId: authClientId
          clientSecretSettingName: 'auth-client-secret'
          openIdIssuer: '${environment().authentication.loginEndpoint}${tenant().tenantId}/v2.0'
        }
        validation: {
          defaultAuthorizationPolicy: {
            allowedPrincipals: { identities: [allowedUserObjectId] }
          }
        }
      }
    }
  }
}

resource budget 'Microsoft.Consumption/budgets@2023-11-01' = {
  name: 'budget-${name}'
  properties: {
    category: 'Cost'
    amount: budgetUsd
    timeGrain: 'Monthly'
    timePeriod: { startDate: '2026-11-01' }
    notifications: {
      actual80: {
        enabled: true
        operator: 'GreaterThan'
        threshold: 80
        contactEmails: [alertEmail]
      }
    }
  }
}

output appUrl string = 'https://${api.properties.configuration.ingress.fqdn}'
output containerAppName string = api.name
output keyVaultName string = vault.name
output storageAccountName string = storage.name
