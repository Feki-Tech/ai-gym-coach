# Azure demo deployment

Hosts **one thing**: the progress dashboard in `--demo` mode — synthetic,
deterministic training history. The product is local-first ([docs/INFRA.md
§2](../docs/INFRA.md)); real workout logs, profiles and video have **no path**
to this deployment. This exists to put "Azure + Terraform + OIDC CD" on the
repo with something visitors can actually click.

Not applied automatically. Run `terraform plan` and check the remaining trial
credit in the portal first (Cost Management — the consumption *API* returns
null costs on this subscription; only the portal shows the truth).

## Cost

| Piece | Cost |
|---|---|
| Dashboard app (scale-to-zero, 0.25 vCPU) | ~€0 idle, cents when visited |
| ACR Basic | ~€4/mo — the standing cost |
| Log Analytics (30d retention) | cents |

Default posture between demo periods: `terraform destroy`.

## First deploy

```bash
cd azure/infra
az login && az account set --subscription <sub-id>
terraform init && terraform validate

# 1) registry + environment + identity first, so the image has somewhere to go
terraform apply -target=azurerm_container_registry.this \
                -target=azurerm_container_app_environment.this \
                -target=azurerm_user_assigned_identity.apps \
                -target=azurerm_role_assignment.acr_pull

# 2) build + push (on this machine — ACR Tasks is disabled on the subscription)
../scripts/build-and-push.sh "$(terraform output -raw acr_name)"

# 3) everything else
terraform apply
terraform output dashboard_url
```

## CD (GitHub Actions, OIDC — one-time setup)

```bash
az ad app create --display-name gymcoach-cd
APP_ID=$(az ad app list --display-name gymcoach-cd --query "[0].appId" -o tsv)
az ad sp create --id "$APP_ID"

az ad app federated-credential create --id "$APP_ID" --parameters '{
  "name": "gymcoach-main",
  "issuer": "https://token.actions.githubusercontent.com",
  "subject": "repo:Feki-Tech/ai-gym-coach:ref:refs/heads/main",
  "audiences": ["api://AzureADTokenExchange"]
}'

SUB=$(az account show --query id -o tsv)
az role assignment create --assignee "$APP_ID" --role Contributor \
  --scope "/subscriptions/$SUB/resourceGroups/gymcoach-rg"
```

Then add repository **variables** (Settings → Secrets and variables → Actions
→ Variables): `AZURE_CLIENT_ID` (= `$APP_ID`), `AZURE_TENANT_ID`,
`AZURE_SUBSCRIPTION_ID`, `ACR_NAME`, `RESOURCE_GROUP` (= `gymcoach-rg`).
`azure-deploy.yml` stays inert until `AZURE_CLIENT_ID` exists, so merging
this before the setup is harmless.

## Known subscription quirks (inherited from the edgesense deployment)

- **ACR Tasks disabled** (`TasksOperationsNotAllowed`) — hence runner-side
  `docker build`, never `az acr build`.
- Several VM SKUs sit at **0 quota**; irrelevant here (no VMs), noted for
  anything this scaffold grows into.
- The **consumption API returns null costs** — check credit in the portal.
