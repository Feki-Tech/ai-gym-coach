# Azure demo deployment

Hosts **one thing**: the progress dashboard in `--demo` mode — synthetic,
deterministic training history. The product is local-first ([docs/INFRA.md
§2](../docs/INFRA.md)); real workout logs, profiles and video have **no path**
to this deployment. This exists to put "Azure + Terraform + OIDC CD" on the
repo with something visitors can actually click.

Nothing applies automatically. `plan`, `apply` and `destroy` are manual
dispatches of the `azure-terraform.yml` workflow — no local Terraform install
needed. Check the remaining trial credit in the portal first (Cost
Management — the consumption *API* returns null costs on this subscription;
only the portal shows the truth).

## Cost

| Piece | Cost |
|---|---|
| Dashboard app (scale-to-zero, 0.25 vCPU) | ~€0 idle, cents when visited |
| ACR Basic | ~€4/mo — the standing cost |
| Log Analytics (30d retention) | cents |
| Terraform state storage (LRS) | cents; survives destroy |

Default posture between demo periods: dispatch `azure-terraform.yml` with
`action=destroy`. The resource group, role assignments and state storage
survive (they are owned by the setup script, not Terraform — see the
`main.tf` header), so the next `apply` needs no re-setup.

## One-time setup

Run on any machine with `az` and an Owner-capable login:

```bash
az login && az account set --subscription <sub-id>
./azure/scripts/setup-azure-cd.sh
```

The script is idempotent. It creates the resource group, the Terraform state
storage account, the `gymcoach-cd` app registration with a federated
credential for `main` (OIDC — no stored secrets), and the role assignments —
including `Role Based Access Control Administrator` scoped to the RG, because
Terraform creates the AcrPull assignment and Contributor alone can't do that.
It ends by printing the repository **variables** to set (Settings → Secrets
and variables → Actions → Variables): `AZURE_CLIENT_ID`, `AZURE_TENANT_ID`,
`AZURE_SUBSCRIPTION_ID`, `RESOURCE_GROUP`, `TFSTATE_STORAGE_ACCOUNT`, and —
after the first apply — `ACR_NAME`.

Both workflows stay inert until their variables exist, so merging this before
the setup is harmless.

## Deploying

Actions → **Terraform (Azure demo infra)** → Run workflow:

- `plan` — review what would change. Run this first, always.
- `apply` — first run bootstraps in two phases (registry/environment/identity,
  then build+push the image on the runner, then the app) because the image
  must exist before the container app can pull it. The run summary prints the
  dashboard URL and the `ACR_NAME` to set for CD.
- `destroy` — tear down between demo periods.

After `ACR_NAME` is set, every push to `main` rolls the dashboard via
`azure-deploy.yml` (build on the runner → push to ACR → `az containerapp
update`), gated on the selftests.

## Validation

`azure-validate.yml` runs `terraform fmt`/`init -backend=false`/`validate` on
every change under `azure/**` — the scaffold can no longer merge unvalidated.

## Running Terraform locally (optional)

```bash
cd azure/infra
az login && az account set --subscription <sub-id>
export ARM_SUBSCRIPTION_ID=$(az account show --query id -o tsv) # azurerm 4.x requires this
terraform init \
  -backend-config="resource_group_name=gymcoach-rg" \
  -backend-config="storage_account_name=<from setup script>" \
  -backend-config="container_name=tfstate" \
  -backend-config="key=gymcoach-demo.tfstate" \
  -backend-config="use_azuread_auth=true"
terraform plan
```

State is shared with CI via the azurerm backend, so local and workflow runs
see the same world. The setup script grants your user the data-plane role the
backend needs.

## Known subscription quirks (inherited from the edgesense deployment)

- **ACR Tasks disabled** (`TasksOperationsNotAllowed`) — hence runner-side
  `docker build`, never `az acr build`.
- Several VM SKUs sit at **0 quota**; irrelevant here (no VMs), noted for
  anything this scaffold grows into.
- The **consumption API returns null costs** — check credit in the portal.
- **azurerm 4.x needs an explicit subscription** (`ARM_SUBSCRIPTION_ID` or
  provider `subscription_id`); `az account set` alone is not enough. The
  workflow sets it from repository variables; locally, export it.
