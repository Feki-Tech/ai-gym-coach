#!/usr/bin/env bash
# One-time setup for the Azure demo deployment + OIDC CD. Idempotent — safe
# to re-run. Run with an account that can create app registrations and role
# assignments (subscription Owner).
#
#   ./setup-azure-cd.sh [github-repo]      default: Feki-Tech/ai-gym-coach
#
# Creates, outside of Terraform (see main.tf header for why):
#   - gymcoach-rg              the RG all demo resources live in
#   - a state storage account  (in gymcoach-rg; survives terraform destroy
#                               because the RG itself is not TF-managed)
#   - gymcoach-cd              Entra app + SP, federated for GitHub OIDC
#   - role assignments for the SP:
#       Contributor                              on gymcoach-rg
#       Role Based Access Control Administrator  on gymcoach-rg
#         (terraform creates the AcrPull assignment; Contributor alone
#          cannot create role assignments)
#       Storage Blob Data Contributor            on the state account
#   - Storage Blob Data Contributor for YOU on the state account, so local
#     terraform runs work too (export ARM_SUBSCRIPTION_ID first — azurerm 4.x
#     refuses to infer the subscription from az login alone)
#
# Finishes by printing the GitHub repository variables to set.
set -euo pipefail

REPO="${1:-Feki-Tech/ai-gym-coach}"
PREFIX=gymcoach
LOCATION=germanywestcentral
TAGS=(project=ai-gym-coach owner=mohamed-feki env=demo)

SUB=$(az account show --query id -o tsv)
TENANT=$(az account show --query tenantId -o tsv)
RG="${PREFIX}-rg"

echo "==> resource group $RG"
az group create -n "$RG" -l "$LOCATION" --tags "${TAGS[@]}" -o none

echo "==> state storage account"
SA=$(az storage account list -g "$RG" \
  --query "[?starts_with(name, '${PREFIX}tfstate')].name | [0]" -o tsv)
if [ -z "$SA" ]; then
  SA="${PREFIX}tfstate$(head -c4 /dev/urandom | od -An -tx1 | tr -d ' \n')"
  az storage account create -n "$SA" -g "$RG" -l "$LOCATION" \
    --sku Standard_LRS --kind StorageV2 --min-tls-version TLS1_2 \
    --allow-blob-public-access false --tags "${TAGS[@]}" -o none
fi
az storage container create --account-name "$SA" -n tfstate -o none
echo "    $SA"

echo "==> app registration $PREFIX-cd"
APP_ID=$(az ad app list --display-name "$PREFIX-cd" --query "[0].appId" -o tsv)
if [ -z "$APP_ID" ]; then
  APP_ID=$(az ad app create --display-name "$PREFIX-cd" --query appId -o tsv)
fi
az ad sp create --id "$APP_ID" -o none 2>/dev/null || true
SP_OID=$(az ad sp show --id "$APP_ID" --query id -o tsv)

if ! az ad app federated-credential list --id "$APP_ID" \
     --query "[?name=='${PREFIX}-main']" -o tsv | grep -q .; then
  az ad app federated-credential create --id "$APP_ID" --parameters "{
    \"name\": \"${PREFIX}-main\",
    \"issuer\": \"https://token.actions.githubusercontent.com\",
    \"subject\": \"repo:${REPO}:ref:refs/heads/main\",
    \"audiences\": [\"api://AzureADTokenExchange\"]
  }" -o none
fi

echo "==> role assignments"
assign() { # role scope principal-oid
  az role assignment create --role "$1" --scope "$2" \
    --assignee-object-id "$3" --assignee-principal-type "${4:-ServicePrincipal}" \
    -o none 2>/dev/null || true # already-exists is fine
}
RG_ID="/subscriptions/$SUB/resourceGroups/$RG"
SA_ID="$RG_ID/providers/Microsoft.Storage/storageAccounts/$SA"
assign "Contributor" "$RG_ID" "$SP_OID"
assign "Role Based Access Control Administrator" "$RG_ID" "$SP_OID"
assign "Storage Blob Data Contributor" "$SA_ID" "$SP_OID"
ME=$(az ad signed-in-user show --query id -o tsv 2>/dev/null || true)
[ -n "$ME" ] && assign "Storage Blob Data Contributor" "$SA_ID" "$ME" User

cat <<EOF

Done. Set these repository VARIABLES on $REPO
(Settings -> Secrets and variables -> Actions -> Variables):

  AZURE_CLIENT_ID          $APP_ID
  AZURE_TENANT_ID          $TENANT
  AZURE_SUBSCRIPTION_ID    $SUB
  RESOURCE_GROUP           $RG
  TFSTATE_STORAGE_ACCOUNT  $SA
  ACR_NAME                 (after the first apply — azure-terraform.yml
                            prints it in the run summary)

Then: Actions -> "Terraform (Azure demo infra)" -> Run workflow -> apply.
EOF
