# AI Gym Coach — Azure demo deployment (docs/INFRA.md §5).
#
# One container app: the progress dashboard in --demo mode, serving SYNTHETIC
# history only. The product is local-first — no user data has any path to
# this deployment (INFRA.md §2). Scale-to-zero keeps idle compute at €0; the
# standing cost is ACR Basic (~€4/mo) + Log Analytics cents.
#
# Plan/apply normally runs in CI (azure-terraform.yml, OIDC) — the workflow
# handles the two-phase bootstrap (registry first, then image, then the app).
# Tear down between demo periods: dispatch the workflow with action=destroy
# (the default posture).
#
# The resource group is NOT managed here: setup-azure-cd.sh creates it, and
# the CD principal's Contributor role is scoped to it. If Terraform owned the
# RG, `terraform destroy` would delete the very scope that role assignment
# lives on and CD would break until someone re-ran the one-time setup.

data "azurerm_resource_group" "this" {
  name = "${var.prefix}-rg"
}

resource "azurerm_log_analytics_workspace" "this" {
  name                = "${var.prefix}-logs"
  resource_group_name = data.azurerm_resource_group.this.name
  location            = data.azurerm_resource_group.this.location
  sku                 = "PerGB2018"
  retention_in_days   = 30
  tags                = var.tags
}

resource "azurerm_container_app_environment" "this" {
  name                       = "${var.prefix}-env"
  resource_group_name        = data.azurerm_resource_group.this.name
  location                   = data.azurerm_resource_group.this.location
  log_analytics_workspace_id = azurerm_log_analytics_workspace.this.id
  tags                       = var.tags
}

# Registry name must be globally unique and alphanumeric-only.
resource "azurerm_container_registry" "this" {
  name                = "${var.prefix}acr${substr(md5(data.azurerm_resource_group.this.id), 0, 6)}"
  resource_group_name = data.azurerm_resource_group.this.name
  location            = data.azurerm_resource_group.this.location
  sku                 = "Basic"
  admin_enabled       = false # pulls go through the managed identity, not passwords
  tags                = var.tags
}

resource "azurerm_user_assigned_identity" "apps" {
  name                = "${var.prefix}-apps-id"
  resource_group_name = data.azurerm_resource_group.this.name
  location            = data.azurerm_resource_group.this.location
  tags                = var.tags
}

resource "azurerm_role_assignment" "acr_pull" {
  scope                = azurerm_container_registry.this.id
  role_definition_name = "AcrPull"
  principal_id         = azurerm_user_assigned_identity.apps.principal_id
}

resource "azurerm_container_app" "dashboard" {
  name                         = "${var.prefix}-dashboard"
  resource_group_name          = data.azurerm_resource_group.this.name
  container_app_environment_id = azurerm_container_app_environment.this.id
  revision_mode                = "Single"
  tags                         = var.tags

  identity {
    type         = "UserAssigned"
    identity_ids = [azurerm_user_assigned_identity.apps.id]
  }

  registry {
    server   = azurerm_container_registry.this.login_server
    identity = azurerm_user_assigned_identity.apps.id
  }

  ingress {
    external_enabled = true
    target_port      = 7788
    transport        = "http"
    traffic_weight {
      latest_revision = true
      percentage      = 100
    }
  }

  template {
    min_replicas = 0 # scale-to-zero: no cost when nobody is looking
    max_replicas = 1

    container {
      name   = "dashboard"
      image  = "${azurerm_container_registry.this.login_server}/ai-gym-coach:${var.image_tag}"
      cpu    = 0.25
      memory = "0.5Gi"

      # The image's entrypoint is pose_coach.py; run the dashboard instead,
      # in --demo mode (synthetic history, deterministic seed — never a log).
      command = ["python", "coach_dashboard.py"]
      args    = ["--demo", "--host", "0.0.0.0", "--no-browser"]
    }

    http_scale_rule {
      name                = "http-requests"
      concurrent_requests = 20
    }
  }

  # CD rolls the image tag with `az containerapp update`, so the live tag
  # drifts from var.image_tag in state — that is expected, ignore it.
  lifecycle {
    ignore_changes = [template[0].container[0].image]
  }
}
