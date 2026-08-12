output "resource_group" {
  value = data.azurerm_resource_group.this.name
}

output "acr_name" {
  description = "Registry name (az acr login --name <this>)."
  value       = azurerm_container_registry.this.name
}

output "acr_login_server" {
  value = azurerm_container_registry.this.login_server
}

output "dashboard_url" {
  description = "Public URL of the demo dashboard (synthetic data)."
  value       = "https://${azurerm_container_app.dashboard.ingress[0].fqdn}"
}
