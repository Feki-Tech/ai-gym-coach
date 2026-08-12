terraform {
  required_version = ">= 1.6"
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 4.0"
    }
  }

  # State lives in Azure Storage so plan/apply can run from CI (or any
  # machine) without hand-carrying local state. Values are supplied at init
  # time (-backend-config=…); setup-azure-cd.sh creates the account and
  # prints them. Validation-only runs use `terraform init -backend=false`.
  backend "azurerm" {}
}

provider "azurerm" {
  features {}
}
