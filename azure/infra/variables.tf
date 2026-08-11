variable "prefix" {
  description = "Name prefix for every resource."
  type        = string
  default     = "gymcoach"
}

variable "location" {
  description = "Azure region (germanywestcentral matches the edgesense stack)."
  type        = string
  default     = "germanywestcentral"
}

variable "image_tag" {
  description = "Image tag the dashboard app starts from; CD rolls new tags."
  type        = string
  default     = "latest"
}

variable "tags" {
  type = map(string)
  default = {
    project = "ai-gym-coach"
    owner   = "mohamed-feki"
    env     = "demo"
  }
}
