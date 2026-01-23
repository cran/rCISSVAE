## ----setup, include=FALSE-----------------------------------------------------
knitr::opts_chunk$set(
  echo = TRUE,
  warning = FALSE,
  error = FALSE,
  fig.width = 7,
  fig.height = 5,
  dpi = 600,
  collapse = TRUE,
  comment = "#>"
)

## ----eval=FALSE---------------------------------------------------------------
# install.packages("reticulate")

## ----eval=FALSE---------------------------------------------------------------
# install.packages("reticulate")
# library(reticulate)

## ----eval=FALSE---------------------------------------------------------------
# # Find Python installations on your system
# reticulate::py_discover_config()
# 
# # Or check for a specific version
# reticulate::virtualenv_starter("3.10")

## ----eval=FALSE---------------------------------------------------------------
# # Install Python 3.10 (recommended)
# reticulate::install_python(version = "3.10")
# 
# # Or install Python 3.11
# reticulate::install_python(version = "3.11")
# 
# # Check available versions after installation
# reticulate::py_versions()
# 
# # Verify the installation worked
# reticulate::virtualenv_starter("3.10")

## ----eval=FALSE---------------------------------------------------------------
# # Create virtual environment in default location (~/.virtualenvs/)
# reticulate::virtualenv_create(
#   envname = "cissvae_env",
#   python = NULL,  # Use system default Python
#   packages = c("pip", "setuptools", "wheel")
# )
# 
# # Alternative: Create in a specific directory
# reticulate::virtualenv_create(
#   envname = "./my_venvs/cissvae_env",  # Relative path
#   python = "/usr/bin/python3.10",     # Specific Python version
#   packages = c("pip", "setuptools", "wheel")
# )

## ----eval=FALSE---------------------------------------------------------------
# # Activate environment (default location)
# reticulate::use_virtualenv("cissvae_env", required = TRUE)
# 
# # Or activate from specific path
# reticulate::use_virtualenv("./my_venvs/cissvae_env", required = TRUE)

## ----eval=FALSE---------------------------------------------------------------
# # Install core dependencies first
# reticulate::virtualenv_install(
#   envname = "cissvae_env",
#   packages = c(
#     "numpy",
#     "pandas",
#     "torch",
#     "scikit-learn",
#     "optuna",
#     "rich",
#     "matplotlib",
#     "leiden-alg",
#     "python-igraph"
#   )
# )
# 
# # Install CISS-VAE from PyPI
# reticulate::virtualenv_install(
#   envname = "cissvae_env",
#   packages = "ciss-vae"
# )

## ----eval=FALSE---------------------------------------------------------------
# # Check if CISS-VAE installed correctly
# reticulate::py_run_string("import ciss_vae; print('CISS-VAE version:', ciss_vae.__version__)")
# 
# # List all installed packages
# reticulate::virtualenv_install(envname = "cissvae_env", packages = character(0))

## ----eval=FALSE---------------------------------------------------------------
# library(reticulate)
# 
# # Point to your manually created environment
# reticulate::use_virtualenv("/path/to/your/project/cissvae_env", required = TRUE)
# 
# # On Windows, the path might look like:
# # reticulate::use_virtualenv("C:/Users/YourName/project/cissvae_env", required = TRUE)
# 
# # Verify connection
# reticulate::py_config()

## ----eval=FALSE---------------------------------------------------------------
# # Create conda environment from R
# reticulate::conda_create(
#   envname = "cissvae_env",
#   python_version = "3.10",
#   packages = c("numpy", "pandas", "matplotlib", "scikit-learn")
# )
# 
# # Activate and install additional packages
# reticulate::use_condaenv("cissvae_env", required = TRUE)
# reticulate::conda_install("cissvae_env", c("torch", "optuna", "rich", "hdbscan", "ciss-vae"))

## ----eval=FALSE---------------------------------------------------------------
# reticulate::virtualenv_create(
#   envname = "cissvae_env",
#   python = "/usr/local/bin/python3.10"  # Full path to Python
# )

## ----eval=FALSE---------------------------------------------------------------
# packages <- c("numpy", "pandas", "torch", "ciss-vae")
# for (pkg in packages) {
#   cat("Installing", pkg, "...\n")
#   reticulate::virtualenv_install("cissvae_env", pkg)
# }

## ----eval=FALSE---------------------------------------------------------------
# # Check Python configuration
# reticulate::py_config()
# 
# # Test import manually
# reticulate::py_run_string("
# try:
#     import ciss_vae
#     print('Success: ciss_vae imported')
#     print('Version:', ciss_vae.__version__)
# except ImportError as e:
#     print('Error importing ciss_vae:', e)
# ")

## ----eval=FALSE---------------------------------------------------------------
# # Add this to the top of your R scripts
# library(reticulate)
# reticulate::use_virtualenv("cissvae_env", required = TRUE)
# 
# # Or add to your .Rprofile for automatic loading
# cat('reticulate::use_virtualenv("cissvae_env", required = TRUE)\n',
#     file = "~/.Rprofile", append = TRUE)

