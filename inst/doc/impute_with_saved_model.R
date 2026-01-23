## ----eval=FALSE---------------------------------------------------------------
# library(reticulate)
# res = run_cissvae(data)
# 
# # Import Python torch module
# torch <- import("torch")
# 
# # Assume `model` is a Python object already available in the R session
# # (e.g., created earlier via reticulate)
# torch$save(res$model, "trained_vae.pt")

## ----eval=FALSE---------------------------------------------------------------
# library(rCISSVAE)
# library(reticulate)
# 
# ## Activate your virtual environment
# reticulate::use_virtualenv("cissvae_environment", required = TRUE)
# 
# ## Use CISSVAE to load the model
# # Import the module so the class is registered (required for full-model loading)
# import("ciss_vae.classes.vae")
# 
# # Load full model object
# model <- torch$load("trained_vae.pt", map_location = "cpu", weights_only = FALSE)
# model$eval()
# 
# # Optional: get imputed dataset
# helpers <- import("ciss_vae.utils.helpers")
# DataLoader <- import("torch.utils.data")$DataLoader
# 
# ## Convert your dataset to python ClusterDataset object
#   CD_mod   <- reticulate::import("ciss_vae.classes.cluster_dataset",
#   convert = FALSE)$ClusterDataset
#   np       <- reticulate::import("numpy", convert = FALSE)
#   pd       <- reticulate::import("pandas", convert = FALSE)
# 
# ## make sure NAs are python compatible
#   data[is.na(data)] <- NaN
# 
# ## Convert data and clusters into python objects
#   data_py <- pd$DataFrame(data = data, dtype = "float64")
#   clusters_py <- np$array(as.integer(clusters), dtype = "int64")
# 
# ## Make ClusterDataset and DataLoader
#   dataset = CD_mod(
#     data = data_py,
#     cluster_labels = clusters_py)
#   data_loader <- DataLoader(dataset, batch_size = 4000L)
# 
# ## Get Imputed Dataset
# imputed_df <- helpers$get_imputed_df(model, data_loader)
# 

