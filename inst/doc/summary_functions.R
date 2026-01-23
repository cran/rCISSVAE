## ----setup--------------------------------------------------------------------
library(tidyverse)
library(reticulate)
library(rCISSVAE)
library(kableExtra)
library(gtsummary)

data(df_missing)
data(clusters)

## Integer clusters must be passed in as a factor
cluster_summary(data = df_missing, factor(clusters$clusters), 
include = setdiff(names(df_missing), "index"), 
statistic = list(
  all_continuous() ~ "{mean} ({sd})",
  all_categorical() ~ "{n} / {N}\n ({p}%)"), 
  missing = "always")


## -----------------------------------------------------------------------------
cluster_heatmap(
  data = df_missing, 
  clusters = paste0("Cluster ", clusters$clusters), ## Adds 'Cluster' to the cluster label
  cols_ignore = "index", 
  observed_color = "#23013aff", ## A dark purple
  missing_color = "yellow")

## ----eval=FALSE---------------------------------------------------------------
# result = run_cissvae(
#   data = df_missing,
#   index_col = "index",
#   val_proportion = 0.1, ## pass a vector for different proportions by cluster
#   columns_ignore = c("Age", "Salary", "ZipCode10001", "ZipCode20002", "ZipCode30003"), ## If there are columns in addition to the index you want to ignore when selecting validation set, list them here. In this case, we ignore the 'demographic' columns because we do not want to remove data from them for validation purposes.
#   clusters = clusters$clusters, ## we have precomputed cluster labels so we pass them here
#   epochs = 5,
#   return_silhouettes = FALSE,
#   return_history = TRUE,  # Get detailed training history
#   verbose = FALSE,
#   return_model = TRUE, ## Allows for plotting model schematic
#   device = "cpu",  # Explicit device selection
#   layer_order_enc = c("unshared", "shared", "unshared"),
#   layer_order_dec = c("shared", "unshared", "shared"),
#   return_validation_dataset = TRUE
# )
# 
# cat(paste("Check necessary returns:", paste0(names(result), collapse = ", ")))

## ----checkres, echo=FALSE-----------------------------------------------------
result = readRDS(system.file("extdata", "demo_run.rds", package = "rCISSVAE"))
cat(paste("Check necessary returns:", paste0(names(result), collapse = ", ")))

## ----perf_by_clus-------------------------------------------------------------
performance_by_cluster(res = result, 
  group_col = NULL, 
  clusters = clusters$clusters,
  feature_cols = NULL, ## default, all numeric columns excluding group_col & cols_ignore
  by_group = FALSE,
  by_cluster = TRUE,
  cols_ignore =  c( "index", "Age", "Salary", "ZipCode10001", "ZipCode20002", "ZipCode30003") ## columns to not score
  )

