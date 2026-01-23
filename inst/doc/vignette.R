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

library(kableExtra)

## ----eval=FALSE---------------------------------------------------------------
# install.packages("remotes")
# # or
# install.packages("devtools")

## ----eval=FALSE---------------------------------------------------------------
# remotes::install_github("CISS-VAE/rCISS-VAE")
# # or
# devtools::install_github("CISS-VAE/rCISS-VAE")

## ----eval=FALSE---------------------------------------------------------------
# reticulate::use_virtualenv("./.venv", required = TRUE)

## ----eval=FALSE---------------------------------------------------------------
# reticulate::use_condaenv("myenv", required = TRUE)

## ----eval=FALSE---------------------------------------------------------------
# create_cissvae_env(
#   envname = "./cissvae_environment", ## name of environment
#   path = NULL, ## add path to wherever you want virtual environment to be
#   install_python = FALSE, ## set to TRUE if you want create_cisssvae_env to install python for you
#   python_version = "3.10" ## set to whatever version you want >=3.10. Python 3.10 or 3.11 recommended
# )

## ----eval=FALSE---------------------------------------------------------------
# reticulate::use_virtualenv("./cissvae_environment", required = TRUE)
# # If you used a non-default environment name then,
# # reticulate::use_virtualenv("./your_environment_name", required = TRUE)

## ----check_devices, eval = FALSE----------------------------------------------
# rCISSVAE::check_devices()

## ----show_dataset, echo=FALSE-------------------------------------------------
suppressPackageStartupMessages({
  library(tidyverse)
  library(reticulate)
  library(rCISSVAE)
  library(gtsummary)
  library(kableExtra)
})


data(df_missing)
data(clusters)

tbl_summary(df_missing, include = -"index") %>%
   kable() %>%
  kable_styling(font_size = 12)

## ----eval=FALSE, message=FALSE, warning=FALSE---------------------------------
#   library(tidyverse)
#   library(reticulate)
#   library(rCISSVAE)
#   library(gtsummary)
# 
# 
# ## Set correct virtualenv
# reticulate::use_virtualenv("./cissvae_environment", required = TRUE)
# 
# ## Load the data
# data(df_missing)
# data(clusters) ## actual cluster labels in clusters$clusters (other column is index)
# 
# cluster_summary(
#   data = df_missing,
#   clusters = clusters$clusters,
#   include =setdiff(names(df_missing), "index"),
#   statistic = list(
#     all_continuous() ~ "{mean} ({sd})",
#     all_categorical() ~ "{n} / {N}\n ({p}%)"),
#   missing = "always")

## ----showruncissvae2, echo=FALSE----------------------------------------------
suppressPackageStartupMessages({
  library(tidyverse)
  library(rCISSVAE)
  library(gtsummary)
})

data(df_missing)
data(clusters)

cluster_summary(
  data = df_missing, 
  clusters = clusters$clusters, 
  include =setdiff(names(df_missing), "index"),
  statistic = list(
    all_continuous() ~ "{mean} ({sd})",
    all_categorical() ~ "{n} / {N}\n ({p}%)"), 
  missing = "always")


## ----eval=FALSE, message=FALSE, warning=FALSE---------------------------------
# ## Run the imputation model.
# dat = run_cissvae(
#   data = df_missing,
#   index_col = "index",
#   val_proportion = 0.1, ## pass a vector for different proportions by cluster
#   columns_ignore = c("Age", "Salary", "ZipCode10001", "ZipCode20002", "ZipCode30003"), ## If there are columns in addition to the index you want to ignore when selecting validation set, list them here. In this case, we ignore the 'demographic' columns because we do not want to remove data from them for validation purposes.
#   clusters = clusters$clusters, ## we have precomputed cluster labels so we pass them here
#   epochs = 500,
#   return_silhouettes = FALSE,
#   return_history = TRUE,  # Get detailed training history
#   verbose = FALSE,
#   return_model = TRUE, ## Allows for plotting model schematic
#   device = "cpu",  # Explicit device selection
#   layer_order_enc = c("unshared", "shared", "unshared"),
#   layer_order_dec = c("shared", "unshared", "shared")
# )
# 
# ## Retrieve results
# imputed_df <- dat$imputed
# silhouette <- dat$silhouettes
# training_history <- dat$history  # Detailed training progress
# 
# ## Plot training progress
# if (!is.null(training_history)) {
#   plot(training_history$epoch, training_history$loss,
#        type = "l", main = "Training Loss Over Time",
#        xlab = "Epoch", ylab = "Loss")
# }
# 
# plot_vae_architecture(model = dat$model, save_path = "test_plot_arch.png")
# 
# print(head(dat$imputed_dataset))

## ----showres1, echo=FALSE-----------------------------------------------------
dat <- readRDS(system.file("extdata", "demo_run.rds", package = "rCISSVAE"))
print(head(dat$imputed_dataset))

## ----clusfeatmissing, eval = FALSE--------------------------------------------
# library(rCISSVAE)
# 
# data(df_missing)
# 
# 
# cluster_result <- cluster_on_missing(
#   data = df_missing,
#   cols_ignore =  c("index", "Age", "Salary", "ZipCode10001", "ZipCode20002", "ZipCode30003"),
#   n_clusters = 4,  # Use KMeans with 4 clusters
#   seed = 42
# )
# 
# cluster_summary(df_missing, factor(cluster_result$clusters), include = setdiff(names(df_missing), "index"),
# statistic = list(
#   gtsummary::all_continuous() ~ "{mean} ({sd})",
#   gtsummary::all_categorical() ~ "{n} / {N}\n ({p}%)"),
#   missing = "always")
# 
# cat(paste("Clustering quality (silhouette):", round(cluster_result$silhouette, 3)))
# 
# 
# result <- run_cissvae(
#   data = df_missing,
#   index_col = "index",
#   clusters = cluster_result$clusters,
#   return_history = TRUE,
#   verbose = FALSE,
#   device = "cpu"
# )

## ----showres2, echo=FALSE-----------------------------------------------------
library(gtsummary)

cluster_summary(df_missing, factor(clusters$clusters), include = setdiff(names(df_missing), "index"), 
statistic = list(
  all_continuous() ~ "{mean} ({sd})",
  all_categorical() ~ "{n} / {N}\n ({p}%)"), 
  missing = "always")  

cat("Clustering quality (silhouette):  0.135")

## ----useprecompmissingprop, eval=FALSE, echo = TRUE---------------------------
# 
# ## Standardize df_missing column names to feature_timepoint format
# colnames(df_missing) = c('index', 'Age', 'Salary', 'ZipCode10001', 'ZipCode20002', 'ZipCode30003', 'Y1_1', 'Y1_2', 'Y1_3', 'Y1_4', 'Y1_5', 'Y2_1', 'Y2_2', 'Y2_3', 'Y2_4', 'Y2_5', 'Y3_1', 'Y3_2', 'Y3_3', 'Y3_4', 'Y3_5', 'Y4_1', 'Y4_2', 'Y4_3', 'Y4_4', 'Y4_5', 'Y5_1', 'Y5_2', 'Y5_3', 'Y5_4', 'Y5_5')
# 
# # Create and examine missingness proportion matrix
# prop_matrix <- create_missingness_prop_matrix(df_missing,
# index_col = "index",
# cols_ignore = c('Age', 'Salary', 'ZipCode10001', 'ZipCode20002', 'ZipCode30003'),
# repeat_feature_names = c("Y1", "Y2", "Y3", "Y4", "Y5"))
# 
# 
# cat("Missingness proportion matrix dimensions:\n")
# cat(dim(prop_matrix), "\n")
# cat("Sample of proportion matrix:\n")
# print(head(prop_matrix[, 1:5]))
# 
# # Use proportion matrix with scaling for better clustering
# advanced_result <- run_cissvae(
#   data = df_missing,
#   index_col = "index",
#   clusters = NULL,  # Let function cluster using prop_matrix
#   columns_ignore = c('Age', 'Salary', 'ZipCode10001', 'ZipCode20002', 'ZipCode30003'),
#   missingness_proportion_matrix = prop_matrix,
#   scale_features = TRUE,  # Standardize features before clustering
#   n_clusters = 4,
#   leiden_resolution = 0.1,
#   epochs = 5,
#   return_history = TRUE,
#   return_silhouettes = TRUE,
#   device = "cpu",
#   verbose = FALSE,
#   return_clusters = TRUE
# )
# 
# print("Clustering quality:")
# print(paste("Silhouette score:", round(advanced_result$silhouette_width, 3)))
# 
# ## Plotting imputation loss by epoch
# 
# ggplot2::ggplot(data = advanced_result$training_history, aes(x = epoch, y = imputation_error)) + geom_point() + labs(y = "Imputation Loss", x = "Epoch") +
#   theme_classic()

## ----showres3, eval=TRUE, echo=FALSE------------------------------------------
colnames(df_missing) = c('index', 'Age', 'Salary', 'ZipCode10001', 'ZipCode20002', 'ZipCode30003', 'Y1_1', 'Y1_2', 'Y1_3', 'Y1_4', 'Y1_5', 'Y2_1', 'Y2_2', 'Y2_3', 'Y2_4', 'Y2_5', 'Y3_1', 'Y3_2', 'Y3_3', 'Y3_4', 'Y3_5', 'Y4_1', 'Y4_2', 'Y4_3', 'Y4_4', 'Y4_5', 'Y5_1', 'Y5_2', 'Y5_3', 'Y5_4', 'Y5_5')

# Create and examine missingness proportion matrix
prop_matrix <- create_missingness_prop_matrix(df_missing, 
index_col = "index", 
cols_ignore = c('Age', 'Salary', 'ZipCode10001', 'ZipCode20002', 'ZipCode30003'),
repeat_feature_names = c("Y1", "Y2", "Y3", "Y4", "Y5"))


cat("Missingness proportion matrix dimensions:\n")
cat(dim(prop_matrix), "\n")
cat("Sample of proportion matrix:\n")
print(head(prop_matrix[, 1:5]))



## ----showres4, echo=FALSE-----------------------------------------------------
advanced_result = readRDS(system.file("extdata", "missing_prop.rds", package = "rCISSVAE"))

# ar$silhouette_width
# [1] 0.6557173

cat("Clustering quality:\n")
cat(paste("Silhouette score:", round(0.6557173, 3)), "\n")

ggplot2::ggplot(data = advanced_result, aes(x = epoch, y = imputation_error)) + geom_point() + labs(y = "Imputation Loss", x = "Epoch") + 
  theme_classic()


## ----eval=FALSE---------------------------------------------------------------
# library(tidyverse)
# library(reticulate)
# library(rCISSVAE)
# reticulate::use_virtualenv("./cissvae_environment", required = TRUE)
# 
# data(df_missing)
# data(clusters)
# 
# aut <- autotune_cissvae(
#   data = df_missing,
#   index_col = "index",
#   clusters = clusters$clusters,
#   save_model_path = NULL,
#   save_search_space_path = NULL,
#   n_trials = 3, ## Using low number of trials for demo
#   study_name = "comprehensive_vae_autotune",
#   device_preference = "cpu",
#   show_progress = FALSE,  # Set true for Rich progress bars with training visualization
#   optuna_dashboard_db = "sqlite:///optuna_study.db",  # Save results to database
#   load_if_exists = FALSE, ## Set true to load and continue study if it exists
#   seed = 42,
#   verbose = FALSE,
# 
#   # Search strategy options
#   constant_layer_size = FALSE,     # Allow different sizes per layer
#   evaluate_all_orders = FALSE,     # Sample layer arrangements efficiently
#   max_exhaustive_orders = 100,     # Limit for exhaustive search
# 
#   ## Hyperparameter search space
#   num_hidden_layers = c(2, 5),     # Try 2-5 hidden layers
#   hidden_dims = c(64, 512),        # Layer sizes from 64 to 512
#   latent_dim = c(10, 100),         # Latent dimension range
#   latent_shared = c(TRUE, FALSE),
#   output_shared = c(TRUE, FALSE),
#   lr = 0.01,  # Learning rate range
#   decay_factor = 0.99,
#   beta = 0.01,  # KL weight range
#   num_epochs = 500,                # Fixed epochs for demo
#   batch_size = c(1000, 4000),     # Batch size options
#   num_shared_encode = c(0, 1, 2, 3),
#   num_shared_decode = c(0, 1, 2, 3),
# 
#   # Layer placement strategies - try different arrangements
#   encoder_shared_placement = c("at_end", "at_start", "alternating", "random"),
#   decoder_shared_placement = c("at_start", "at_end", "alternating", "random"),
# 
#   refit_patience = 2,        # Early stopping patience
#   refit_loops = 100,                # Fixed refit loops
#   epochs_per_loop = 100,   # Epochs per refit loop
#   reset_lr_refit = c(TRUE, FALSE)
# )
# 
# # Analyze results
# imputed <- aut$imputed
# best_model <- aut$model
# study <- aut$study
# results <- aut$results
# 
# # View best hyperparameters
# print("Trial results:")
# results %>% kable() %>%
#   kable_styling(font_size=12)
# 
# # Plot model architecture
# plot_vae_architecture(best_model, title = "Optimized CISSVAE Architecture")

## ----showres5, echo=FALSE-----------------------------------------------------
results = readRDS(system.file("extdata", "autotune.rds", package = "rCISSVAE"))



# View best hyperparameters
print("Trial results:")
results %>% kableExtra::kable() %>%
  kableExtra::kable_styling(font_size=12)

