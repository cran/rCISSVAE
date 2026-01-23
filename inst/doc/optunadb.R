## ----run_autotune, eval=FALSE-------------------------------------------------
# library(tidyverse)
# library(reticulate)
# library(rCISSVAE)
# library(kableExtra)
# 
# ## Set the virtual environment
# # reticulate::use_virtualenv("./.venv", required = TRUE)
# 
# ## Load the data
# data(df_missing)
# data(clusters)
# 
# aut <- autotune_cissvae(
#   data = df_missing,
#   index_col = "index",
#   clusters = clusters$clusters,
#   n_trials = 3, ## Using low number of trials for demo
#   study_name = "ShowOptunaDB",
#   device_preference = "cpu",
#   optuna_dashboard_db = "sqlite:///optuna_study_demo.db",  # Save results to database
#   load_if_exists = TRUE, ## Set true to load and continue study if it exists
#   seed = 42,
#   verbose = FALSE,
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
#   num_epochs = 5,                # Fixed epochs for demo
#   batch_size = c(1000, 4000),     # Batch size options
#   num_shared_encode = c(0, 1, 2, 3),
#   num_shared_decode = c(0, 1, 2, 3),
# 
#   # Layer placement strategies - try different arrangements
#   encoder_shared_placement = c("at_end", "at_start", "alternating", "random"),
#   decoder_shared_placement = c("at_start", "at_end", "alternating", "random"),
# 
#   refit_patience = 2,        # Early stopping patience
#   refit_loops = 10,                # Fixed refit loops
#   epochs_per_loop = 5,   # Epochs per refit loop
#   reset_lr_refit = FALSE
# )
# 
# # Analyze results
# imputed <- aut$imputed
# best_model <- aut$model
# results <- aut$results
# 
# # View best hyperparameters
# print("Trial results:")
# results |> kable() |>
#   kable_styling(font_size=12)
# 

## ----showres, echo=FALSE------------------------------------------------------
library(tidyverse)
results = readRDS(system.file("extdata", "autotune.rds", package = "rCISSVAE"))


print("Trial results:")
results |> kableExtra::kable() |>
  kableExtra::kable_styling(font_size=12)


