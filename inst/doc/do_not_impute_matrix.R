## ----setup, echo=FALSE, include=FALSE-----------------------------------------
rm(list = ls())

library(rCISSVAE)
library(tidyverse)
library(kableExtra)
library(reticulate)
library(gtsummary)
set.seed(42)


## -----------------------------------------------------------------------------
library(rCISSVAE)

data(dni, mock_surv)

mock_surv %>% 
   tbl_summary(include = -c("patient_id")) %>%
   as_kable()

## -----------------------------------------------------------------------------
mock_surv[7,] %>% kableExtra::kable()

dni[7,] %>%  kableExtra::kable()

## ----eval=FALSE---------------------------------------------------------------
# res <- run_cissvae(
#   mock_surv,
#   index_col = "patient_id",
#   columns_ignore = c("death_event","death_year"),
#   imputable_matrix = dni,
#   val_proportion = 0.3,
#   return_clusters =FALSE,
#   return_history = FALSE,
#   epochs = 100,
#   leiden_resolution = 0.01,
#   k_neighbors = 5,
#   return_silhouettes = FALSE
# )
# res$imputed_dataset[1:7,] %>% kableExtra::kable()
# 

## ----include=FALSE, echo=FALSE------------------------------------------------
res <- readRDS(system.file("extdata", "imputable.rds", package = "rCISSVAE"))
res[1:7,] %>% kableExtra::kable()

