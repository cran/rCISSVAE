## ----setup, echo=FALSE, include=FALSE-----------------------------------------
rm(list = ls())

library(rCISSVAE)
library(tidyverse)
library(kableExtra)
library(reticulate)

set.seed(42)


## ----start--------------------------------------------------------------------
library(tidyverse)
library(kableExtra)
library(reticulate)
library(rCISSVAE)
library(fastDummies)
library(palmerpenguins)
data(package = 'palmerpenguins')

penguins_clean = na.omit(penguins)%>%
        select(year, everything()) ## removing existing incomplete rows for illustration purposes

glue::glue("Dimensions: {paste0(dim(penguins), collapse = ',')}")

head(penguins_clean) %>% kable()

## create penguins_missing
n  <- nrow(penguins_clean)
p  <- ncol(penguins_clean)
m  <- floor(0.20 * n * p)               # number of cells to mask
idx <- sample.int(n * p, m)             # positions in a logical matrix

mask <- matrix(FALSE, nrow = n, ncol = p)
mask[idx] <- TRUE

penguins_missing <- penguins_clean

## anything can be missing except the year
for (j in seq(2, p, 1)) {
  penguins_missing[[j]][mask[, j]] <- NaN
}

# quick check of missingness rate
glue::glue("\nMissingness proportion of penguins_missing: {round(mean(is.na(as.matrix(penguins_missing))), 2)}") 

## create dummy vars

penguin_dummies_complete = penguins_clean %>% 
    dummy_cols(select_columns = c("species", "island", "sex"),
    ignore_na = TRUE,
    remove_first_dummy = TRUE,
    remove_selected_columns = TRUE) 

penguin_dummies = penguins_missing %>% 
    dummy_cols(select_columns = c("species", "island", "sex"),
    ignore_na = TRUE,
    remove_first_dummy = TRUE,
    remove_selected_columns = TRUE)

head(penguin_dummies) %>% kable()


## ----eval=FALSE---------------------------------------------------------------
# binary_feature_mask = c(rep(FALSE, 5), rep(TRUE, 5))
# 
# glue::glue("Binary Feature Mask: {paste0(binary_feature_mask, collapse = ', ')}")
# 
# results = run_cissvae(
#     data = penguin_dummies,
#     val_proportion = 0.20, ## small dataset so using higher val proportion
#     columns_ignore = "year",
#     binary_feature_mask = binary_feature_mask,
#     clusters = NULL,
#     n_clusters = 1,
#     scale_features = TRUE,
#     epochs = 500,
#     debug = FALSE
# )
# 
# head(results$imputed_dataset)
# head(penguin_dummies)

## ----echo=FALSE---------------------------------------------------------------
results = readRDS(system.file("extdata", "binary_res.rds", package = "rCISSVAE"))

head(results)
head(penguin_dummies)

## ----eval=FALSE---------------------------------------------------------------
# results$imputed_dataset <- results$imputed_dataset %>%
#   mutate(across(
#     .cols = matches("species|island|sex"),
#     .fns = ~ case_when(
#       .x > 0.5 ~ 1,
#       .x <= 0.5 ~ 0,
#       TRUE ~ .x
#     )
#   ))
# 
# head(results$imputed_dataset)
# head(penguin_dummies)
# head(penguin_dummies_complete)

## ----echo=FALSE---------------------------------------------------------------
results <- results %>%
  mutate(across(
    .cols = matches("species|island|sex"),
    .fns = ~ case_when(
      .x > 0.5 ~ 1,
      .x <= 0.5 ~ 0,
      TRUE ~ .x
    )
  ))

head(results)
head(penguin_dummies)
head(penguin_dummies_complete)

