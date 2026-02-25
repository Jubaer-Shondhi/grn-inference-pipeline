#!/usr/bin/env Rscript

#' Install required R packages for GRN simulation
#' 
#' This script installs all necessary R packages for running the
#' GRN simulation pipeline.

install_dependencies <- function() {
  
  cat("========================================\n")
  cat("Installing R dependencies for GRN Simulation\n")
  cat("========================================\n\n")
  
  # List of required packages
  packages <- c(
    "decoupleR",      # For CollectRI network
    "scMultiSim",     # For simulation
    "data.table",     # For efficient data handling
    "dplyr",          # For data manipulation
    "ggplot2",        # For plotting
    "argparse"        # For command line arguments
  )
  
  # Install BiocManager if needed (for Bioconductor packages)
  if (!requireNamespace("BiocManager", quietly = TRUE)) {
    cat("\nInstalling BiocManager...\n")
    install.packages("BiocManager", repos = "https://cloud.r-project.org")
  }
  
  # Track installation status
  installed <- c()
  failed <- c()
  
  # Install each package
  for (pkg in packages) {
    cat(sprintf("\nChecking %s...\n", pkg))
    
    if (!requireNamespace(pkg, quietly = TRUE)) {
      cat(sprintf("Installing %s...\n", pkg))
      
      tryCatch({
        if (pkg %in% c("decoupleR", "scMultiSim")) {
          # Bioconductor packages
          BiocManager::install(pkg, update = FALSE, ask = FALSE)
        } else {
          # CRAN packages
          install.packages(pkg, repos = "https://cloud.r-project.org")
        }
        
        # Verify installation
        if (requireNamespace(pkg, quietly = TRUE)) {
          cat(sprintf("✓ %s installed successfully\n", pkg))
          installed <- c(installed, pkg)
        } else {
          cat(sprintf("✗ %s installation failed\n", pkg))
          failed <- c(failed, pkg)
        }
      }, error = function(e) {
        cat(sprintf("✗ Error installing %s: %s\n", pkg, e$message))
        failed <- c(failed, pkg)
      })
    } else {
      cat(sprintf("✓ %s already installed\n", pkg))
      installed <- c(installed, pkg)
    }
  }
  
  # Print summary
  cat("\n\n========================================\n")
  cat("Installation Summary\n")
  cat("========================================\n")
  cat(sprintf("Successfully installed: %d packages\n", length(installed)))
  if (length(failed) > 0) {
    cat(sprintf("Failed to install: %d packages\n", length(failed)))
    cat("Failed packages: ", paste(failed, collapse = ", "), "\n")
  }
  
  # Test CollectRI loading
  cat("\nTesting CollectRI network loading...\n")
  tryCatch({
    collectri <- decoupleR::get_collectri()
    cat(sprintf("✓ CollectRI loaded successfully: %d edges\n", nrow(collectri)))
  }, error = function(e) {
    cat(sprintf("✗ Failed to load CollectRI: %s\n", e$message))
  })
  
  invisible(list(installed = installed, failed = failed))
}

# Run installation if script is executed directly
if (sys.nframe() == 0) {
  install_dependencies()
}