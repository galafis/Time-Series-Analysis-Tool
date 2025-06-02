# Advanced Time Series Analysis Tool - R Implementation
# Author: Gabriel Demetrios Lafis
# Comprehensive time series analysis with forecasting and anomaly detection

# Load required libraries
library(forecast)
library(tseries)
library(ggplot2)
library(dplyr)
library(lubridate)
library(plotly)
library(xts)
library(zoo)
library(changepoint)
library(prophet)
library(seasonal)
library(VIM)
library(corrplot)
library(gridExtra)

# Set theme for plots
theme_set(theme_minimal())

# Advanced Time Series Analysis Class
TimeSeriesAnalyzer <- setRefClass("TimeSeriesAnalyzer",
  fields = list(
    data = "data.frame",
    ts_data = "ts",
    frequency = "numeric",
    models = "list",
    forecasts = "list",
    diagnostics = "list"
  ),
  
  methods = list(
    # Initialize with data
    initialize = function(data_path = NULL, date_col = "date", value_col = "value", freq = 12) {
      if (!is.null(data_path)) {
        data <<- read.csv(data_path)
        data[[date_col]] <<- as.Date(data[[date_col]])
      } else {
        # Generate sample time series data
        set.seed(42)
        dates <- seq(as.Date("2020-01-01"), as.Date("2023-12-31"), by = "month")
        trend <- seq(100, 200, length.out = length(dates))
        seasonal <- 10 * sin(2 * pi * seq_along(dates) / 12)
        noise <- rnorm(length(dates), 0, 5)
        values <- trend + seasonal + noise
        
        data <<- data.frame(
          date = dates,
          value = values,
          category = sample(c("A", "B", "C"), length(dates), replace = TRUE)
        )
      }
      
      # Create time series object
      ts_data <<- ts(data[[value_col]], frequency = freq)
      frequency <<- freq
      models <<- list()
      forecasts <<- list()
      diagnostics <<- list()
      
      cat("Time Series Analyzer initialized with", nrow(data), "observations\n")
    },
    
    # Exploratory Data Analysis
    perform_eda = function() {
      cat("=== Time Series Exploratory Data Analysis ===\n")
      
      # Basic statistics
      cat("Basic Statistics:\n")
      print(summary(data$value))
      
      # Time series plot
      ts_plot <- ggplot(data, aes(x = date, y = value)) +
        geom_line(color = "steelblue", size = 1) +
        geom_smooth(method = "loess", color = "red", alpha = 0.3) +
        labs(title = "Time Series Overview",
             subtitle = "Original data with trend line",
             x = "Date", y = "Value") +
        theme_minimal()
      
      print(ts_plot)
      
      # Decomposition
      decomp <- decompose(ts_data)
      
      # Create decomposition plots
      decomp_df <- data.frame(
        date = rep(data$date, 4),
        component = rep(c("Observed", "Trend", "Seasonal", "Random"), each = nrow(data)),
        value = c(as.numeric(decomp$x), as.numeric(decomp$trend), 
                 as.numeric(decomp$seasonal), as.numeric(decomp$random))
      )
      
      decomp_plot <- ggplot(decomp_df, aes(x = date, y = value)) +
        geom_line(color = "steelblue") +
        facet_wrap(~component, scales = "free_y", ncol = 1) +
        labs(title = "Time Series Decomposition",
             x = "Date", y = "Value") +
        theme_minimal()
      
      print(decomp_plot)
      
      # Stationarity tests
      cat("\n=== Stationarity Tests ===\n")
      adf_test <- adf.test(ts_data)
      kpss_test <- kpss.test(ts_data)
      
      cat("Augmented Dickey-Fuller Test:\n")
      cat("p-value:", adf_test$p.value, "\n")
      cat("Stationary:", ifelse(adf_test$p.value < 0.05, "Yes", "No"), "\n\n")
      
      cat("KPSS Test:\n")
      cat("p-value:", kpss_test$p.value, "\n")
      cat("Stationary:", ifelse(kpss_test$p.value > 0.05, "Yes", "No"), "\n\n")
      
      # ACF and PACF plots
      par(mfrow = c(2, 2))
      acf(ts_data, main = "Autocorrelation Function")
      pacf(ts_data, main = "Partial Autocorrelation Function")
      acf(diff(ts_data), main = "ACF of Differenced Series")
      pacf(diff(ts_data), main = "PACF of Differenced Series")
      par(mfrow = c(1, 1))
      
      return(list(
        summary = summary(data$value),
        adf_test = adf_test,
        kpss_test = kpss_test,
        decomposition = decomp
      ))
    },
    
    # Anomaly Detection
    detect_anomalies = function(method = "iqr") {
      cat("=== Anomaly Detection ===\n")
      
      anomalies <- data.frame()
      
      if (method == "iqr") {
        # IQR method
        Q1 <- quantile(data$value, 0.25)
        Q3 <- quantile(data$value, 0.75)
        IQR <- Q3 - Q1
        lower_bound <- Q1 - 1.5 * IQR
        upper_bound <- Q3 + 1.5 * IQR
        
        anomalies <- data[data$value < lower_bound | data$value > upper_bound, ]
        
      } else if (method == "zscore") {
        # Z-score method
        z_scores <- abs(scale(data$value))
        anomalies <- data[z_scores > 3, ]
        
      } else if (method == "changepoint") {
        # Change point detection
        cpt_result <- cpt.mean(ts_data, method = "PELT")
        changepoints <- cpts(cpt_result)
        
        if (length(changepoints) > 0) {
          anomalies <- data[changepoints, ]
        }
      }
      
      cat("Detected", nrow(anomalies), "anomalies using", method, "method\n")
      
      # Plot anomalies
      anomaly_plot <- ggplot(data, aes(x = date, y = value)) +
        geom_line(color = "steelblue", alpha = 0.7) +
        geom_point(data = anomalies, aes(x = date, y = value), 
                  color = "red", size = 3, alpha = 0.8) +
        labs(title = paste("Anomaly Detection -", toupper(method), "Method"),
             subtitle = paste("Detected", nrow(anomalies), "anomalies"),
             x = "Date", y = "Value") +
        theme_minimal()
      
      print(anomaly_plot)
      
      return(anomalies)
    },
    
    # Fit multiple forecasting models
    fit_models = function() {
      cat("=== Fitting Forecasting Models ===\n")
      
      # ARIMA Model
      cat("Fitting ARIMA model...\n")
      models$arima <<- auto.arima(ts_data, seasonal = TRUE, stepwise = FALSE)
      
      # Exponential Smoothing
      cat("Fitting ETS model...\n")
      models$ets <<- ets(ts_data)
      
      # Seasonal Naive
      cat("Fitting Seasonal Naive model...\n")
      models$snaive <<- snaive(ts_data)
      
      # Prophet Model (if data has date column)
      if ("date" %in% names(data)) {
        cat("Fitting Prophet model...\n")
        prophet_data <- data.frame(
          ds = data$date,
          y = data$value
        )
        models$prophet <<- prophet(prophet_data, yearly.seasonality = TRUE)
      }
      
      # Linear Model with trend and seasonality
      cat("Fitting Linear Trend model...\n")
      time_index <- 1:length(ts_data)
      seasonal_dummies <- seasonaldummy(ts_data)
      lm_data <- data.frame(
        y = as.numeric(ts_data),
        time = time_index,
        seasonal_dummies
      )
      models$lm <<- lm(y ~ ., data = lm_data)
      
      cat("All models fitted successfully!\n")
    },
    
    # Generate forecasts
    generate_forecasts = function(h = 12) {
      cat("=== Generating Forecasts ===\n")
      
      if (length(models) == 0) {
        stop("No models fitted. Please run fit_models() first.")
      }
      
      # ARIMA forecast
      if ("arima" %in% names(models)) {
        forecasts$arima <<- forecast(models$arima, h = h)
      }
      
      # ETS forecast
      if ("ets" %in% names(models)) {
        forecasts$ets <<- forecast(models$ets, h = h)
      }
      
      # Seasonal Naive forecast
      if ("snaive" %in% names(models)) {
        forecasts$snaive <<- forecast(models$snaive, h = h)
      }
      
      # Prophet forecast
      if ("prophet" %in% names(models)) {
        future_dates <- make_future_dataframe(models$prophet, periods = h, freq = "month")
        forecasts$prophet <<- predict(models$prophet, future_dates)
      }
      
      cat("Forecasts generated for", h, "periods ahead\n")
    },
    
    # Evaluate model performance
    evaluate_models = function() {
      cat("=== Model Evaluation ===\n")
      
      if (length(forecasts) == 0) {
        stop("No forecasts available. Please run generate_forecasts() first.")
      }
      
      # Split data for validation
      train_size <- floor(0.8 * length(ts_data))
      train_data <- window(ts_data, end = c(start(ts_data)[1] + train_size %/% frequency, 
                                           start(ts_data)[2] + train_size %% frequency))
      test_data <- window(ts_data, start = c(start(ts_data)[1] + train_size %/% frequency + 1, 
                                            start(ts_data)[2] + train_size %% frequency))
      
      # Fit models on training data
      train_arima <- auto.arima(train_data)
      train_ets <- ets(train_data)
      train_snaive <- snaive(train_data)
      
      # Generate forecasts for test period
      test_length <- length(test_data)
      arima_forecast <- forecast(train_arima, h = test_length)
      ets_forecast <- forecast(train_ets, h = test_length)
      snaive_forecast <- forecast(train_snaive, h = test_length)
      
      # Calculate accuracy metrics
      arima_accuracy <- accuracy(arima_forecast, test_data)
      ets_accuracy <- accuracy(ets_forecast, test_data)
      snaive_accuracy <- accuracy(snaive_forecast, test_data)
      
      # Combine results
      accuracy_results <- rbind(
        data.frame(Model = "ARIMA", arima_accuracy[2, ]),
        data.frame(Model = "ETS", ets_accuracy[2, ]),
        data.frame(Model = "Seasonal Naive", snaive_accuracy[2, ])
      )
      
      print(accuracy_results)
      
      # Plot model comparison
      forecast_df <- data.frame(
        date = rep(time(test_data), 3),
        actual = rep(as.numeric(test_data), 3),
        forecast = c(as.numeric(arima_forecast$mean), 
                    as.numeric(ets_forecast$mean), 
                    as.numeric(snaive_forecast$mean)),
        model = rep(c("ARIMA", "ETS", "Seasonal Naive"), each = length(test_data))
      )
      
      comparison_plot <- ggplot(forecast_df, aes(x = date)) +
        geom_line(aes(y = actual), color = "black", size = 1, alpha = 0.8) +
        geom_line(aes(y = forecast, color = model), size = 1) +
        labs(title = "Model Comparison on Test Data",
             subtitle = "Black line = Actual, Colored lines = Forecasts",
             x = "Time", y = "Value", color = "Model") +
        theme_minimal() +
        theme(legend.position = "bottom")
      
      print(comparison_plot)
      
      diagnostics$accuracy <<- accuracy_results
      return(accuracy_results)
    },
    
    # Plot forecasts
    plot_forecasts = function() {
      cat("=== Plotting Forecasts ===\n")
      
      if (length(forecasts) == 0) {
        stop("No forecasts available. Please run generate_forecasts() first.")
      }
      
      # Create forecast plots
      plots <- list()
      
      if ("arima" %in% names(forecasts)) {
        plots$arima <- autoplot(forecasts$arima) +
          labs(title = "ARIMA Forecast", x = "Time", y = "Value") +
          theme_minimal()
      }
      
      if ("ets" %in% names(forecasts)) {
        plots$ets <- autoplot(forecasts$ets) +
          labs(title = "ETS Forecast", x = "Time", y = "Value") +
          theme_minimal()
      }
      
      if ("snaive" %in% names(forecasts)) {
        plots$snaive <- autoplot(forecasts$snaive) +
          labs(title = "Seasonal Naive Forecast", x = "Time", y = "Value") +
          theme_minimal()
      }
      
      # Arrange plots
      if (length(plots) > 0) {
        do.call(grid.arrange, c(plots, ncol = 2))
      }
    },
    
    # Advanced analysis
    perform_advanced_analysis = function() {
      cat("=== Advanced Time Series Analysis ===\n")
      
      # Spectral analysis
      spectrum_result <- spectrum(ts_data, plot = FALSE)
      
      # Create spectral density plot
      spectral_df <- data.frame(
        frequency = spectrum_result$freq,
        density = spectrum_result$spec
      )
      
      spectral_plot <- ggplot(spectral_df, aes(x = frequency, y = density)) +
        geom_line(color = "steelblue", size = 1) +
        labs(title = "Spectral Density",
             x = "Frequency", y = "Spectral Density") +
        theme_minimal()
      
      print(spectral_plot)
      
      # Cross-correlation analysis (if multiple series)
      if (ncol(data) > 2) {
        cat("\nCross-correlation analysis:\n")
        numeric_cols <- sapply(data, is.numeric)
        if (sum(numeric_cols) > 1) {
          cor_matrix <- cor(data[, numeric_cols], use = "complete.obs")
          corrplot(cor_matrix, method = "circle", type = "upper")
        }
      }
      
      # Volatility analysis
      returns <- diff(log(ts_data))
      volatility <- rollapply(returns^2, width = 12, FUN = mean, align = "right", fill = NA)
      
      vol_df <- data.frame(
        date = data$date[-1],
        returns = as.numeric(returns),
        volatility = as.numeric(volatility[-1])
      )
      
      vol_plot <- ggplot(vol_df, aes(x = date, y = volatility)) +
        geom_line(color = "red", alpha = 0.7) +
        labs(title = "Rolling Volatility (12-period window)",
             x = "Date", y = "Volatility") +
        theme_minimal()
      
      print(vol_plot)
      
      return(list(
        spectrum = spectrum_result,
        volatility = vol_df
      ))
    },
    
    # Generate comprehensive report
    generate_report = function() {
      cat("=== Time Series Analysis Report ===\n")
      cat("Author: Gabriel Demetrios Lafis\n")
      cat("Date:", Sys.Date(), "\n\n")
      
      cat("Dataset Summary:\n")
      cat("- Observations:", nrow(data), "\n")
      cat("- Time Range:", min(data$date), "to", max(data$date), "\n")
      cat("- Frequency:", frequency, "\n")
      cat("- Mean Value:", round(mean(data$value, na.rm = TRUE), 2), "\n")
      cat("- Standard Deviation:", round(sd(data$value, na.rm = TRUE), 2), "\n\n")
      
      if (length(models) > 0) {
        cat("Models Fitted:\n")
        for (model_name in names(models)) {
          cat("-", toupper(model_name), "\n")
        }
        cat("\n")
      }
      
      if (length(diagnostics) > 0 && "accuracy" %in% names(diagnostics)) {
        cat("Best Performing Model (by MAPE):\n")
        best_model <- diagnostics$accuracy[which.min(diagnostics$accuracy$MAPE), ]
        cat("- Model:", best_model$Model, "\n")
        cat("- MAPE:", round(best_model$MAPE, 2), "%\n")
        cat("- RMSE:", round(best_model$RMSE, 2), "\n")
      }
      
      cat("\nAnalysis completed successfully!\n")
      cat("This demonstrates advanced R capabilities for time series analysis:\n")
      cat("- Multiple forecasting models (ARIMA, ETS, Prophet)\n")
      cat("- Anomaly detection algorithms\n")
      cat("- Model evaluation and comparison\n")
      cat("- Advanced visualizations with ggplot2\n")
      cat("- Spectral analysis and volatility modeling\n")
    }
  )
)

# Example usage and demonstration
cat("=== Advanced Time Series Analysis Tool ===\n")
cat("Author: Gabriel Demetrios Lafis\n")
cat("Demonstrating comprehensive time series analysis in R\n\n")

# Create instance
analyzer <- TimeSeriesAnalyzer$new()

# Run complete analysis pipeline
cat("Running complete time series analysis pipeline...\n")
analyzer$perform_eda()
analyzer$detect_anomalies("iqr")
analyzer$fit_models()
analyzer$generate_forecasts(12)
analyzer$evaluate_models()
analyzer$plot_forecasts()
analyzer$perform_advanced_analysis()
analyzer$generate_report()

cat("\n=== Advanced Time Series Analysis Complete ===\n")
cat("This demonstrates:\n")
cat("- Object-oriented programming in R\n")
cat("- Multiple forecasting algorithms\n")
cat("- Statistical testing and validation\n")
cat("- Advanced data visualization\n")
cat("- Anomaly detection methods\n")
cat("- Model comparison and evaluation\n")
cat("- Spectral analysis and volatility modeling\n")

