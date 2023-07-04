# load required libraries
library("tidyverse")
library("patchwork")
library("RColorBrewer")
library("ggbreak")
library("grid")

# set working directory
setwd("/home/pokepup/DTU_Subjects/MSc_thesis/results/Benchmark/Model_3/")

# load benchmark file
df = read_csv("Benchmark_emp_Model3_alpha01.csv")

# load benchmarks with changes alpha or gradient
df1 = read_csv("Method_3/Benchmark_empirical_M3.csv")
df2 = read_csv("Benchmark_empirical_noeigvec.csv")
df3 = read_csv("Benchmark_empirical_nogradient.csv")

# Function: normalize_fstat
# ------------------------
# This function takes a dataframe of statistical metrics, calculates the normalized F-statistic,
# and returns a new dataframe with the normalized F-statistic values along with the mean and standard deviation
# of the p-values and the normalized F-statistic, grouped by sparse_level, metric_ID, and sample_size.
#
# Arguments:
#   - df_stats: The input dataframe containing statistical metrics.
#
# Returns:
#   - A new dataframe with normalized F-statistic values, mean and standard deviation of p-values and normalized F-statistic.
#
# Usage:
#   normalized_stats <- normalize_fstat(df_stats)
#
normalize_fstat <- function(df_stats) {
  # Extract unweighted and weighted F-statistics
  unweighted_Fstat <- df_stats[grep("_w", df_stats$metric_ID, invert = TRUE), ]
  weighted_Fstat <- df_stats[grep("_w", df_stats$metric_ID), ]
  
  # Merge F-statistics into a new dataframe
  df_merged <- data.frame(x = weighted_Fstat$F_stat, y = unweighted_Fstat$F_stat)
  
  # Calculate normalized F-statistic by dividing weighted F-statistic by unweighted F-statistic
  new_df <- weighted_Fstat %>% 
    mutate(fstat_norm = df_merged$x / df_merged$y) %>% 
    group_by(sparse_level, metric_ID, sample_size) %>% 
    summarise_at(c("p_val", "fstat_norm"), c(mean, sd)) %>% 
    as_tibble()
  
  return(new_df)
}

# Function: emp_stats
# -------------------
# This function takes a dataframe of empirical data values, calculates the mean and standard deviation
# of the variable explained (in percentage), and returns a new dataframe with the aggregated statistics
# grouped by sparse_level, metric_ID, and sample_size.
#
# Arguments:
#   - df: The input dataframe containing empirical data values
#
# Returns:
#   - A new dataframe with aggregated statistics for the variable explained.
#
# Usage:
#   empirical_stats <- emp_stats(df)
#
emp_stats <- function(df) {
  # Multiply sparse_level and var_explained by 100 for percentage representation
  df_stats <- df %>% 
    mutate(sparse_level = sparse_level * 100,
           var_explained = var_explained * 100) %>% 
    group_by(sparse_level, metric_ID, sample_size) %>% 
    summarise_at(c("var_explained"), c(mean, sd)) %>% 
    as_tibble()
  
  return(df_stats)
}

# Function: sim_stats
# -------------------
# This function takes a dataframe of simulated data values, calculates the mean and standard deviation
# of the variable explained (in percentage) after adjusting the sparse_level, and returns a new dataframe
# with the aggregated statistics grouped by sparse_level, metric_ID, and sample_size.
#
# Arguments:
#   - df: The input dataframe containing simulated data values
#
# Returns:
#   - A new dataframe with aggregated statistics for the variable explained.
#
# Usage:
#   simulated_stats <- sim_stats(df)
#
sim_stats <- function(df) {
  # Adjust sparse_level and multiply var_explained by 100 for percentage representation
  df_stats <- df %>% 
    mutate(sparse_level = (1 - sparse_level) * 100,
           var_explained = var_explained * 100) %>% 
    group_by(sparse_level, metric_ID, sample_size) %>% 
    summarise_at(c("var_explained"), c(mean, sd)) %>% 
    as_tibble()
  
  return(df_stats)
}

# Function: Kruskal_test_df
# ------------------------
# This function performs the Kruskal-Wallis test on three dataframes and computes kruskal wallis tests
# between the F_stat_fn1 and var_explained_fn1 columns.
#
# Arguments:
#   - df1: The first dataframe.
#   - df2: The second dataframe.
#   - df3: The third dataframe.
#   - filename: The name of the CSV file to store the results.
#
# Returns:
#   - Kruskal wallis test results
# Usage:
#   Kruskal_test_df(df1, df2, df3, "results.csv")
#
Kruskal_test_df <- function(df1, df2, df3, filename) {
  # Combine the dataframes
  combined_df <- bind_rows(
    df1 %>% mutate(Method = "Method A"),
    df2 %>% mutate(Method = "Method B"),
    df3 %>% mutate(Method = "Method C")
  )
  
  # Perform the Kruskal-Wallis test
  kruskal_test <- kruskal.test(var_explained + F_stat ~ Method, data = combined_df)
  return(kruskal_test)
}

# Function: Variance_plot
# -----------------------
# This function creates a variance plot based on the given dataframe. It filters the data for a specific sample size,
# and then uses ggplot to generate a plot with points, lines, and error bars. The x-axis represents the sparse level,
# the y-axis represents the fn1 values, and the color represents the metric ID. The plot title and legend position can be customized.
#
# Arguments:
#   - df: The input dataframe containing the data.
#   - n_size: The sample size to filter the data.
#   - plot_title: The title of the plot.
#   - legend: The position of the legend in the plot.
#
# Returns:
#   - The generated variance plot.
#
# Usage:
#   variance_plot <- Variance_plot(df, n_size = 100, plot_title = "Variance Plot", legend = "bottom")
#
Variance_plot <- function(df, n_size, plot_title, legend) {
  x_ticks_labels <- c("0", "10", "20", "30", "40", "50", "60", "70", "80", "90")
  
  var_plot <- df %>% 
    filter(sample_size == n_size) %>% 
    ggplot(mapping = aes(x = sparse_level,
                         y = fn1,
                         color = metric_ID)) +
    geom_point(position = position_dodge()) +
    geom_line() +
    geom_errorbar(aes(ymin = fn1 - fn2,
                      ymax = fn1 + fn2),
                  width = 0.2) +
    scale_x_continuous(labels = x_ticks_labels,
                       limits = c(0, 90), 
                       breaks = seq(0, 90, by = 10)) +
    scale_y_continuous(limits = c(0, 100), breaks = seq(0, 100, by = 10)) +
    labs(x = "zero's [%]",
         y = "distance explained [%]",
         title = plot_title) +
    theme_gray() + 
    theme(legend.position = legend) +
    scale_color_brewer(palette = "Paired") +
    guides(fill = guide_legend(title = "Distance explained metric ID's")) +
    theme(text = element_text(size = 20))
  
  return(var_plot)
}

# Function: permanova_plot
# ------------------------
# This function generates a bar plot using a dataframe of statistical metrics. The plot shows the log-transformed
# Pseudo F statistic values (y-axis) for different levels of zeros (x-axis) and different metric IDs (legend).
# Error bars are also included. The function allows customization of plot parameters such as plot title, legend position,
# axis limits, tick labels, and more.
#
# Arguments:
#   - df: The input dataframe containing statistical metrics.
#   - n_size: The sample size to be considered for the plot.
#   - plot_title: The title of the plot.
#   - legend: The position of the legend in the plot.
#   - x1: The lower limit of the y-axis.
#   - x2: The upper limit of the y-axis.
#   - x_step: The step size between tick marks on the y-axis.
#   - ppos: The vertical position of the significance asterisk.
#
# Returns:
#   - A ggplot object representing the PERMANOVA plot.
#
# Usage:
#   plot <- permanova_plot(df, n_size = 100, plot_title = "PERMANOVA Plot", legend = "right",
#                          x1 = 0.01, x2 = 10, x_step = 0.1, ppos = 0.5)
#
permanova_plot <- function(df, n_size, plot_title, legend, x1, x2, x_step, ppos) {
  x_ticks_labels <- c("0", "10", "20", "30", "40", "50", "60", "70", "80", "90")
  Fstat_plot <- df %>% 
    filter(sample_size == n_size) %>% 
    ggplot(mapping = aes(x = as.factor(sparse_level),
                         y = fstat_norm_fn1,
                         fill = metric_ID)) +
    geom_col(position = "dodge") + 
    geom_errorbar(aes(ymin=fstat_norm_fn1,
                      ymax=fstat_norm_fn1 + fstat_norm_fn2),
                  width=0.2, position = position_dodge(width = .9)) +
    scale_y_log10() +
    scale_x_discrete(labels = x_ticks_labels) +
    scale_y_continuous(limits = c(x1, x2), breaks = seq(x1, x2, by = x_step)) +
    labs(x = "zero's [%]",
         y = "log(Pseudo F statistic)",
         title = plot_title) +
    theme_gray() + 
    scale_fill_brewer(palette = "Set2") + 
    theme(legend.position = legend) +
    geom_text(aes(label = ifelse((p_val_fn1 + p_val_fn2) < 0.05 & fstat_norm_fn1 > 1, "*", "")), 
              position = position_dodge(width = .9), vjust = ppos, size = 5, color="red") +
    guides(fill = guide_legend(title = "PERMANOVA metric ID")) +
    theme(text = element_text(size = 20))
  return(Fstat_plot)
}

# Function: correlations
# ----------------------
# This function calculates the correlation matrix between different statistical metrics from three input dataframes,
# and generates a correlation plot using the corrplot package. The correlation plot visualizes the correlation coefficients
# between the metrics, highlighting the upper triangular matrix.
#
# Arguments:
#   - df1_stat: The first dataframe containing statistical metrics.
#   - df2_stat: The second dataframe containing statistical metrics.
#   - df3_stat: The third dataframe containing statistical metrics.
#
# Returns:
#   - A correlation plot representing the correlation matrix between the metrics.
#
# Usage:
#   plot <- correlations(df1_stat, df2_stat, df3_stat)
#
correlations <- function(df1_stat, df2_stat, df3_stat) {
  # Merge the statistical metrics into a single dataframe
  df_merged <- df1_stat %>% 
    mutate(F_stat_alpha01 = F_stat_fn1,
           var_explained_alpha01 = var_explained_fn1,
           F_stat_alpha001 = df2_stat$F_stat_fn1,
           var_explained_alpha001 = df2_stat$var_explained_fn1,
           F_stat_alpha_unfixed = df3_stat$F_stat_fn1,
           var_explained_alpha_unfixed = df3_stat$var_explained_fn1)
  
  # Calculate the correlation matrix
  cor_mat <- df_merged %>% select(F_stat_alpha01,
                                  var_explained_alpha01,
                                  F_stat_alpha001,
                                  var_explained_alpha001,
                                  F_stat_alpha_unfixed,
                                  var_explained_alpha_unfixed
  ) %>% 
    cor()
  
  # Generate the correlation plot
  cor_plot <- corrplot(cor_mat, type="upper", order="hclust", tl.srt=45)
  return(cor_plot)
}

#----------------------------------------------------------------------------------------------------------------------------------------#
### Variance explained Benchmark, sample series ###
#----------------------------------------------------------------------------------------------------------------------------------------#
V20 <- df %>% emp_stats() %>% 
  Variance_plot(n_size = 20, 
                plot_title = "Sample size = 20",
                legend = "none") +
  theme(panel.border = element_rect(color="black", fill = NA, size = 1),
        legend.position = "right")

V40 <- df %>% emp_stats() %>% 
  Variance_plot(n_size = 40, 
                plot_title = "Sample size = 40",
                legend = "none") + 
  theme(panel.border = element_rect(color="black", fill = NA, size = 1))

V60 <- df %>% emp_stats() %>% 
  Variance_plot(n_size = 60, 
                plot_title = "Sample size = 60",
                legend = "none") + 
  theme(panel.border = element_rect(color="black", fill = NA, size = 1))

V80 <- df %>% emp_stats() %>% 
  Variance_plot(n_size = 80, 
                plot_title = "Sample size = 80",
                legend = "bottom") + 
  theme(panel.border = element_rect(color="black", fill = NA, size = 1))

F20 <- df %>% 
  normalize_fstat() %>% 
  permanova_plot(n_size = 20,
                 plot_title = "Sample size = 20",
                 legend = "none",
                 x1 = 0, x2 = 2950, x_step = 5,
                 ppos = -.1) + scale_y_break(c(5,10)) + scale_y_break(c(30,405)) + scale_y_break(c(407, 2945)) +
  theme(panel.border = element_rect(color="black", fill = NA, size = 1),
        legend.position = "right")#+ scale_y_break(c(420,1350)) + scale_y_break(c(1370,2940))

F40 <- df %>% 
  normalize_fstat() %>% 
  permanova_plot(n_size = 40,
                 plot_title = "Sample size = 40",
                 legend = "none",
                 x1 = 0, x2 = 500, x_step = 5,
                 ppos = -.1) + scale_y_break(c(10,50)) + scale_y_break(c(75,500)) +
  theme(panel.border = element_rect(color="black", fill = NA, size = 1))

F60 <- df %>% 
  normalize_fstat() %>% 
  permanova_plot(n_size = 60,
                 plot_title = "Sample size = 60",
                 legend = "none",
                 x1 = 0, x2 = 500, x_step = 5,
                 ppos = -.1) + scale_y_break(c(50, 460)) + scale_y_break(c(467,500)) +
  theme(panel.border = element_rect(color="black", fill = NA, size = 1),
        plot.margin = margin(t = 0, r = 0, b = 5, l = 0)) #+ scale_y_break(c(5,100))

F80 <- df %>% 
  normalize_fstat() %>% 
  permanova_plot(n_size = 80,
                 plot_title = "Sample size = 80",
                 legend = "right",
                 x1 = 0, x2 = 500, x_step = 5,
                 ppos = -.1) + scale_y_break(c(15, 100)) + scale_y_break(c(105,267)) + scale_y_break(c(270,500)) +
  theme(panel.border = element_rect(color="black", fill = NA, size = 1))# + scale_y_break(c(5,100))

Final_V <- (V20 + V40) / (V60 + V80) + plot_annotation(tag_levels = "A") +
  plot_layout(guides = "collect") &
  theme(plot.tag = element_text(size = 25, face = "bold")) & theme(legend.position = 'bottom')

ggsave(filename = "/home/pokepup/DTU_Subjects/MSc_thesis/results/Benchmark/Model_1/Method_3/Benchmark_emp_M3_Var.png", 
       plot = Final_V,
       width = 15,
       height = 10)

# PERMANOVA individual plots
ggsave(filename = "/home/pokepup/DTU_Subjects/MSc_thesis/results/Benchmark/Model_1/Method_3/Benchmark_emp_M3_F80.png", 
       plot = F80,
       width = 8,
       height = 5)

Final <- (V20 + F20) + plot_annotation(tag_levels = "A") + 
  plot_layout(ncol = 2) &
  theme(plot.tag = element_text(size = 25, face = "bold")) & theme(legend.position = 'right')

ggsave(filename = "/home/pokepup/DTU_Subjects/MSc_thesis/results/Benchmark/Model_3/Benchmark_emp_Model3_S20.png", 
       plot = Final,
       width = 17,
       height = 10)
#----------------------------------------------------------------------------------------------------------------------------------------#
### Benchmarking of gradients ###
#----------------------------------------------------------------------------------------------------------------------------------------#

Var_grad <- df1 %>% 
  filter(sample_size == 20) %>%
  emp_stats() %>% 
  Variance_plot(n_size = 20, 
                plot_title = "gradient Method 3",
                legend = "none") +
  theme(panel.border = element_rect(color="black", fill = NA, size = 1))

Var_noeigvec <- df2 %>% emp_stats() %>% 
  Variance_plot(n_size = 20, 
                plot_title = "Without eigenvectors",
                legend = "none") +
  theme(panel.border = element_rect(color="black", fill = NA, size = 1))

Var_nograd <- df3 %>% emp_stats() %>% 
  Variance_plot(n_size = 20, 
                plot_title = "Without gradient",
                legend = "right") +
  theme(panel.border = element_rect(color="black", fill = NA, size = 1))

F_grad <- df1 %>% 
  filter(sample_size == 20) %>% 
  normalize_fstat() %>% 
  permanova_plot(n_size = 20,
                 plot_title = "gradient Method 3",
                 legend = "none",
                 x1 = 0, x2 = 50, x_step = 2,
                 ppos = -.1) + scale_y_break(c(6,37)) + scale_y_break(c(39,50)) +
  theme(panel.border = element_rect(color="black", fill = NA, size = 1))

F_noeigvec <- df2 %>% 
  normalize_fstat() %>% 
  permanova_plot(n_size = 20,
                 plot_title = "Without eigenvectors",
                 legend = "none",
                 x1 = 0, x2 = 50, x_step = 2,
                 ppos = -.1) + scale_y_break(c(4,14)) + scale_y_break(c(17,50)) +
  theme(panel.border = element_rect(color="black", fill = NA, size = 1))

F_nograd <- df3 %>% 
  normalize_fstat() %>% 
  permanova_plot(n_size = 20,
                 plot_title = "Without gradient",
                 legend = "right",
                 x1 = 0, x2 = 50, x_step = 2,
                 ppos = -.1) + scale_y_break(c(10,50)) + 
  theme(panel.border = element_rect(color="black", fill = NA, size = 1))

Final_Vgrad <- (Var_grad / Var_noeigvec / Var_nograd) + plot_annotation(tag_levels = "A") +
  plot_layout(guides = "collect") &
  theme(plot.tag = element_text(size = 25, face = "bold")) & theme(legend.position = 'right')

ggsave(filename = "/home/pokepup/DTU_Subjects/MSc_thesis/results/Benchmark/Model_1/Benchmark_emp_gradients_Var.png", 
       plot = Final_Vgrad,
       width = 10,
       height = 15)

sub_grads <- (F_grad + F_noeigvec) | F_nograd

Final_Fgrad <- sub_grads + plot_annotation(tag_levels = "A") +
  plot_layout(guides = "collect") &
  theme(plot.tag = element_text(size = 25, face = "bold"),
        legend.position = "right")

ggsave(filename = "/home/pokepup/DTU_Subjects/MSc_thesis/results/Benchmark/Model_1/Benchmark_emp_gradients_Fgrad.png", 
       plot = F_grad,
       width = 8,
       height = 5)


#----------------------------------------------------------------------------------------------------------------------------------------#
### Benchmarking of alphas ###
#----------------------------------------------------------------------------------------------------------------------------------------#
Var_alpha01 <- df1 %>% 
  filter(sample_size == 20) %>%
  emp_stats() %>% 
  Variance_plot(n_size = 20, 
                plot_title = "alpha = 0.1",
                legend = "none") +
  theme(panel.border = element_rect(color="black", fill = NA, size = 1))

Var_alpha001 <- df2 %>% emp_stats() %>% 
  Variance_plot(n_size = 20, 
                plot_title = "alpha = 0.01",
                legend = "none") +
  theme(panel.border = element_rect(color="black", fill = NA, size = 1))

Var_alphaUnf <- df3 %>% emp_stats() %>% 
  Variance_plot(n_size = 20, 
                plot_title = "alpha = variable",
                legend = "right") +
  theme(panel.border = element_rect(color="black", fill = NA, size = 1))

Final_Valph <- (Var_alpha01 / Var_alpha001 / Var_alphaUnf) + plot_annotation(tag_levels = "A") +
  plot_layout(guides = "collect") &
  theme(plot.tag = element_text(size = 25, face = "bold")) & theme(legend.position = 'right')

ggsave(filename = "/home/pokepup/DTU_Subjects/MSc_thesis/results/Benchmark/Model_1/Benchmark_emp_alphas_Var.png", 
       plot = Final_Valph,
       width = 10,
       height = 15)


F_alpha01 <- df1 %>% 
  filter(sample_size == 20) %>% 
  normalize_fstat() %>% 
  permanova_plot(n_size = 20,
                 plot_title = "alpha = 0.1",
                 legend = "none",
                 x1 = 0, x2 = 50, x_step = 2,
                 ppos = -.1) + scale_y_break(c(6,38)) + scale_y_break(c(39,50)) +
  theme(panel.border = element_rect(color="black", fill = NA, size = 1))

F_alpha001 <- df2 %>% 
  normalize_fstat() %>% 
  permanova_plot(n_size = 20,
                 plot_title = "alpha = 0.01",
                 legend = "none",
                 x1 = 0, x2 = 50, x_step = 2,
                 ppos = -.1) + scale_y_break(c(8,34)) + scale_y_break(c(42,50)) + 
  theme(panel.border = element_rect(color="black", fill = NA, size = 1))

F_alphaUnf <- df3 %>% 
  normalize_fstat() %>% 
  permanova_plot(n_size = 20,
                 plot_title = "alpha = variable",
                 legend = "right",
                 x1 = 0, x2 = 50, x_step = 2,
                 ppos = -.1) + scale_y_break(c(11,15)) + scale_y_break(c(17,50)) +
  theme(panel.border = element_rect(color="black", fill = NA, size = 1))
  
sub_grads <- (F_alpha01 + F_alpha001) | F_alphaUnf

Final_Fgrad <- sub_grads + plot_annotation(tag_levels = "A") +
  plot_layout(guides = "collect") &
  theme(plot.tag = element_text(size = 25, face = "bold"),
        legend.position = "right")

ggsave(filename = "/home/pokepup/DTU_Subjects/MSc_thesis/results/Benchmark/Model_1/Benchmark_emp_alphas_01.png", 
       plot = F_alpha01,
       width = 8,
       height = 5)

