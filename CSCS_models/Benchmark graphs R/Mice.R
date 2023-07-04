# load required libraries
library("tidyverse")
library("patchwork")
library("RColorBrewer")
library("corrplot")
library("grid")
library("ggbreak")

# set working directory
setwd("/home/pokepup/DTU_Subjects/MSc_thesis/data/metagenomics/Mice_data/")

# load abundance table (otu_table) and metadata
otu_table = read_csv("otu_table.csv")
metadata = read_csv("metadata.csv")

# Define two groups with sample IDs based on Sample.Time column
group_A <- metadata %>% 
  filter(Sample.Time == "Pre diet") %>% 
  pull(Sample.ID)

group_B <- metadata %>% 
  filter(Sample.Time == "Termination") %>% 
  pull(Sample.ID)

# Subset data based on sample IDs
subset_A <- otu_table[, colnames(otu_table) %in% group_A]
subset_B <- otu_table[, colnames(otu_table) %in% group_B]

# Plots the abundance distribution of group A
dist_A <- subset_A %>% 
  pivot_longer(cols = everything(), names_to = "Sample", values_to = "Counts") %>% 
  group_by(Sample) %>% 
  ggplot(mapping = aes(x = Counts)) +
  geom_density(fill = "blue", color = "black") +
  scale_x_break(c(200, 13500), space = 0.2) +
  xlim(0, 13633) + ylim(0, 2.5) +
  labs(x = "Abundance", y = "Frequency") +
  ggtitle("Pre diet") + 
  theme(text = element_text(size = 20))

# Plots the abundance distribution of group B
dist_B <- subset_B %>% 
  pivot_longer(cols = everything(), names_to = "Sample", values_to = "Counts") %>% 
  group_by(Sample) %>% 
  ggplot(mapping = aes(x = Counts)) +
  geom_density(fill = "blue", color = "black") +
  scale_x_break(c(200, 17000), space = 0.2) +
  xlim(0, 17573) + ylim(0, 2.5) +
  labs(x = "Abundance", y = "Frequency") +
  ggtitle("Termination") +
  theme(text = element_text(size = 20))

# Creates dataframe of number of positive and zero in group A
count_results_A <- data.frame(
  Column = colnames(subset_A),
  Zeros = apply(subset_A, 2, function(x) sum(x == 0)),
  Positives = apply(subset_A, 2, function(x) sum(x > 0))
) %>% pivot_longer(cols = c(Zeros, Positives),
                   names_to = "Category", values_to = "Count")

# Creates dataframe of number of positive and zero in group B
count_results_B <- data.frame(
  Column = colnames(subset_B),
  Zeros = apply(subset_B, 2, function(x) sum(x == 0)),
  Positives = apply(subset_B, 2, function(x) sum(x > 0))
) %>% pivot_longer(cols = c(Zeros, Positives),
                   names_to = "Category", values_to = "Count")


# Violin plots of count_results from group A
p1 <- count_results_A %>% 
  ggplot(mapping = aes(x = Category, y = Count)) +
  geom_violin() +
  geom_jitter(height = 0, width = 0.1) +
  ylim(250, 1000) +
  labs(x = "Category", y = "Count") +
  ggtitle("Pre diet") +
  theme_classic() +
  theme(text = element_text(size = 20))

# Violin plots of count_results from group B
p2 <- count_results_B %>% 
  ggplot(mapping = aes(x = Category, y = Count)) +
  geom_violin() +
  geom_jitter(height = 0, width = 0.1) +
  ylim(250, 1000) +
  labs(x = "Category", y = "Count") +
  ggtitle("Termination") +
  theme_classic() +
  theme(text = element_text(size = 20))

# Combined final plot of abundance distribution in Mice data
dists <- dist_A + dist_B
final_dist <- dists + plot_annotation(tag_levels = "A",
                        title = "Distribution of Mice data") &
                        theme(plot.tag = element_text(size = 20, face = "bold")) &
  theme(plot.title = element_text(size = 20))

ggsave(filename = "/home/pokepup/DTU_Subjects/MSc_thesis/results/Benchmark/Model_1/mice_dist.png", 
       plot = final_dist,
       width = 20,
       height = 10)

# Combined final plot of positive and zero counts in Mice data
violins <- p1 + p2
final_violins <- violins + plot_annotation(tag_levels = "A",
                                           ) &
  theme(plot.tag = element_text(size = 25, face = "bold"))

ggsave(filename = "/home/pokepup/DTU_Subjects/MSc_thesis/results/Benchmark/Model_1/mice_zeros_pos.png", 
       plot = final_violins,
       width = 15,
       height = 10)


# features_dist function: Generates a bar plot of the number of features against the sparse level
#
# Arguments:
#   - df1: A data frame containing the data for analysis
#   - n_size: The sample size to filter the data frame
#   - plot_title: The title of the plot
#
# Return:
#   - p1: A ggplot object representing the bar plot
#
# Usage: 
#   p <- features_dist(data_frame, 100, "Feature Distribution")
#
features_dist <- function(df1, n_size, plot_title) {
  x_ticks_labels <- c("0", "10", "20", "30", "40", "50", "60", "70", "80", "90")
  p1 <- df1 %>% mutate(sparse_level = sparse_level*100) %>% 
    select(sparse_level, n_features, sample_size) %>% 
    distinct() %>% 
    filter(sample_size == n_size) %>% 
    ggplot(mapping = aes(x = as.factor(sparse_level),
                         y = n_features)) +
    geom_col() + 
    scale_x_discrete(labels = x_ticks_labels) +
    ylim(0, 1000) +
    labs(x = "zero's [%]",
         y = "number of features",
         title = plot_title) +
    theme_classic() +
    theme(text = element_text(size = 20))
  return(p1)
}

# Sets working directory
setwd("/home/pokepup/DTU_Subjects/MSc_thesis/results/Benchmark/Model_1/Method_3/")

# loads empirical data with number of features used
df1 <- read_csv("Benchmark_empirical_M3.csv")

# Generates 4 plots for difference sample sizes
p1 <- df1 %>% features_dist(n_size = 20, plot_title = "Sample size = 20")
p2 <- df1 %>% features_dist(n_size = 40, plot_title = "Sample size = 40")
p3 <- df1 %>% features_dist(n_size = 60, plot_title = "Sample size = 60")
p4 <- df1 %>% features_dist(n_size = 80, plot_title = "Sample size = 80")

# Creates final plot of number of features with increasing sample size
patch_Var <- (p1 + p2) / (p3 + p4)
var_plot <- patch_Var + plot_annotation(tag_levels = "A") +
  plot_layout(guides = "collect") &
  theme(plot.tag = element_text(size = 25, face = "bold"))

ggsave(filename = "/home/pokepup/DTU_Subjects/MSc_thesis/results/Benchmark/Model_1/Feature_distribution_empirical_model.png", 
       plot = var_plot,
       width = 15,
       height = 10)
