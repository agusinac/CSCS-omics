library("tidyverse")
library("patchwork")
library("RColorBrewer")
library("corrplot")
library("grid")
library("ggbreak")

setwd("/home/pokepup/DTU_Subjects/MSc_thesis/data/metagenomics/Mice_data/")
otu_table = read_csv("otu_table.csv")
metadata = read_csv("metadata.csv")

group_A <- metadata %>% 
  filter(Sample.Time == "Pre diet") %>% 
  pull(Sample.ID)

group_B <- metadata %>% 
  filter(Sample.Time == "Termination") %>% 
  pull(Sample.ID)

subset_A <- otu_table[, colnames(otu_table) %in% group_A]
subset_B <- otu_table[, colnames(otu_table) %in% group_B]

# Plot the distribution using a histogram
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

count_results_A <- data.frame(
  Column = colnames(subset_A),
  Zeros = apply(subset_A, 2, function(x) sum(x == 0)),
  Positives = apply(subset_A, 2, function(x) sum(x > 0))
) %>% pivot_longer(cols = c(Zeros, Positives),
                   names_to = "Category", values_to = "Count")

count_results_B <- data.frame(
  Column = colnames(subset_B),
  Zeros = apply(subset_B, 2, function(x) sum(x == 0)),
  Positives = apply(subset_B, 2, function(x) sum(x > 0))
) %>% pivot_longer(cols = c(Zeros, Positives),
                   names_to = "Category", values_to = "Count")

# Calculate mean and standard deviation of combined samples
p1 <- count_results_A %>% 
  ggplot(mapping = aes(x = Category, y = Count)) +
  geom_violin() +
  geom_jitter(height = 0, width = 0.1) +
  ylim(250, 1000) +
  labs(x = "Category", y = "Count") +
  ggtitle("Pre diet") +
  theme_classic() +
  theme(text = element_text(size = 20))

p2 <- count_results_B %>% 
  ggplot(mapping = aes(x = Category, y = Count)) +
  geom_violin() +
  geom_jitter(height = 0, width = 0.1) +
  ylim(250, 1000) +
  labs(x = "Category", y = "Count") +
  ggtitle("Termination") +
  theme_classic() +
  theme(text = element_text(size = 20))

dists <- dist_A + dist_B
final_dist <- dists + plot_annotation(tag_levels = "A",
                        title = "Distribution of Mice data") &
                        theme(plot.tag = element_text(size = 20, face = "bold")) &
  theme(plot.title = element_text(size = 20))

ggsave(filename = "/home/pokepup/DTU_Subjects/MSc_thesis/results/Benchmark/Model_1/mice_dist.png", 
       plot = final_dist,
       width = 20,
       height = 10)

violins <- p1 + p2
final_violins <- violins + plot_annotation(tag_levels = "A",
                                           ) &
  theme(plot.tag = element_text(size = 25, face = "bold"))

ggsave(filename = "/home/pokepup/DTU_Subjects/MSc_thesis/results/Benchmark/Model_1/mice_zeros_pos.png", 
       plot = final_violins,
       width = 15,
       height = 10)

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
setwd("/home/pokepup/DTU_Subjects/MSc_thesis/results/Benchmark/Model_1/Method_3/")
df1 <- read_csv("Benchmark_empirical_M3.csv")

p1 <- df1 %>% features_dist(n_size = 20, plot_title = "Sample size = 20")
p2 <- df1 %>% features_dist(n_size = 40, plot_title = "Sample size = 40")
p3 <- df1 %>% features_dist(n_size = 60, plot_title = "Sample size = 60")
p4 <- df1 %>% features_dist(n_size = 80, plot_title = "Sample size = 80")

patch_Var <- (p1 + p2) / (p3 + p4)
var_plot <- patch_Var + plot_annotation(tag_levels = "A") +
  plot_layout(guides = "collect") &
  theme(plot.tag = element_text(size = 25, face = "bold"))

ggsave(filename = "/home/pokepup/DTU_Subjects/MSc_thesis/results/Benchmark/Model_1/Feature_distribution_empirical_model.png", 
       plot = var_plot,
       width = 15,
       height = 10)
