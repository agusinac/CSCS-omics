---
title: 'CSCSomics: A generalized similarity metric for omics analysis'
tags:
  - Python
  - omics
authors:
  - name: Alem Gusinac
    orcid: 0009-0006-1896-4176
    affiliation: "1"
affiliations:
 - name: Independent Researcher, Netherlands
   index: 1
date: 20 June 2025
bibliography: paper.bib
---

# Summary

Omics is the study of many genes, proteins or metabolites in a system. Omics analysis is compositional, meaning it represents the relative information from the absolute. Omics data may suffer of high dimensionality and sparsity that complicates downstream processes, causing overfitting and bias. Common dimensional reduction methods, such as principal coordinate analysis (PCoA), are often poor in handling sparse and high-dimensional data, making it challenging to interpret the results due to increased susceptibility to noise and outliers. In this work, the CSCSomics tool is developed that is capable to reduce the dimensionality of sparse data sets of metagenomics, proteomics and metabolomics data, in turn increasing the biological interpretation of PCoA. The CSCSomics is a generalized tool capable of applying the same method on a variety of omics data and allows optimization of other similarity/dissimilarity metrics via a custom option.

# Statement of need

All the three big omics; metagenomics, proteomics and metabolomics suffer from sparsity and high dimensionality [@Misra2019]. Sparse data is characterized by the absence of certain compounds (DNA, protein or metabolite) when analyzing multiple samples [@Ronan2016]. These absent compounds are denoted by a zero, this can be caused due to their absence in the sample or their low abundance, which puts them below the detection limit of the detector [@Busato2023]. Sparse omics is highly compositional and dimensional reduction methods, such as Principal Coordinate Analysis (PCoA) are often poor in handling sparse and high-dimensional data [@Martino2019] [@Dollhopf2001]. This leads to challenges in interpretation and increased sensitivity to noise and outliers [@Dollhopf2001].

In the present, a variety of statistical metrics exist to try to explain the compositional data [@Lozupone2010] [@Bray1957] [@Endres2003]. The Weighted UniFrac metric has been widely used to capture the dissimilarity between two communities by using a phylogentic tree of the features [@Lozupone2010]. Unfortunately, phylogenetic tree reconstruction is computationally expensive and can only be constructed from sequencing data [@Lozupone2010]. 
Therefore, these is a need for a metric that is more generic and can be applied on a variety of sparse omics data. The CSCSomics tool implements the CSCS similarity metric, which was originally developed and applied for metabolomics data to study tree communities [@Sedio2017b]. The algorithm was later optimised with standard library's in both Python and R [@Brejnrod2019]. The algorithm requires a Chemical Structure Similarity (CSS) matrix that is constructed from the sequencing data (i.e. of metagenomics and proteomics) or directly obtained from the cosine scores of GNPS (i.e. of metabolomics) [@Networking2016] (\autoref{fig:cscsomics}). Eventually, the CSCS similarity metric is computed and optimised via a gradient descent algorithm that finds the maximum similarity explained between sample pairs. This leads to higher sample dispersion and may improve the biological interpretation of PCoAs. The \autoref{eq:diff}, is applied to find the maximum similarity explained $\lambda$ given a symmetric matrix $A$ [@Karim2010], where $\theta$ is sampled from a beta distribution that is computed from $A$.

\begin{equation}
\label{eq:diff}
    \frac{\partial \lambda}{\partial \theta} = U_1 \cdot (\frac{\partial A}{\partial \theta}) \cdot U_1^T
\end{equation}

The CSCSomics tool offers a quick approach to improve the similarity explained along the first two principal components of metagenomics, proteomics, metabolomics or custom metrics that are influenced by sparsity and high dimensionality. It also gives the user the option to perform a PERMANOVA statistical test and visualization of PCoA plots before versus after optimisations [@Anderson2017] (\autoref{fig:cscsomics}). 

![Figure 1: Overview of the CSCSomics pipeline, The input are either metagenomics, proteomics or metabolomics data that will be converted into a CSS matrix either via GNPS cosine scoring or Blastn [@cock2009biopython] [@Camacho2009]. The CSCS matrix is computed and optimised. Finally, PCoA and permanova graphs are generated upon metadata specification. \footnotesize{*This step is parallelized via the python multiprocessing module, it can also handle a custom metric that is specified by the user. This means the custom metric will be optimised and the steps beforehand will be skipped.}\label{fig:cscsomics}](figures/CSCSomics.png)

# Acknowledgements

Thanks to Asker Brejnrod for his support during this project.

# References