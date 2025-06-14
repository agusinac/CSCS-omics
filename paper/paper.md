---
title: 'Incorporating feature similarity in omics analysis for improved modelling'
tags:
  - Python
  - omics
authors:
  - name: Alem Gusinac
    orcid: 0009-0006-1896-4176
    equal-contrib: true
    affiliation: "1" # (Multiple affiliations must be quoted)
affiliations: # Need to check if I need to mention DTU health tech ..
 - name: Lyman Spitzer, Jr. Fellow, Princeton University, United States
   index: 1
   ror: 00hx57361
 - name: Institution Name, Country
   index: 2
 - name: Independent Researcher, Country
   index: 3
date: 30 June 2025
bibliography: paper.bib
---

# Summary

Omics is the study of many genes, proteins or metabolites in a system. Omics analysis is compositional, meaning it represents the relative information from the absolute. Omics data may suffer of high dimensionality and sparsity that complicates downstream processes, causing overfitting and bias. Common dimensional reduction methods, such as principal coordinate analysis (PCoA), are often poor in handling sparse and high-dimensional data, making it challenging to interpret the results due to increased susceptibility to noise and outliers. In this work, the CSCSomics tool is developed that is capable to reduce the dimensionality of sparse data sets of metagenomics, proteomics and metabolomics data, in turn increasing the biological interpretation of PCoA (\autoref{fig:cscsomics}). The CSCSomics is a generalized tool that applies the Chemical Structural and Compositional Similarity (CSCS) metric on a variety of omics data, performs eigenvalue optimization and statistical visualization of the results. 

![Figure 1: Overview of the CSCSomics pipeline, The input are either metagenomics, proteomics or metabolomics data that will be converted into a CSS matrix either via GNPS cosine scoring or Blastn. The CSCS metric requires the counts file for CSCS matrix construction and optimization. Finally, PCoA and permanova graphs are generated upon metadata specification. \footnotesize{*This step is parallelized via multiprocessing module, it can also handle a custom metric that is specified by the user. This means the custom metric will be optimized and the steps beforehand will be skipped.}\label{fig:cscsomics}](figures/CSCSomics.png)

# Statement of need
All the three big omics; metagenomics, proteomics and metabolomics suffer from sparsity and high dimensionality \cite{Misra2019}. Sparse data is characterized by the absence of certain compounds (DNA, protein or metabolite) when analyzing multiple samples \cite{Ronan2016}. These absent compounds are denoted by a zero, this can be caused due to their absence in the sample or their low abundance, which puts them below the detection limit of the detector \cite{Busato2023}. Sparse omics is highly compositional and dimensional reduction methods, such as Principal Coordinate Analysis (PCoA) are often poor in handling sparse and high-dimensional data \cite{Martino2019}\cite{Dollhopf2001}. This leads to challenges in interpretation and increased sensitivity to noise and outliers \cite{Dollhopf2001}.

In the present, a variety of statistical metrics exist to try to explain the compositional data. The Weighted UniFrac metric has been widely used to capture the dissimilarity between two communities by using a phylogentic tree of the features \cite{Lozupone2010}. Unfortunately, phylogenetic tree reconstruction is computationally expensive and time consuming. The CSCSomics tool uses the CSCS metric \cite{Sedio2017b}, which incorporates relative abundances and feature similarity from either sequencing format (FASTA) or cosine scores from GNPS (i.e. metabolomics) \cite{Networking2016}. 

<!-- Explain how to use the tool & math behind the optimisation step, namely the equation, citation and weight sampling -->

# Acknowledgements

We acknowledge contributions from Brigitta Sipocz, Syrtis Major, and Semyeong
Oh, and support from Kathryn Johnston during the genesis of this project.

# References