# Table Retrieval Task

This repository contains my PhD implementations. We focus on matching unstructured data (e.g., news articles) and semi-structured data (e.g., web tables). Specifically, our work addresses the table retrieval task: given an article and a set of web tables, the goal is to find a table most relevant to an article.

## Article Aspects

We consider a news article is composed by two aspects: title and body. The title briefly describes the news and it has generally 10 or 12 words. The body is a large text about the news-content. Furthermore, both aspects are presented in natural language.

## Table Aspects

Unlike articles, the table is a semi-structured data arranged by rows and columns. The table aspects include headers, body and caption. The header indicates the column proprieties and helps to describe its meaning.  The body is composed by cells and contains all table content. Finally, the caption summarizes the table subject. Just as articles, the table aspects also are expressed in natural language but some cells can contain numbers or dates. Moreover, we consider two more table aspects: page title and page section title. These aspects are collected from where the table was extracted.
