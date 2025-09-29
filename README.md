# TruthLens: Misinformation Detection Platform

TruthLens is a web-based misinformation detection system that leverages advanced NLP and machine learning techniques to assess the credibility of online content. The platform analyzes either text or URLs and provides insights including factual accuracy, source credibility, content patterns, and propagation risk.

## 🧾 Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Architecture Overview](#architecture-overview)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Dependencies](#dependencies)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)
- [Contributors](#contributors)
- [License](#license)

## 🧠 Introduction

TruthLens is a full-stack application designed to detect and analyze misinformation in digital content. It offers both a React-based frontend interface and a FastAPI backend with machine learning pipelines for:

- Claim detection
- Fact verification
- Credibility scoring
- Content analysis
- Propagation forecasting

## ✨ Features

- 🔍 **Text & URL input support**
- 📊 **Factual accuracy scoring** using semantic similarity and stance detection
- 📰 **Content analysis** (readability, emotional tone, clickbait detection)
- 🔗 **Source credibility scoring** based on multiple features
- 📈 **Propagation analysis** (bot likelihood, velocity, etc.)
- 📋 **Interactive UI** with real-time API feedback
- 🔌 **RESTful API** compatible with external integrations

## 🏗️ Architecture Overview

Client (Browser)
   |
   |  User inputs text or URL
   ↓
Frontend (React)
 - File: MisinformationDetector.jsx
 - Renders UI and handles user input
 - Sends POST requests to FastAPI backend at /analyze
   |
   ↓
Backend (FastAPI - main.py)
 - Endpoint: /analyze
 - Handles:
     • Claim detection (transformers)
     • Semantic similarity (sentence-transformers)
     • Stance detection (BART, etc.)
     • Content & source analysis
     • Optional propagation modeling
   |
   ↓
ML Pipelines
 - HuggingFace Transformers (claim/stance classification)
 - Scikit-learn (credibility models)
 - TensorFlow (deep learning components)
 - SpaCy & NLTK (linguistic features)
 - Knowledge base for fact comparison

   |
   ↓
JSON Response
 - Verdicts, scores, evidence, propagation data

   ↑
Frontend (React)
 - Displays results interactively
