# Advanced Fact-Checking Backend
# Dependencies: pip install fastapi uvicorn tensorflow sentence-transformers scikit-learn transformers torch spacy newspaper3k requests beautifulsoup4 pandas numpy nltk textstat

import asyncio
import logging
import re
import json
import hashlib
import aiohttp
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from urllib.parse import urlparse, urljoin
import warnings
warnings.filterwarnings("ignore")

# FastAPI
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, HttpUrl

# ML & NLP Libraries
import tensorflow as tf
from sentence_transformers import SentenceTransformer, util
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import spacy
import nltk
from nltk.tokenize import sent_tokenize
from textstat import flesch_reading_ease, automated_readability_index

# Web scraping
from newspaper import Article
import requests
from bs4 import BeautifulSoup

# Database (SQLAlchemy for production)
import sqlite3
from contextlib import asynccontextmanager

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="VERITAS Fact-Checking API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# DATA MODELS
# ============================================================================

class AnalysisRequest(BaseModel):
    text: Optional[str] = None
    url: Optional[HttpUrl] = None
    language: str = "en"
    include_evidence: bool = True
    include_propagation: bool = True

@dataclass
class Claim:
    text: str
    sentence_id: int
    start_pos: int
    end_pos: int
    claim_type: str
    confidence: float
    verdict: str  # supported, contradicted, ambiguous
    evidence: List[Dict]
    reasoning: str

@dataclass
class SourceCredibility:
    domain: str
    credibility_score: float
    reputation: str
    bias_score: float
    fact_check_history: Dict
    editorial_standards: float
    transparency_score: float

@dataclass
class ContentAnalysis:
    readability_score: float
    sentiment_score: float
    emotional_language_score: float
    sensational_words_count: int
    hedging_words_count: int
    certainty_score: float
    urgency_indicators: int
    clickbait_score: float

@dataclass
class PropagationAnalysis:
    first_seen: datetime
    spread_velocity: float
    bot_likelihood: float
    viral_coefficient: float
    source_diversity: float
    geographic_spread: Dict

@dataclass
class FactCheckResult:
    overall_credibility_score: float
    risk_level: str
    claims: List[Claim]
    source_credibility: SourceCredibility
    content_analysis: ContentAnalysis
    propagation_analysis: Optional[PropagationAnalysis]
    processing_time: float
    model_version: str
    confidence_interval: Tuple[float, float]

# ============================================================================
# CORE ML MODELS INITIALIZATION
# ============================================================================

class FactCheckingModels:
    def __init__(self):
        logger.info("Initializing ML models...")
        
        # Sentence transformer for semantic similarity
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Claim detection model (using transformer)
        self.claim_detector = pipeline(
            "text-classification",
            model="microsoft/DialoGPT-medium",
            return_all_scores=True
        )
        
        # Stance detection model
        self.stance_model = pipeline(
            "text-classification",
            model="facebook/bart-large-mnli",
            return_all_scores=True
        )
        
        # Load spaCy for NER and linguistic features
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
        
        # Initialize traditional ML models
        self.credibility_model = self._init_credibility_model()
        self.content_analyzer = self._init_content_analyzer()
        
        # TF-IDF vectorizer for text features
        self.tfidf = TfidfVectorizer(max_features=10000, stop_words='english')
        
        # Knowledge base embeddings (mock for demo)
        self.knowledge_base = self._load_knowledge_base()
        
        logger.info("All ML models initialized successfully!")
    
    def _init_credibility_model(self) -> RandomForestClassifier:
        """Initialize source credibility prediction model"""
        # This would be trained on real data in production
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # Mock training data (in production, use real labeled data)
        X_mock = np.random.rand(1000, 15)  # 15 features
        y_mock = np.random.randint(0, 3, 1000)  # 3 credibility levels
        model.fit(X_mock, y_mock)
        
        return model
    
    def _init_content_analyzer(self) -> GradientBoostingClassifier:
        """Initialize content pattern analyzer"""
        model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        
        # Mock training for content patterns
        X_mock = np.random.rand(1000, 20)
        y_mock = np.random.randint(0, 2, 1000)  # Binary: reliable/unreliable
        model.fit(X_mock, y_mock)
        
        return model
    
    def _load_knowledge_base(self) -> Dict:
        """Load and prepare knowledge base for fact verification"""
        # In production, this would load from a vector database like Milvus
        knowledge_base = {
            'climate_change': [
                "Global temperatures have risen by approximately 1.1°C since pre-industrial times according to IPCC.",
                "Sea levels are rising at 3.3 mm per year based on satellite measurements.",
                "Arctic sea ice is declining at 13% per decade according to NSIDC data."
            ],
            'covid19': [
                "mRNA vaccines show 95% efficacy in preventing severe COVID-19 according to clinical trials.",
                "Social distancing reduces transmission by 40-60% according to epidemiological studies.",
                "Long COVID affects 10-30% of infected individuals according to medical research."
            ],
            'technology': [
                "Current battery energy density is around 250-300 Wh/kg for lithium-ion batteries.",
                "Quantum computers require temperatures near absolute zero to function.",
                "5G networks operate at frequencies between 24-40 GHz for millimeter wave bands."
            ]
        }
        
        # Create embeddings for knowledge base
        kb_embeddings = {}
        for category, facts in knowledge_base.items():
            kb_embeddings[category] = {
                'texts': facts,
                'embeddings': self.sentence_model.encode(facts)
            }
        
        return kb_embeddings

# Initialize global models instance
models = FactCheckingModels()

# [Continue with rest of the backend code - ContentExtractor, ClaimExtractor, etc.]
# Due to length limits, I'm showing the structure. The full backend code is in the previous artifact.

if __name__ == "__main__":
    import uvicorn
    
    # Production configuration
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=2,
        log_level="info",
        access_log=True
    )