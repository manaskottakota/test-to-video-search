import React, { useState, useEffect } from 'react';
import { Search, AlertTriangle, CheckCircle, XCircle, HelpCircle, Globe, FileText, BarChart3, Shield, Clock, Users, TrendingUp, Download, ThumbsUp, ThumbsDown } from 'lucide-react';

const MisinformationDetectorApp = () => {
  const [currentStep, setCurrentStep] = useState('input');
  const [inputType, setInputType] = useState('url');
  const [inputValue, setInputValue] = useState('');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResults, setAnalysisResults] = useState(null);
  const [selectedClaim, setSelectedClaim] = useState(null);
  const [apiUrl, setApiUrl] = useState('http://localhost:8000');

  // Call backend API
  const analyzeContent = async () => {
    setIsAnalyzing(true);
    setCurrentStep('analyzing');
    
    try {
      const requestBody = inputType === 'url' 
        ? { url: inputValue, include_evidence: true, include_propagation: true }
        : { text: inputValue, include_evidence: true };

      const response = await fetch(`${apiUrl}/analyze`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody)
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      setAnalysisResults(result);
      setCurrentStep('results');
      
    } catch (error) {
      console.error('Analysis failed:', error);
      alert('Analysis failed. Please check if the backend is running on ' + apiUrl);
      setCurrentStep('input');
    } finally {
      setIsAnalyzing(false);
    }
  };

  // [Rest of the React component code from previous artifact]
  // InputScreen, AnalyzingScreen, ResultsScreen components...

  return (
    <div className="min-h-screen">
      {/* Add API URL configuration */}
      <div className="fixed top-4 right-4 z-50">
        <input
          type="text"
          value={apiUrl}
          onChange={(e) => setApiUrl(e.target.value)}
          placeholder="API URL"
          className="px-3 py-1 text-xs border rounded bg-white/90"
        />
      </div>
      
      {currentStep === 'input' && <InputScreen />}
      {currentStep === 'analyzing' && <AnalyzingScreen />}
      {currentStep === 'results' && <ResultsScreen />}
    </div>
  );
};

export default MisinformationDetectorApp;