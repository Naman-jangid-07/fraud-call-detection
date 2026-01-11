"""
Fraud analysis service using Gemini API
"""

from google import genai
from google.genai import types
import logging
from typing import Dict
import os
from pathlib import Path
import sys
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing.text_cleaner import TextCleaner
from src.preprocessing.feature_extractor import FeatureExtractor

logger = logging.getLogger(__name__)


class FraudAnalyzer:
    """Analyze text for fraud indicators using Gemini AI"""
    
    def __init__(self, api_key: str = None):
        """
        Initialize fraud analyzer with Gemini
        
        Args:
            api_key: Gemini API key (or set GEMINI_API_KEY env variable)
        """
        # Get API key
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        
        if not self.api_key:
            raise ValueError("Gemini API key not provided. Set GEMINI_API_KEY environment variable or pass it directly.")
        
        # Initialize Gemini client
        self.client = genai.Client(api_key=self.api_key)
        
        # Initialize text processing
        self.text_cleaner = TextCleaner()
        self.feature_extractor = FeatureExtractor()
        
        logger.info("Fraud analyzer ready with Gemini AI!")
    
    def analyze_text(self, text: str) -> Dict:
        """
        Analyze text for fraud using Gemini
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with analysis results
        """
        # Clean text
        cleaned_text = self.text_cleaner.clean_text(text)
        
        # Extract features for pattern detection
        features = self.feature_extractor.extract_all_features(text)
        
        # Create detailed prompt for Gemini
        prompt = f"""
You are an expert fraud detection AI. Analyze the following phone call conversation and determine if it's fraudulent.

CONVERSATION:
"{text}"

Analyze for these fraud indicators:
1. Urgency tactics (immediate action required, limited time)
2. Financial information requests (bank details, credit cards, SSN)
3. Impersonation (claiming to be from IRS, Microsoft, banks, etc.)
4. Threats or intimidation (arrest, legal action, account suspension)
5. Verification requests (asking to confirm personal information)

Provide your analysis in this EXACT JSON format (no markdown, just pure JSON):
{{
    "is_fraud": true or false,
    "fraud_probability": 0.0 to 1.0,
    "risk_level": "LOW" or "MEDIUM" or "HIGH" or "CRITICAL",
    "detected_patterns": {{
        "urgency_language": true or false,
        "financial_requests": true or false,
        "impersonation": true or false,
        "threats": true or false,
        "verification_request": true or false
    }},
    "reasoning": "Brief explanation of why this is or is not fraud",
    "key_phrases": ["suspicious phrase 1", "suspicious phrase 2"]
}}

Only respond with valid JSON, nothing else.
"""
        
        try:
            # Get Gemini analysis
            response = self.client.models.generate_content(
                model='gemini-2.0-flash-exp',
                contents=prompt
            )
            
            # Parse response
            result_text = response.text.strip()
            
            # Remove markdown code blocks if present
            if result_text.startswith('```'):
                lines = result_text.split('\n')
                result_text = '\n'.join(lines[1:-1]) if len(lines) > 2 else result_text
                if result_text.startswith('json'):
                    result_text = result_text[4:].strip()
            
            # Parse JSON
            try:
                gemini_result = json.loads(result_text)
            except json.JSONDecodeError as je:
                logger.error(f"JSON decode error: {je}")
                logger.error(f"Response text: {result_text}")
                # Fallback to basic analysis
                return self._fallback_analysis(text, features)
            
            # Combine with our feature extraction
            fraud_prob = float(gemini_result.get('fraud_probability', 0.5))
            
            # Ensure fraud_prob is between 0 and 1
            fraud_prob = max(0.0, min(1.0, fraud_prob))
            
            # Override patterns with our feature extraction if detected
            detected_patterns = gemini_result.get('detected_patterns', {})
            detected_patterns['urgency_language'] = detected_patterns.get('urgency_language', False) or features.get('has_urgency', 0) > 0
            detected_patterns['financial_requests'] = detected_patterns.get('financial_requests', False) or features.get('has_financial', 0) > 0
            detected_patterns['impersonation'] = detected_patterns.get('impersonation', False) or features.get('has_impersonation', 0) > 0
            detected_patterns['threats'] = detected_patterns.get('threats', False) or features.get('has_threats', 0) > 0
            detected_patterns['verification_request'] = detected_patterns.get('verification_request', False) or features.get('has_verification', 0) > 0
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                fraud_prob, 
                detected_patterns,
                gemini_result.get('reasoning', ''),
                gemini_result.get('key_phrases', [])
            )
            
            return {
                "is_fraud": bool(fraud_prob > 0.5),
                "fraud_probability": round(fraud_prob, 4),
                "legitimate_probability": round(1 - fraud_prob, 4),
                "risk_level": gemini_result.get('risk_level', 'UNKNOWN'),
                "detected_patterns": detected_patterns,
                "recommendations": recommendations,
                "reasoning": gemini_result.get('reasoning', ''),
                "key_phrases": gemini_result.get('key_phrases', [])
            }
            
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            logger.exception("Full error traceback:")
            return self._fallback_analysis(text, features)
    
    def _fallback_analysis(self, text: str, features: Dict) -> Dict:
        """Fallback analysis if Gemini fails"""
        
        logger.info("Using fallback analysis")
        
        # Simple rule-based scoring
        patterns = {
            'urgency_language': features.get('has_urgency', 0) > 0,
            'financial_requests': features.get('has_financial', 0) > 0,
            'impersonation': features.get('has_impersonation', 0) > 0,
            'threats': features.get('has_threats', 0) > 0,
            'verification_request': features.get('has_verification', 0) > 0
        }
        
        score = sum(patterns.values()) / len(patterns)
        
        if score >= 0.6:
            risk_level = "HIGH"
        elif score >= 0.4:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        recommendations = self._generate_recommendations(score, patterns, "Fallback analysis used", [])
        
        return {
            "is_fraud": bool(score > 0.5),
            "fraud_probability": round(score, 4),
            "legitimate_probability": round(1 - score, 4),
            "risk_level": risk_level,
            "detected_patterns": patterns,
            "recommendations": recommendations,
            "reasoning": "Basic pattern-based analysis (Gemini unavailable)",
            "key_phrases": []
        }
    
    def _generate_recommendations(self, fraud_prob: float, patterns: Dict, 
                                  reasoning: str, key_phrases: list) -> list:
        """Generate user recommendations based on analysis"""
        recommendations = []
        
        if fraud_prob > 0.7:
            recommendations.append("🚨 HIGH RISK: This call exhibits strong fraud indicators")
            recommendations.append("⚠️ Do NOT share any personal or financial information")
            recommendations.append("📞 Hang up immediately and verify through official channels")
        
        if patterns.get('urgency_language'):
            recommendations.append("⏰ Caller is using urgency tactics to pressure you")
        
        if patterns.get('financial_requests'):
            recommendations.append("💳 Caller is requesting financial information - this is suspicious")
        
        if patterns.get('impersonation'):
            recommendations.append("👤 Possible impersonation of official organization")
        
        if patterns.get('threats'):
            recommendations.append("⚖️ Caller is using threats or intimidation - common scam tactic")
        
        if patterns.get('verification_request'):
            recommendations.append("🔐 Never verify your identity by sharing sensitive information")
        
        if fraud_prob < 0.3:
            recommendations.append("✅ This call appears legitimate, but always stay cautious")
        
        if reasoning:
            recommendations.append(f"📝 AI Analysis: {reasoning}")
        
        if key_phrases:
            key_phrase_text = ", ".join(key_phrases[:3])
            recommendations.append(f"🔑 Suspicious phrases detected: {key_phrase_text}")
        
        return recommendations