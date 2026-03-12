# Standard library imports
import re
import json
import datetime
import random
import uuid
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum

# Third-party imports
import nltk
import spacy
from fuzzywuzzy import fuzz


# ============================================================================
# ENUMS
# ============================================================================

class SeverityLevel(Enum):
    """Severity levels for symptoms"""
    NONE = "none"
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"
    EMERGENCY = "emergency"

class RiskCategory(Enum):
    """Risk categories for patients"""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class PatientProfile:
    """Stores patient information"""
    age: Optional[int] = None
    risk_factors: List[str] = field(default_factory=list)
    temperature: Optional[float] = None
    pregnancy_status: Optional[bool] = None
    chronic_conditions: List[str] = field(default_factory=list)

@dataclass
class Symptom:
    """Stores symptom information"""
    name: str
    reported: bool = True
    severity: Optional[int] = None  # 1-10 scale
    duration_days: Optional[int] = None
    first_mentioned: str = field(default_factory=lambda: datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

@dataclass
class ConversationSession:
    """Stores entire conversation context"""
    session_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    patient: PatientProfile = field(default_factory=PatientProfile)
    symptoms: Dict[str, Symptom] = field(default_factory=dict)
    emergency_signs: List[str] = field(default_factory=list)
    message_history: List[Dict] = field(default_factory=list)
    start_time: str = field(default_factory=lambda: datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    last_interaction: str = field(default_factory=lambda: datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


# ============================================================================
# MAIN CHATBOT CLASS
# ============================================================================

class FluChatbot:
    """Main Flu AI Chatbot Class"""
    
    def __init__(self):
        """Initialize the chatbot"""
        self.name = "FluBuddy"
        self.version = "2.0"
        
        # Create a new conversation session
        self.session = ConversationSession()
        
        # Load NLP model
        print("Loading NLP model...")
        try:
            self.nlp = spacy.load("en_core_web_sm")
            print("✅ NLP model loaded")
        except:
            print("Downloading NLP model...")
            import subprocess
            subprocess.run(["python3", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")
        
        print(f"✅ FluChatbot v{self.version} initialized")
        print(f"📝 Session ID: {self.session.session_id}")