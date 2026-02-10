# Data Schemas - Type-Safe Data Models
# Production-grade schemas with validation and serialization

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List, Optional, Dict, Any
from enum import Enum
import json


class Gender(Enum):
    """User gender categories."""
    MALE = "male"
    FEMALE = "female"
    OTHER = "other"
    PREFER_NOT_TO_SAY = "prefer_not_to_say"


class AgeGroup(Enum):
    """User age demographics."""
    TEEN = "13-17"
    YOUNG_ADULT = "18-24"
    ADULT = "25-34"
    MIDDLE_AGED = "35-44"
    SENIOR = "45-54"
    ELDERLY = "55+"


class InteractionType(Enum):
    """User-item interaction types with implicit weights."""
    VIEW = ("view", 1.0)
    CLICK = ("click", 2.0)
    LIKE = ("like", 4.0)
    PURCHASE = ("purchase", 5.0)
    
    def __init__(self, action: str, weight: float):
        self.action = action
        self.weight = weight


class ContentCategory(Enum):
    """Item content categories (movie/entertainment domain)."""
    ACTION = "Action"
    COMEDY = "Comedy"
    DRAMA = "Drama"
    HORROR = "Horror"
    ROMANCE = "Romance"
    SCIFI = "Sci-Fi"
    THRILLER = "Thriller"
    DOCUMENTARY = "Documentary"
    ANIMATION = "Animation"
    FANTASY = "Fantasy"


@dataclass
class User:
    """User entity with demographics and preferences.
    
    Attributes:
        user_id: Unique identifier
        username: Display name
        age_group: Demographic segment
        gender: Gender category
        signup_date: Account creation timestamp
        preferred_categories: Top content preferences
        is_premium: Subscription status
    """
    user_id: int
    username: str
    age_group: AgeGroup
    gender: Gender
    signup_date: datetime
    preferred_categories: List[str] = field(default_factory=list)
    is_premium: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for storage."""
        return {
            "user_id": self.user_id,
            "username": self.username,
            "age_group": self.age_group.value,
            "gender": self.gender.value,
            "signup_date": self.signup_date.isoformat(),
            "preferred_categories": self.preferred_categories,
            "is_premium": self.is_premium
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "User":
        """Deserialize from dictionary."""
        return cls(
            user_id=data["user_id"],
            username=data["username"],
            age_group=AgeGroup(data["age_group"]),
            gender=Gender(data["gender"]),
            signup_date=datetime.fromisoformat(data["signup_date"]),
            preferred_categories=data.get("preferred_categories", []),
            is_premium=data.get("is_premium", False)
        )


@dataclass
class Item:
    """Item entity representing content in the catalog.
    
    Attributes:
        item_id: Unique identifier
        title: Display title
        category: Primary content category
        tags: Descriptive tags for content-based filtering
        release_date: Publication/release timestamp
        avg_rating: Aggregate user rating (1-5)
        rating_count: Number of ratings received
        description: Brief content description
        image_url: Thumbnail/poster URL
    """
    item_id: int
    title: str
    category: str
    tags: List[str] = field(default_factory=list)
    release_date: Optional[datetime] = None
    avg_rating: float = 0.0
    rating_count: int = 0
    description: str = ""
    image_url: str = ""
    
    @property
    def freshness_score(self) -> float:
        """Calculate freshness based on release date (0-1 scale)."""
        if not self.release_date:
            return 0.5
        days_old = (datetime.now() - self.release_date).days
        # Exponential decay with 90-day half-life
        return 2 ** (-days_old / 90)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for storage."""
        return {
            "item_id": self.item_id,
            "title": self.title,
            "category": self.category,
            "tags": self.tags,
            "release_date": self.release_date.isoformat() if self.release_date else None,
            "avg_rating": self.avg_rating,
            "rating_count": self.rating_count,
            "description": self.description,
            "image_url": self.image_url
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Item":
        """Deserialize from dictionary."""
        return cls(
            item_id=data["item_id"],
            title=data["title"],
            category=data["category"],
            tags=data.get("tags", []),
            release_date=datetime.fromisoformat(data["release_date"]) if data.get("release_date") else None,
            avg_rating=data.get("avg_rating", 0.0),
            rating_count=data.get("rating_count", 0),
            description=data.get("description", ""),
            image_url=data.get("image_url", "")
        )


@dataclass
class Interaction:
    """User-item interaction event.
    
    Attributes:
        user_id: Acting user
        item_id: Target item
        interaction_type: Type of engagement
        timestamp: When interaction occurred
        duration_seconds: Engagement duration (for views)
        rating: Explicit rating if provided (1-5)
    """
    user_id: int
    item_id: int
    interaction_type: InteractionType
    timestamp: datetime
    duration_seconds: Optional[int] = None
    rating: Optional[float] = None
    
    @property
    def implicit_score(self) -> float:
        """Calculate implicit preference score from interaction."""
        base_score = self.interaction_type.weight
        
        # Boost for longer engagement
        if self.duration_seconds and self.duration_seconds > 60:
            base_score *= min(1.5, 1 + (self.duration_seconds / 300))
        
        # Include explicit rating if available
        if self.rating:
            base_score *= (self.rating / 3.0)  # Normalize around 3
            
        return base_score
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for storage."""
        return {
            "user_id": self.user_id,
            "item_id": self.item_id,
            "interaction_type": self.interaction_type.action,
            "timestamp": self.timestamp.isoformat(),
            "duration_seconds": self.duration_seconds,
            "rating": self.rating
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Interaction":
        """Deserialize from dictionary."""
        interaction_map = {it.action: it for it in InteractionType}
        return cls(
            user_id=data["user_id"],
            item_id=data["item_id"],
            interaction_type=interaction_map[data["interaction_type"]],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            duration_seconds=data.get("duration_seconds"),
            rating=data.get("rating")
        )


@dataclass
class Recommendation:
    """A single recommendation with explanation.
    
    Attributes:
        item: Recommended item
        score: Predicted relevance score
        rank: Position in recommendation list
        explanation: Human-readable reason
        model_source: Which model generated this
        confidence: Model confidence (0-1)
    """
    item: Item
    score: float
    rank: int
    explanation: str
    model_source: str
    confidence: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize for API response."""
        return {
            "item": self.item.to_dict(),
            "score": round(self.score, 4),
            "rank": self.rank,
            "explanation": self.explanation,
            "model_source": self.model_source,
            "confidence": round(self.confidence, 3)
        }
