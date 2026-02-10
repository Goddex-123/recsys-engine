# Realistic Data Generator - Production-Quality Synthetic Data
# Simulates authentic user behavior patterns with proper statistical distributions

import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import os
import json

from .schemas import (
    User, Item, Interaction, InteractionType,
    Gender, AgeGroup, ContentCategory
)

# Seed for reproducibility
np.random.seed(42)
random.seed(42)


# Realistic content data for movie/entertainment domain
MOVIE_TITLES = {
    "Action": [
        "Shadow Strike", "Velocity X", "Iron Protocol", "Edge of Fury", "Dark Horizon",
        "Rogue Thunder", "Silent Operative", "Final Countdown", "Steel Fury", "Night Hunter",
        "Crimson Force", "Apex Predator", "Lethal Zone", "Storm Command", "Nuclear Dawn"
    ],
    "Comedy": [
        "The Mishap", "Accidentally Yours", "Office Chaos", "Best Laid Plans", "The Mix-Up",
        "Dads on Duty", "Roommate Rules", "Wedding Disasters", "Startup Madness", "The Scheme",
        "Vacation Mode", "Family Reunion", "Dating Disasters", "Promotion Panic", "Neighbors"
    ],
    "Drama": [
        "Echoes of Silence", "The Last Letter", "Broken Promises", "Crossroads", "Redemption",
        "The Distance Between", "Fading Lights", "Unspoken Words", "The Weight of Days", "Homecoming",
        "Second Chances", "The Long Road", "Parallel Lives", "Silent Witness", "The Choice"
    ],
    "Horror": [
        "The Hollow", "Whispers in the Dark", "Crimson Night", "The Forgotten", "Shadow House",
        "Blood Moon Rising", "The Lurking", "Nightmare Corridor", "The Possession", "Dead End",
        "The Séance", "Midnight Terror", "Cursed Ground", "The Haunting", "Fear Factory"
    ],
    "Romance": [
        "Love in Paris", "Second Look", "The Proposal", "Meant to Be", "Summer Hearts",
        "Finding You", "The Wedding Planner", "Destiny Calls", "Perfect Match", "Hearts Aligned",
        "Love Letters", "The Last Dance", "Between Us", "Falling for You", "True Colors"
    ],
    "Sci-Fi": [
        "Quantum Drift", "Beyond the Stars", "Neural Link", "The Colony", "Time Fracture",
        "Synthetic Dawn", "Asteroid X", "Digital Souls", "The Simulation", "Warp Speed",
        "Mars Rising", "AI Awakening", "The Void", "Parallel Worlds", "Cyborg Revolution"
    ],
    "Thriller": [
        "The Witness", "Dead Drop", "Conspiracy", "The Informant", "Deadly Game",
        "No Exit", "The Setup", "Hidden Agenda", "Trust No One", "The Confession",
        "Double Cross", "The Pursuit", "Vanishing Act", "The Negotiator", "The Inside Man"
    ],
    "Documentary": [
        "Planet Earth: Unveiled", "The Human Journey", "Tech Giants", "Ocean Mysteries", "Wildlife Warriors",
        "History Revealed", "Space Exploration", "Climate Chronicles", "Ancient Civilizations", "Food Revolution",
        "Music Evolution", "Sports Legends", "Medical Breakthroughs", "Art of Innovation", "Nature's Fury"
    ],
    "Animation": [
        "The Magic Kingdom", "Robot Friends", "Dragon Quest", "Undersea Adventure", "Space Cadets",
        "Jungle Tales", "Hero Academy", "Pixel World", "Monster Academy", "Flying High",
        "Ice Age Adventures", "Dinosaur Land", "Super Pets", "Fantasy Forest", "Future Kids"
    ],
    "Fantasy": [
        "The Dragon King", "Realm of Shadows", "Magic Awakens", "The Chosen One", "Kingdom Fall",
        "Enchanted Forest", "The Dark Sorcerer", "Crystal Quest", "Legends Reborn", "The Portal",
        "Mystic Knights", "The Prophecy", "Elemental War", "The Last Wizard", "Dark Throne"
    ]
}

TAGS_BY_CATEGORY = {
    "Action": ["explosive", "fast-paced", "stunts", "military", "combat", "chase"],
    "Comedy": ["funny", "witty", "slapstick", "romantic-comedy", "satire", "parody"],
    "Drama": ["emotional", "character-driven", "family", "biographical", "intense", "award-winning"],
    "Horror": ["scary", "supernatural", "psychological", "slasher", "gore", "suspenseful"],
    "Romance": ["love-story", "heartwarming", "tear-jerker", "dating", "wedding", "second-chance"],
    "Sci-Fi": ["futuristic", "space", "technology", "dystopian", "time-travel", "aliens"],
    "Thriller": ["suspense", "crime", "mystery", "psychological", "conspiracy", "heist"],
    "Documentary": ["informative", "educational", "nature", "true-story", "inspiring", "eye-opening"],
    "Animation": ["family-friendly", "colorful", "adventure", "fantasy", "musical", "kids"],
    "Fantasy": ["magical", "epic", "mythical", "adventure", "quest", "sword-and-sorcery"]
}

FIRST_NAMES = [
    "Emma", "Liam", "Olivia", "Noah", "Ava", "Oliver", "Sophia", "Elijah", "Isabella", "Lucas",
    "Mia", "Mason", "Charlotte", "James", "Amelia", "Benjamin", "Harper", "Alexander", "Evelyn", "Henry",
    "Luna", "Sebastian", "Camila", "Jack", "Aria", "Daniel", "Scarlett", "Michael", "Madison", "Owen",
    "Layla", "Ethan", "Chloe", "Aiden", "Penelope", "Matthew", "Riley", "William", "Zoey", "Joseph"
]

LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez",
    "Anderson", "Taylor", "Thomas", "Moore", "Jackson", "Martin", "Lee", "Thompson", "White", "Harris",
    "Clark", "Lewis", "Robinson", "Walker", "Hall", "Young", "King", "Wright", "Scott", "Green"
]


class DataGenerator:
    """Generates realistic synthetic data for recommendation system testing.
    
    Uses power-law distributions to create authentic patterns:
    - Popular items receive disproportionate attention
    - Active users generate most interactions
    - Temporal patterns reflect real viewing habits
    """
    
    def __init__(
        self,
        n_users: int = 10_000,
        n_items: int = 5_000,
        n_interactions: int = 500_000,
        history_days: int = 365,
        random_seed: int = 42
    ):
        self.n_users = n_users
        self.n_items = n_items
        self.n_interactions = n_interactions
        self.history_days = history_days
        self.random_seed = random_seed
        
        np.random.seed(random_seed)
        random.seed(random_seed)
        
        self.users: List[User] = []
        self.items: List[Item] = []
        self.interactions: List[Interaction] = []
        
        # Pre-computed distributions
        self._user_activity_weights: Optional[np.ndarray] = None
        self._item_popularity_weights: Optional[np.ndarray] = None
        
    def _generate_power_law_weights(self, n: int, alpha: float = 1.5) -> np.ndarray:
        """Generate power-law distributed weights (Pareto principle)."""
        # Zipf-like distribution
        ranks = np.arange(1, n + 1)
        weights = 1.0 / (ranks ** alpha)
        # Normalize to probabilities
        return weights / weights.sum()
    
    def generate_users(self) -> List[User]:
        """Generate diverse user population with realistic demographics."""
        print(f"[USER] Generating {self.n_users:,} users...")
        
        self.users = []
        categories = [c.value for c in ContentCategory]
        
        # Age distribution weights (realistic population)
        age_weights = {
            AgeGroup.TEEN: 0.08,
            AgeGroup.YOUNG_ADULT: 0.25,
            AgeGroup.ADULT: 0.30,
            AgeGroup.MIDDLE_AGED: 0.20,
            AgeGroup.SENIOR: 0.12,
            AgeGroup.ELDERLY: 0.05
        }
        
        for i in range(self.n_users):
            # Demographics
            age_group = random.choices(
                list(age_weights.keys()),
                weights=list(age_weights.values())
            )[0]
            
            gender = random.choices(
                [Gender.MALE, Gender.FEMALE, Gender.OTHER, Gender.PREFER_NOT_TO_SAY],
                weights=[0.48, 0.48, 0.02, 0.02]
            )[0]
            
            # Signup date (power-law: more recent signups)
            days_ago = int(np.random.pareto(1.5) * 30) % self.history_days
            signup_date = datetime.now() - timedelta(days=days_ago)
            
            # Preferences (2-4 preferred categories)
            n_prefs = random.randint(2, 4)
            preferred_categories = random.sample(categories, n_prefs)
            
            # Premium status (10% of users)
            is_premium = random.random() < 0.10
            
            user = User(
                user_id=i,
                username=f"{random.choice(FIRST_NAMES)}_{random.choice(LAST_NAMES)}_{i}",
                age_group=age_group,
                gender=gender,
                signup_date=signup_date,
                preferred_categories=preferred_categories,
                is_premium=is_premium
            )
            self.users.append(user)
        
        # Pre-compute activity weights
        self._user_activity_weights = self._generate_power_law_weights(
            self.n_users, alpha=1.2
        )
        
        print(f"   [OK] Generated {len(self.users):,} users")
        return self.users
    
    def generate_items(self) -> List[Item]:
        """Generate item catalog with diverse content."""
        print(f"[ITEM] Generating {self.n_items:,} items...")
        
        self.items = []
        categories = list(MOVIE_TITLES.keys())
        
        for i in range(self.n_items):
            category = random.choice(categories)
            
            # Title generation
            base_titles = MOVIE_TITLES[category]
            title = random.choice(base_titles)
            
            # Add suffix for uniqueness
            suffix_type = random.choice(["number", "subtitle", "year", "none"])
            if suffix_type == "number":
                title = f"{title} {random.randint(2, 5)}"
            elif suffix_type == "subtitle":
                subtitles = ["Reloaded", "Origins", "Legacy", "Returns", "Rises", "Unleashed"]
                title = f"{title}: {random.choice(subtitles)}"
            elif suffix_type == "year":
                title = f"{title} ({random.randint(2018, 2024)})"
            
            # Tags
            category_tags = TAGS_BY_CATEGORY[category]
            tags = random.sample(category_tags, random.randint(2, 4))
            
            # Release date (weighted towards recent)
            days_ago = int(np.random.exponential(180)) % (self.history_days * 2)
            release_date = datetime.now() - timedelta(days=days_ago)
            
            # Rating (normal distribution around 3.5)
            avg_rating = max(1.0, min(5.0, np.random.normal(3.5, 0.8)))
            rating_count = int(np.random.pareto(1.5) * 100) + 1
            
            item = Item(
                item_id=i,
                title=f"{title}",
                category=category,
                tags=tags,
                release_date=release_date,
                avg_rating=round(avg_rating, 1),
                rating_count=rating_count,
                description=f"A captivating {category.lower()} experience featuring {', '.join(tags[:2])} elements.",
                image_url=f"https://picsum.photos/seed/{i}/300/450"
            )
            self.items.append(item)
        
        # Pre-compute popularity weights
        self._item_popularity_weights = self._generate_power_law_weights(
            self.n_items, alpha=1.5
        )
        
        print(f"   [OK] Generated {len(self.items):,} items")
        return self.items
    
    def generate_interactions(self) -> List[Interaction]:
        """Generate realistic user-item interactions with temporal patterns."""
        print(f"[FAST] Generating {self.n_interactions:,} interactions...")
        
        if not self.users or not self.items:
            raise ValueError("Generate users and items first")
        
        self.interactions = []
        
        # Interaction type weights
        interaction_weights = [0.50, 0.25, 0.15, 0.10]  # view, click, like, purchase
        interaction_types = list(InteractionType)
        
        # Build user preferences matrix for correlation
        user_prefs = {u.user_id: set(u.preferred_categories) for u in self.users}
        item_cats = {it.item_id: it.category for it in self.items}
        
        for _ in range(self.n_interactions):
            # Select user (power-law: some users more active)
            user_idx = np.random.choice(
                self.n_users,
                p=self._user_activity_weights
            )
            user = self.users[user_idx]
            
            # Select item (biased towards popularity AND user preferences)
            item_weights = self._item_popularity_weights.copy()
            
            # Boost items matching user preferences (2x for preferred categories)
            for i, item in enumerate(self.items):
                if item.category in user_prefs[user.user_id]:
                    item_weights[i] *= 2.0
            
            # Renormalize
            item_weights = item_weights / item_weights.sum()
            
            item_idx = np.random.choice(self.n_items, p=item_weights)
            item = self.items[item_idx]
            
            # Interaction type (weighted)
            interaction_type = random.choices(
                interaction_types,
                weights=interaction_weights
            )[0]
            
            # Timestamp (exponential decay: more recent = more likely)
            days_ago = int(np.random.exponential(60)) % self.history_days
            hours_offset = random.randint(0, 23)
            minutes_offset = random.randint(0, 59)
            timestamp = datetime.now() - timedelta(
                days=days_ago,
                hours=hours_offset,
                minutes=minutes_offset
            )
            
            # Duration for views (longer for preferred content)
            duration = None
            if interaction_type == InteractionType.VIEW:
                base_duration = random.randint(30, 180)
                if item.category in user_prefs[user.user_id]:
                    base_duration = int(base_duration * random.uniform(1.2, 2.0))
                duration = base_duration
            
            # Rating (more likely for purchases and likes)
            rating = None
            if interaction_type in [InteractionType.LIKE, InteractionType.PURCHASE]:
                if random.random() < 0.3:
                    # Positive bias for interactions
                    rating = round(random.uniform(3.5, 5.0), 1)
            
            interaction = Interaction(
                user_id=user.user_id,
                item_id=item.item_id,
                interaction_type=interaction_type,
                timestamp=timestamp,
                duration_seconds=duration,
                rating=rating
            )
            self.interactions.append(interaction)
        
        # Sort by timestamp
        self.interactions.sort(key=lambda x: x.timestamp)
        
        print(f"   [OK] Generated {len(self.interactions):,} interactions")
        return self.interactions
    
    def generate_all(self) -> Tuple[List[User], List[Item], List[Interaction]]:
        """Generate complete dataset."""
        print("\n" + "="*60)
        print("[START] GENERATING SYNTHETIC RECOMMENDATION DATA")
        print("="*60 + "\n")
        
        users = self.generate_users()
        items = self.generate_items()
        interactions = self.generate_interactions()
        
        print("\n" + "="*60)
        print("[DONE] DATA GENERATION COMPLETE")
        print(f"   Users:        {len(users):,}")
        print(f"   Items:        {len(items):,}")
        print(f"   Interactions: {len(interactions):,}")
        print("="*60 + "\n")
        
        return users, items, interactions
    
    def to_dataframes(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Convert generated data to pandas DataFrames."""
        users_df = pd.DataFrame([u.to_dict() for u in self.users])
        items_df = pd.DataFrame([it.to_dict() for it in self.items])
        interactions_df = pd.DataFrame([i.to_dict() for i in self.interactions])
        
        return users_df, items_df, interactions_df
    
    def save_to_csv(self, output_dir: str = "generated_data") -> None:
        """Save generated data to CSV files."""
        os.makedirs(output_dir, exist_ok=True)
        
        users_df, items_df, interactions_df = self.to_dataframes()
        
        users_df.to_csv(os.path.join(output_dir, "users.csv"), index=False)
        items_df.to_csv(os.path.join(output_dir, "items.csv"), index=False)
        interactions_df.to_csv(os.path.join(output_dir, "interactions.csv"), index=False)
        
        print(f"[FILE] Data saved to {output_dir}/")
    
    def get_statistics(self) -> Dict:
        """Calculate dataset statistics for validation."""
        stats = {
            "n_users": len(self.users),
            "n_items": len(self.items),
            "n_interactions": len(self.interactions),
            "sparsity": 1 - (len(self.interactions) / (len(self.users) * len(self.items))),
            "avg_interactions_per_user": len(self.interactions) / max(1, len(self.users)),
            "avg_interactions_per_item": len(self.interactions) / max(1, len(self.items)),
        }
        
        if self.interactions:
            interaction_counts = {}
            for i in self.interactions:
                key = i.interaction_type.action
                interaction_counts[key] = interaction_counts.get(key, 0) + 1
            stats["interaction_distribution"] = interaction_counts
        
        return stats
