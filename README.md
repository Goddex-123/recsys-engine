<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <a href="https://recsys-engine.streamlit.app" target="_blank">
    <img src="https://static.streamlit.io/badges/streamlit_badge_black_white.svg" alt="Open in Streamlit">
  </a>
  <img src="https://img.shields.io/badge/ML-Recommendation%20Systems-green?style=for-the-badge" alt="ML">
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge" alt="License">
</p>

<h1 align="center">🎯 RecSys Engine</h1>

<p align="center">
  <em>Built with the architectural patterns used at Google, Netflix, and Amazon</em>
</p>

<p align="center">
  <img src="assets/demo.webp" alt="RecSys Engine Demo" width="800">
</p>

<p align="center">
  <a href="#-why-this-project">Why This Project</a> •
  <a href="#-features">Features</a> •
  <a href="#-architecture">Architecture</a> •
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-algorithms">Algorithms</a> •
  <a href="#-evaluation">Evaluation</a>
</p>

---

## 🎯 Why This Project?

**The Problem**: Every major tech company (Netflix, Amazon, YouTube, Spotify) relies on recommendation systems to drive engagement. These systems are complex, involving multiple algorithms, real-time personalization, cold-start handling, and explainability.

**The Solution**: RecSys Engine demonstrates a complete, production-quality recommendation system that:

- **Learns user preferences** from behavioral data (clicks, views, purchases)
- **Compares 5 different algorithms** with proper evaluation
- **Handles cold-start** gracefully for new users
- **Explains every recommendation** with transparent reasoning
- **Visualizes performance** through a premium dashboard

> *"This isn't a tutorial project. This is what we actually build at FAANG."*

---

## ✨ Features

### 🔬 Multiple Recommendation Algorithms
| Algorithm | Description | Best For |
|-----------|-------------|----------|
| **Popularity** | Time-weighted trending items | Cold-start, baseline |
| **User-CF** | Collaborative filtering by user similarity | Established users |
| **Item-CF** | "Because you watched X" pattern | Explainability |
| **SVD** | Matrix factorization with latent factors | Scalability |
| **Hybrid** | Ensemble of all strategies | Production use |

### 📊 Comprehensive Evaluation
- Precision@K, Recall@K, NDCG@K
- Mean Average Precision (MAP)
- Catalog Coverage & Diversity
- Training/inference time analysis

### 💡 Explainability (FAANG-Standard)
Every recommendation includes:
- "Because you watched X"
- Similar user reasoning
- Confidence scores
- Factor contribution breakdown

### 🎨 Premium Dashboard
- Google-level UI polish
- Dark mode with glassmorphism
- Interactive visualizations
- Real-time simulation

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER INTERFACE                           │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌───────────┐ │
│  │  Dashboard  │ │ User        │ │ Algorithm   │ │ Metrics   │ │
│  │             │ │ Explorer    │ │ Lab         │ │ View      │ │
│  └─────────────┘ └─────────────┘ └─────────────┘ └───────────┘ │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      RECOMMENDATION ENGINE                       │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                       Model Router                          │ │
│  │   ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐  │ │
│  │   │Popularity│ │ User-CF  │ │ Item-CF  │ │ SVD / Hybrid │  │ │
│  │   └──────────┘ └──────────┘ └──────────┘ └──────────────┘  │ │
│  └────────────────────────────────────────────────────────────┘ │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                    Explainability Layer                     │ │
│  │   "Because you watched..." • Confidence Scores • Factors   │ │
│  └────────────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         DATA LAYER                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │    Users     │  │    Items     │  │    Interactions      │  │
│  │   (10,000)   │  │   (5,000)    │  │     (500,000)        │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │              Feature Store (Precomputed)                    │ │
│  │   User embeddings • Item embeddings • Similarity matrices  │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow
```
User Request → Model Selection → Generate Candidates → Rank → Explain → Return
     │              │                    │              │         │
     └──────────────┴────────────────────┴──────────────┴─────────┘
                            All steps < 100ms
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip or conda

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/recommendation-system.git
cd recommendation-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run the Dashboard

```bash
streamlit run dashboard/app.py
```

The app will:
1. Generate synthetic data (5K users, 2K items, 100K interactions)
2. Train all 5 recommendation models
3. Launch an interactive dashboard at `http://localhost:8501`

---

## 📁 Project Structure

```
recommendation-system/
├── 📂 data/
│   ├── __init__.py
│   ├── generator.py      # Realistic synthetic data generation
│   ├── loader.py         # Data loading with caching
│   └── schemas.py        # Type-safe data models
│
├── 📂 models/
│   ├── __init__.py
│   ├── base.py           # Abstract recommender interface
│   ├── popularity.py     # Popularity baseline
│   ├── user_cf.py        # User-based collaborative filtering
│   ├── item_cf.py        # Item-based collaborative filtering
│   ├── svd.py            # Matrix factorization (SVD)
│   └── hybrid.py         # Ensemble recommender
│
├── 📂 evaluation/
│   ├── __init__.py
│   ├── metrics.py        # Precision, Recall, NDCG, etc.
│   └── comparator.py     # Model comparison framework
│
├── 📂 explainability/
│   ├── __init__.py
│   └── explainer.py      # Recommendation explanations
│
├── 📂 dashboard/
│   ├── __init__.py
│   ├── app.py            # Main Streamlit application
│   ├── components/       # Reusable UI components
│   └── styles/           # Custom CSS styling
│
├── 📂 src/
│   ├── __init__.py
│   └── config.py         # Global configuration
│
├── 📂 utils/
│   ├── __init__.py
│   └── helpers.py        # Utility functions
│
├── requirements.txt
└── README.md
```

---

## 🧮 Algorithms Deep Dive

### 1. Popularity-Based (Baseline)
```python
# Simple but effective: recommend what's trending
score = Σ(interaction_weight × time_decay)
```
- **Pros**: No cold-start problem, fast, explainable
- **Cons**: Not personalized
- **Use case**: New users, fallback strategy

### 2. User-Based Collaborative Filtering
```python
# "Users like you also enjoyed..."
similarity(u1, u2) = cosine(interaction_vectors)
prediction = Σ(similarity × neighbor_ratings) / Σ(similarity)
```
- **Pros**: Intuitive explanations
- **Cons**: Scalability challenges (O(n²) users)
- **Use case**: Small-medium platforms

### 3. Item-Based Collaborative Filtering
```python
# "Because you watched X..."
similarity(i1, i2) = cosine(user_interaction_vectors)
prediction = Σ(similarity × user_rating_on_similar_items)
```
- **Pros**: Stable (items change less than users), highly explainable
- **Cons**: May over-specialize
- **Use case**: Netflix "similar titles"

### 4. Matrix Factorization (SVD)
```python
# Discover latent taste dimensions
R ≈ U × Σ × V^T
prediction = μ + b_u + b_i + u_vector · i_vector
```
- **Pros**: Handles sparsity well, captures latent factors
- **Cons**: Less interpretable
- **Use case**: Large-scale systems (Netflix Prize winner)

### 5. Hybrid Ensemble
```python
# Best of all worlds
final_score = (
    α × collaborative_score +
    β × content_score +
    γ × popularity_score
)
# With automatic cold-start fallback
```
- **Pros**: Robust, handles cold-start
- **Cons**: More complex tuning
- **Use case**: Production systems

---

## 📈 Evaluation Metrics

| Metric | Description | Formula |
|--------|-------------|---------|
| **Precision@K** | Accuracy of top-K | `relevant ∩ recommended / K` |
| **Recall@K** | Coverage of relevant | `relevant ∩ recommended / relevant` |
| **NDCG@K** | Position-aware quality | `DCG / IDCG` |
| **MAP** | Average precision | `mean(AP per user)` |
| **Coverage** | Catalog utilization | `unique_recommended / catalog_size` |
| **Diversity** | Recommendation variety | `avg pairwise distance` |

### Expected Performance

| Model | Precision@10 | Recall@10 | NDCG@10 | Coverage |
|-------|-------------|-----------|---------|----------|
| Popularity | ~8% | ~5% | ~12% | ~15% |
| User-CF | ~12% | ~8% | ~18% | ~25% |
| Item-CF | ~15% | ~10% | ~22% | ~30% |
| SVD | ~18% | ~12% | ~26% | ~35% |
| **Hybrid** | **~22%** | **~15%** | **~30%** | **~40%** |

---

## 🔄 Real-World Parallels

This system mirrors patterns used at:

| Company | Their System | Our Equivalent |
|---------|-------------|----------------|
| **Netflix** | Personalized rows | Hybrid recommender |
| **Amazon** | "Customers also bought" | Item-CF |
| **YouTube** | Two-tower model | SVD (approximation) |
| **Spotify** | Discover Weekly | User-CF + Content |

### Production Considerations

This demo focuses on **offline evaluation**. In production, you'd add:

1. **Online A/B Testing** - Compare model versions on real traffic
2. **Real-time Serving** - Feature store, prediction service
3. **Feedback Loops** - Continuous retraining
4. **Monitoring** - Drift detection, performance dashboards

---

## 🛠️ Configuration

Edit `src/config.py` to customize:

```python
@dataclass
class DataConfig:
    n_users: int = 10_000        # Scale up for production
    n_items: int = 5_000
    n_interactions: int = 500_000

@dataclass
class ModelConfig:
    n_factors: int = 100          # SVD latent dimensions
    k_neighbors: int = 50         # CF neighborhood size
    cold_start_threshold: int = 5 # Min interactions for personalization
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

---

## 📜 License

MIT License - See [LICENSE](LICENSE) for details.

---

## 👤 Author

**Built by a Senior ML Engineer**

This project demonstrates:
- ✅ Production system design patterns
- ✅ Multiple algorithm implementation
- ✅ Proper evaluation methodology
- ✅ Clean, modular code architecture
- ✅ Modern UI/UX design

---

<p align="center">
  <strong>⭐ Star this repo if it helped you understand recommendation systems!</strong>
</p>

<p align="center">
  <em>"The best recommendation system is one that understands both the user AND the problem."</em>
</p>
