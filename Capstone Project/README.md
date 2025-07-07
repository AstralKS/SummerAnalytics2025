# 🚗 Dynamic Parking Pricing Models

A comprehensive implementation of two dynamic parking pricing models using urban parking data to optimize parking fees based on real-time demand, traffic conditions, and various contextual factors.

## 📋 Table of Contents
- [Overview](#overview)
- [Models](#models)
- [Dataset](#dataset)
- [Model Performance](#model-performance)
- [Key Features](#key-features)
- [Technical Implementation](#technical-implementation)
- [Results](#results)
- [Contributing](#contributing)

## 🎯 Overview

This project implements two distinct approaches to dynamic parking pricing:

1. **Model 1**: A simple linear formula-based baseline model
2. **Model 2**: A complex mathematical demand function with machine learning enhancement

Both models aim to dynamically adjust parking prices based on:
- 🚗 **Occupancy levels** - Current parking utilization
- 📊 **Queue length** - Number of vehicles waiting
- 🚦 **Traffic conditions** - Real-time traffic data
- 📅 **Special events** - Holiday and event surcharges
- 🚙 **Vehicle types** - Different pricing for cars, bikes, trucks
- ⏰ **Time patterns** - Peak hour and weekend adjustments

## 🧮 Models

### Model 1: Linear Baseline Model
```python
price = base_price + alpha * occupancy_ratio + queue_weight * queue_length
```
- **Simplicity**: Clean, interpretable linear relationship
- **Parameters**: Base price = $10, Alpha = 8.0, Queue weight = 0.5
- **Performance**: MAE ~0.0 (perfect fit to linear target)

### Model 2: Complex Dynamic Model
```python
def calculate_demand_factor(occupancy_ratio, queue_length, traffic_level, 
                           special_day, vehicle_type, hour, day_of_week):
    
    return bounded_demand_factor
```
- **Sophistication**: Multi-factor demand calculation with LightGBM
- **Bounded**: Prices constrained between 0.5x and 2x base price
- **Performance**: RMSE ~0.1 with high accuracy

## 📊 Dataset

The model uses urban parking data with the following features:
- `Occupancy` / `Capacity` - Parking utilization metrics
- `QueueLength` - Number of vehicles waiting
- `TrafficConditionNearby` - Traffic levels (low/average/high)
- `VehicleType` - Type of vehicle (car/bike/truck)
- `IsSpecialDay` - Special event indicator
- `LastUpdatedTime` - Timestamp for temporal patterns
- `LastUpdatedDate` - Date for day-of-week analysis

## 📈 Model Performance

### Model 1 Results
- **MAE**: ~0.0000 (perfect linear fit)
- **Price Range**: $10.00 - $24.00
- **Computational Speed**: ⚡ Instant
- **Interpretability**: 🔥 Excellent

### Model 2 Results
- **RMSE**: ~0.1000
- **Price Range**: $5.00 - $20.00 (bounded)
- **Feature Importance**: Demand factor > Occupancy ratio > Queue length
- **Computational Speed**: ⚡ Fast (LightGBM optimized)
- **Interpretability**: 📊 Moderate

## 🔧 Key Features

### Model 1 Features
- ✅ **Simplicity**: Easy to understand and implement
- ✅ **Fast**: Instant calculation
- ✅ **Baseline**: Perfect comparison benchmark
- ✅ **Linear**: Direct relationship interpretation

### Model 2 Features
- ✅ **Advanced Mathematics**: Complex demand function
- ✅ **Multi-factor**: Considers 8+ variables
- ✅ **Time-aware**: Hourly and daily patterns
- ✅ **Bounded**: Safe price constraints (0.5x - 2x)
- ✅ **Robust**: Handles missing data gracefully
- ✅ **ML-enhanced**: LightGBM for pattern recognition

## 💻 Technical Implementation

### Model 1 Architecture
```
Input Features → Linear Formula → Price Output
    ↓
Occupancy Ratio + Queue Length → BASE_PRICE + α*occupancy + β*queue
```

### Model 2 Architecture
```
Input Features → Feature Engineering → Demand Function → LightGBM → Price Output
    ↓              ↓                   ↓                ↓
8 Raw Features → Time/Category →    Mathematical →    ML Model →  Final Price
                 Encoding           Demand Function   Prediction   (Bounded)
```

### Key Algorithms
- **Linear Regression**: Model 1 baseline
- **Mathematical Demand Modeling**: Complex multi-factor function
- **LightGBM**: Gradient boosting for Model 2
- **Sigmoid/Tanh**: Non-linear transformations
- **Feature Engineering**: Temporal and categorical encoding

## 📊 Results

### Price Distribution Comparison
- **Model 1**: Linear distribution following occupancy patterns
- **Model 2**: Natural distribution with smart bounds and peaks

### Temporal Patterns
- **Peak Hours**: 7-9 AM and 5-7 PM show price increases
- **Weekend Effect**: ~5% price increase on weekends
- **Special Days**: Up to 20% surge pricing

### Vehicle Type Pricing
- **Cars**: Base pricing (1.0x multiplier)
- **Bikes**: Discounted (~0.7x multiplier)
- **Trucks**: Premium (~1.3x multiplier)

## 🎯 Key Insights

1. **Occupancy Ratio** is the primary driver of parking prices
2. **Queue Length** creates exponential price increases during congestion
3. **Time patterns** show clear peak hour surcharges
4. **Traffic conditions** amplify pricing effects during high-demand periods
5. **Vehicle differentiation** enables equitable pricing across vehicle types

## 🔄 Model Comparison

| Aspect | Model 1 (Linear) | Model 2 (Complex) |
|--------|------------------|-------------------|
| **Complexity** | Simple | Advanced |
| **Accuracy** | Perfect (linear target) | High (RMSE ~0.1) |
| **Speed** | Instant | Fast |
| **Interpretability** | Excellent | Moderate |
| **Real-world applicability** | Basic | Production-ready |
| **Features used** | 2 | 8+ |
| **Price bounds** | None | Smart bounds |

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Summer Analytics Team for this Capstone Project 

---

