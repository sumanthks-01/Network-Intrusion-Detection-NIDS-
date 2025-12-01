# NIDS Deployment Guide

## 🚀 Deploy to Render

### Prerequisites
1. GitHub account
2. Render account (free tier available)
3. Supabase project setup

### Step 1: Prepare Repository
```bash
# Initialize git repository
git init
git add .
git commit -m "Initial commit"

# Push to GitHub
git remote add origin https://github.com/yourusername/nids-system.git
git push -u origin main
```

### Step 2: Supabase Setup
Create these tables in your Supabase project:

**Users Table:**
```sql
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    password VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);
```

**Detections Table:**
```sql
CREATE TABLE detections (
    id SERIAL PRIMARY KEY,
    prediction VARCHAR(255),
    score FLOAT,
    meta JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);
```

### Step 3: Deploy to Render
1. Connect your GitHub repository to Render
2. Use the `render.yaml` configuration
3. Set environment variables:
   - `SUPABASE_URL`: Your Supabase project URL
   - `SUPABASE_ANON_KEY`: Your Supabase anon key
   - `SUPABASE_SERVICE_ROLE_KEY`: Your Supabase service role key

### Step 4: Access Your Application
- Your app will be available at: `https://your-app-name.onrender.com`
- Login/Signup functionality with database storage
- Real-time detection dashboard
- Mock detection demo

## 🔧 Environment Variables
```
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_ANON_KEY=your-anon-key
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key
MODEL_PATH=ids_model.pkl
```

## 📱 Features Available Online
- ✅ User Authentication (stored in database)
- ✅ Real-time Dashboard
- ✅ Mock Attack Detection
- ✅ Attack Logging to Database
- ✅ Statistics and Charts
- ✅ Responsive Design

## 🛠️ Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
cp backend/.env.example backend/.env
# Edit .env with your Supabase credentials

# Run locally
python start_complete_system.py
```

## 📊 Database Schema
The system uses two main tables:
- `users`: Authentication data
- `detections`: Attack detection logs

All mock detections and real detections are stored with timestamps, attack types, and metadata.