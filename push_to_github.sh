#!/bin/bash

# 🚀 Quick Push to GitHub Script
# This script helps you push Deep-PPG-Research-Core to GitHub

echo "🚀 Pushing Deep-PPG-Research-Core to GitHub..."
echo ""

# Check if git is installed
if ! command -v git &> /dev/null; then
    echo "❌ Git is not installed. Please install git first."
    exit 1
fi

# Initialize git if not already initialized
if [ ! -d .git ]; then
    echo "📦 Initializing git repository..."
    git init
    git branch -M main
fi

# Add all files
echo "➕ Adding files to git..."
git add .

# Commit
echo "💾 Committing changes..."
read -p "Enter commit message (default: 'Initial commit - Deep PPG Research Core'): " commit_msg
commit_msg=${commit_msg:-"Initial commit - Deep PPG Research Core"}
git commit -m "$commit_msg"

# Add remote (if not exists)
if ! git remote | grep -q origin; then
    echo ""
    echo "🔗 Setting up GitHub remote..."
    echo "Please go to https://github.com/new and create a new repository named:"
    echo "   Deep-PPG-Research-Core"
    echo ""
    read -p "Enter your GitHub repository URL (e.g., https://github.com/BuccKyy/Deep-PPG-Research-Core.git): " repo_url
    
    if [ -z "$repo_url" ]; then
        echo "❌ No URL provided. Exiting."
        exit 1
    fi
    
    git remote add origin "$repo_url"
fi

# Push to GitHub
echo ""
echo "⬆️  Pushing to GitHub..."
git push -u origin main

echo ""
echo "✅ Done! Your project is now on GitHub!"
echo "🌐 Visit: https://github.com/YOUR_USERNAME/Deep-PPG-Research-Core"
