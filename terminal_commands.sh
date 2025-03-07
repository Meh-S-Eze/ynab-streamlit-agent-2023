#!/bin/bash

# Check current git status
echo "Checking git status..."
git status

# Fetch updates from remote
echo -e "\nFetching updates from remote repository..."
git fetch --all

# Check for any differences between local and remote
echo -e "\nChecking for differences between local and remote branches..."
git branch -v

# Pull any changes from remote
echo -e "\nPulling any changes from remote repository..."
git pull

# Push any local commits to remote
echo -e "\nPushing any local commits to remote repository..."
git push

echo -e "\nGit repository update complete!" 