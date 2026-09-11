#!/usr/bin/env python3
"""Clean git history by removing API keys from .env.example"""

import subprocess
import os
import sys

os.chdir(r'c:\Users\varun\Downloads\hiver-support-agent')

# Use git filter-branch via subprocess
cmd = [
    'git', 'filter-branch', '--force', '--tree-filter',
    'python -c "import re; p = \'.env.example\'; open(p).read() if not open(p).read() else open(p, \'w\').write(re.sub(r\'GEMINI_API_KEY=.*\', \'GEMINI_API_KEY=your_gemini_api_key_here\', open(p).read())) if open(p).read() else None"',
    '--', '--all'
]

# Simpler approach using sed via git
result = subprocess.run(
    ['git', 'filter-branch', '--force', '--tree-filter', 
     'sed -i.bak "s/^GEMINI_API_KEY=.*/GEMINI_API_KEY=your_gemini_api_key_here/" .env.example || true',
     '--', '--all'],
    capture_output=True,
    text=True
)

print("STDOUT:")
print(result.stdout)
print("\nSTDERR:")
print(result.stderr)
print("\nReturn code:", result.returncode)

# Force push
print("\n" + "="*50)
print("Pushing to GitHub...")
result2 = subprocess.run(
    ['git', 'push', '-u', 'origin', 'main', '--force'],
    capture_output=True,
    text=True
)

print("STDOUT:")
print(result2.stdout)
print("\nSTDERR:")
print(result2.stderr)
print("\nReturn code:", result2.returncode)
