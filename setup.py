#!/usr/bin/env python3
"""
Setup Script for Agentic Document Intelligence System
Automates environment setup and dependency installation
"""

import os
import subprocess
import sys
from pathlib import Path


def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        print("Error: Python 3.10 or higher is required")
        print(f"   Current version: {version.major}.{version.minor}")
        return False
    print(f"Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True


def install_dependencies():
    """Install Python dependencies"""
    print("\nInstalling dependencies...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print("Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to install dependencies: {e}")
        return False


def create_directories():
    """Create necessary directories"""
    print("\nCreating directories...")
    directories = [
        "data/documents",
        "data/demo_documents",
        "logs",
        "output/ingestion",
        "output/analysis",
        "output/reasoning",
        "output/responses",
        "output/sessions"
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"   Created: {directory}")

    print("All directories created")
    return True


def setup_environment():
    """Setup environment file"""
    print("\nSetting up environment...")

    env_example = Path(".env.example")
    env_file = Path(".env")

    if env_example.exists() and not env_file.exists():
        # Copy example to .env
        with open(env_example) as src, open(env_file, 'w') as dst:
            content = src.read()
            dst.write(content)
        print("Environment file created from template")
        print("Please edit .env file with your Google AI API key")
        return True
    elif env_file.exists():
        print("Environment file already exists")
        return True
    else:
        print("No environment template found")
        return False


def verify_installation():
    """Verify installation by running basic imports"""
    print("\n🧪 Verifying installation...")

    try:
        # Test basic imports
        import asyncio
        import pathlib

        import yaml
        print("Core dependencies verified")

        # Test Google AI import (if available)
        try:
            import google.generativeai
            print("Google AI SDK available")
        except ImportError:
            print("Google AI SDK not found - install with: pip install google-generativeai")

        return True

    except ImportError as e:
        print(f"Import error: {e}")
        return False


def check_api_key():
    """Check if API key is configured"""
    print("\n🔑 Checking API key configuration...")

    # Check environment variable
    api_key = os.getenv('GOOGLE_AI_API_KEY')
    if api_key:
        print("GOOGLE_AI_API_KEY environment variable found")
        return True

    # Check .env file
    env_file = Path(".env")
    if env_file.exists():
        with open(env_file) as f:
            content = f.read()
            if 'GOOGLE_AI_API_KEY=' in content and 'your-google-ai-api-key' not in content:
                print("API key found in .env file")
                return True

    print("Google AI API key not configured")
    print("   Please set GOOGLE_AI_API_KEY in .env file or environment variable")
    print("   Get your API key from: https://ai.google.dev/")
    return False


def run_demo_check():
    """Check if demo can run"""
    print("\nDemo readiness check...")

    try:
        # Check if main modules can be imported
        sys.path.insert(0, str(Path.cwd()))

        from pipeline.config import PipelineConfig
        from utils.document_processor import DocumentProcessor
        print("Core modules can be imported")

        return True

    except ImportError as e:
        print(f"Module import error: {e}")
        return False


def main():
    """Main setup function"""
    print("Agentic Document Intelligence System - Setup")
    print("=" * 60)

    success_steps = 0
    total_steps = 7

    # Step 1: Check Python version
    if check_python_version():
        success_steps += 1

    # Step 2: Install dependencies
    if install_dependencies():
        success_steps += 1

    # Step 3: Create directories
    if create_directories():
        success_steps += 1

    # Step 4: Setup environment
    if setup_environment():
        success_steps += 1

    # Step 5: Verify installation
    if verify_installation():
        success_steps += 1

    # Step 6: Check API key
    if check_api_key():
        success_steps += 1

    # Step 7: Demo readiness
    if run_demo_check():
        success_steps += 1

    # Final status
    print(f"\nSetup Results: {success_steps}/{total_steps} steps completed")

    if success_steps == total_steps:
        print("\nSetup completed successfully!")
        print("\nNext Steps:")
        print("   1. Start the app: docker compose up -d")
        print("   2. Open http://localhost:8000")
        print("   3. Read README.md for detailed documentation")

    elif success_steps >= total_steps - 1:
        print("\nSetup mostly complete with minor issues")
        print("   Please review the warnings above")
        print("   System should still be functional")

    else:
        print("\nSetup encountered significant issues")
        print("   Please resolve the errors above before proceeding")
        print("   Check README.md for troubleshooting guide")

    print("\n📖 Full documentation: README.md")
    print("🆘 Support: Open an issue on GitHub")


if __name__ == "__main__":
    main()
