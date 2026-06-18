#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Agentic Document Intelligence — Setup Script
# =============================================================================
# One-time setup: pulls ML models and builds the Docker images.
# Run this once, then use `docker compose up -d` to start.
# =============================================================================

CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${CYAN}╔══════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║   Agentic Document Intelligence Setup           ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════╝${NC}"
echo ""

# ── Check prerequisites ──────────────────────────────────────────────────────
echo -e "${YELLOW}[1/4] Checking prerequisites...${NC}"

if ! command -v docker &>/dev/null; then
  echo "ERROR: Docker is not installed."
  echo "Install it first: https://docs.docker.com/engine/install/"
  exit 1
fi

if ! docker compose version &>/dev/null; then
  echo "ERROR: Docker Compose is not available."
  echo "Install it first: https://docs.docker.com/compose/install/"
  exit 1
fi

# Check available memory (Linux only)
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
  total_mem=$(free -g 2>/dev/null | awk '/^Mem:/{print $2}' || echo "0")
  if [ "$total_mem" -lt 8 ]; then
    echo -e "${YELLOW}WARNING: Less than 8 GB RAM detected ($total_mem GB).${NC}"
    echo "  The pipeline needs at least 8 GB, 16 GB recommended."
    echo ""
  fi
fi

echo -e "${GREEN}  ✓ Docker found${NC}"
echo ""

# ── Copy env file ────────────────────────────────────────────────────────────
echo -e "${YELLOW}[2/4] Setting up environment...${NC}"
if [ ! -f .env ]; then
  cp .env.example .env
  echo -e "${GREEN}  ✓ Created .env from .env.example${NC}"
else
  echo -e "${GREEN}  ✓ .env already exists${NC}"
fi
echo ""

# ── Pull ML models ──────────────────────────────────────────────────────────
echo -e "${YELLOW}[3/4] Pulling ML models (this downloads ~8 GB, may take a while)...${NC}"
echo -e "  Extraction model: ${CYAN}qwen2.5:7b-instruct-q4_K_M${NC} (~4.7 GB)"
echo -e "  VLM OCR model:    ${CYAN}gemma3:4b${NC} (~3.2 GB)"
echo ""

docker compose --profile setup up model-puller

echo -e "${GREEN}  ✓ Models pulled${NC}"
echo ""

# ── Build app image ─────────────────────────────────────────────────────────
echo -e "${YELLOW}[4/4] Building application image...${NC}"
docker compose build app

echo -e "${GREEN}  ✓ App image built${NC}"
echo ""

# ── Done ─────────────────────────────────────────────────────────────────────
echo -e "${CYAN}╔══════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║   Setup complete!                               ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════╝${NC}"
echo ""
echo "Start the application:"
echo "  docker compose up -d"
echo ""
echo "Then open:"
echo "  http://localhost:8000"
echo ""
echo "To stop:"
echo "  docker compose down"
echo ""
echo "To view logs:"
echo "  docker compose logs -f app"
