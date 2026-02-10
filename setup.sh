#!/bin/bash

################################################################################
# Task 3: Dashboard Setup Script
# Brent Oil Price Analysis - Interactive Dashboard
#
# This script sets up both the Flask backend and React frontend
################################################################################

echo "================================================================================"
echo "TASK 3: INTERACTIVE DASHBOARD SETUP"
echo "Brent Oil Price Analysis"
echo "================================================================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "Step 1: Setting up Backend (Flask)"
echo "================================================================================"
echo ""

# Create backend directory
mkdir -p dashboard/backend
cd dashboard/backend || exit 1

# Create app.py placeholder
if [ ! -f "app.py" ]; then
    echo -e "${YELLOW}⚠ app.py not found${NC}"
    echo "Please copy backend_app.py to dashboard/backend/app.py"
    echo ""
fi

# Create requirements.txt
cat > requirements.txt << 'EOF'
Flask==3.0.0
flask-cors==4.0.0
pandas>=2.0.0
numpy>=1.24.0
python-dotenv==1.0.0
EOF

echo -e "${GREEN}✓ Backend requirements.txt created${NC}"

# Install backend dependencies
echo ""
echo "Installing backend dependencies..."
if command -v pip &> /dev/null; then
    pip install -r requirements.txt
    echo -e "${GREEN}✓ Backend dependencies installed${NC}"
else
    echo -e "${YELLOW}⚠ pip not found. Please install manually:${NC}"
    echo "  pip install -r requirements.txt"
fi

cd ../..

echo ""
echo "Step 2: Setting up Frontend (React)"
echo "================================================================================"
echo ""

cd dashboard || exit 1

# Create frontend with create-react-app
if [ ! -d "frontend" ]; then
    echo "Creating React app..."
    if command -v npx &> /dev/null; then
        npx create-react-app frontend
        echo -e "${GREEN}✓ React app created${NC}"
    else
        echo -e "${RED}Error: npx not found${NC}"
        echo "Please install Node.js and npm first"
        exit 1
    fi
else
    echo "✓ Frontend directory already exists"
fi

cd frontend || exit 1

# Install recharts
echo ""
echo "Installing recharts..."
if command -v npm &> /dev/null; then
    npm install recharts
    echo -e "${GREEN}✓ Recharts installed${NC}"
else
    echo -e "${YELLOW}⚠ npm not found${NC}"
fi

# Create components directory
mkdir -p src/components

echo ""
echo -e "${GREEN}✓ Frontend structure created${NC}"

cd ../..

echo ""
echo "================================================================================"
echo "SETUP COMPLETE!"
echo "================================================================================"
echo ""
echo "Next steps:"
echo ""
echo "1. Copy backend files:"
echo "   - backend_app.py → dashboard/backend/app.py"
echo ""
echo "2. Copy frontend files to dashboard/frontend/src/:"
echo "   - frontend_App.js → App.js"
echo "   - frontend_App.css → App.css"
echo "   - frontend_index.js → index.js"
echo "   - frontend_index.css → index.css"
echo ""
echo "3. Copy frontend components to dashboard/frontend/src/components/:"
echo "   - frontend_PriceChart.js → components/PriceChart.js"
echo "   - frontend_StatsCards.js → components/StatsCards.js"
echo "   - frontend_StatsCards.css → components/StatsCards.css"
echo "   - frontend_EventsPanel.js → components/EventsPanel.js"
echo "   - frontend_EventsPanel.css → components/EventsPanel.css"
echo "   - frontend_VolatilityChart.js → components/VolatilityChart.js"
echo "   - frontend_EventCorrelation.js → components/EventCorrelation.js"
echo "   - frontend_DateRangeFilter.js → components/DateRangeFilter.js"
echo "   - frontend_DateRangeFilter.css → components/DateRangeFilter.css"
echo ""
echo "4. Copy public files to dashboard/frontend/public/:"
echo "   - frontend_index.html → public/index.html"
echo ""
echo "5. Start the backend:"
echo "   cd dashboard/backend"
echo "   python app.py"
echo ""
echo "6. Start the frontend (in a new terminal):"
echo "   cd dashboard/frontend"
echo "   npm start"
echo ""
echo "7. Open http://localhost:3000 in your browser"
echo ""
echo "================================================================================"
